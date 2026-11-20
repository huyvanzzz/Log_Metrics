"""CPU-only COCO caption metrics; deliberately fail closed on scorer errors."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading

METRICS = ('BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr')


def _identity():
    try:
        version = importlib.metadata.version('pycocoevalcap')
        spec = importlib.util.find_spec('pycocoevalcap.meteor.meteor')
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError('Install pycocoevalcap==1.2 before scoring') from exc
    if version != '1.2' or spec is None:
        raise RuntimeError('Scoring requires pycocoevalcap==1.2')
    jar = Path(spec.origin).parent / 'meteor-1.5.jar'
    java = shutil.which('java')
    if not java or not jar.is_file():
        raise RuntimeError('Scoring requires Java and the packaged METEOR JAR')
    completed = subprocess.run([java, '-version'], capture_output=True, text=True, timeout=30, check=True)
    sources = {}
    for module in ('bleu.bleu', 'bleu.bleu_scorer', 'meteor.meteor', 'rouge.rouge', 'cider.cider', 'cider.cider_scorer'):
        source = Path(importlib.util.find_spec('pycocoevalcap.' + module).origin)
        sources[module] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    return {'pycocoevalcap_version': version, 'java_path': java,
            'java_version': (completed.stderr or completed.stdout).strip(),
            'meteor_jar': str(jar), 'meteor_jar_sha256': hashlib.sha256(jar.read_bytes()).hexdigest(),
            'source': 'efficient_vlm_ad.evaluation.caption_metrics',
            'reference_commit': '4d7dbd238c7960d305c882b75220fb95beb37644',
            'scorer_sources': sources,
            'tokenization': 'direct strings; no PTB or lowercasing',
            'meteor_sanitization': 'CR and LF replaced by spaces; ||| removed; hypothesis double-space replacement matches upstream'}


def _meteor_text(text):
    return text.replace('\r', ' ').replace('\n', ' ').replace('|||', '')


def _meteor(gts, res, identity):
    """Own the Java lifetime: no library destructor/lock, bounded protocol reads."""
    jar = Path(identity['meteor_jar'])
    command = [identity['java_path'], '-Xmx2G', '-jar', str(jar), '-', '-', '-stdio', '-l', 'en', '-norm']
    output = queue.Queue()
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(command, cwd=jar.parent, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=errors)
        def reader():
            for line in iter(proc.stdout.readline, b''):
                output.put(line)
            output.put(None)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        def send(line):
            proc.stdin.write((line + '\n').encode('utf-8'))
            proc.stdin.flush()
        def receive():
            try:
                line = output.get(timeout=120)
            except queue.Empty as exc:
                raise RuntimeError('METEOR protocol timed out') from exc
            if line is None:
                errors.seek(0)
                raise RuntimeError('METEOR exited: ' + errors.read().decode('utf-8', errors='replace'))
            return line.decode('utf-8').strip()
        try:
            stats = []
            for key in gts:
                reference = _meteor_text(gts[key][0])
                hypothesis = _meteor_text(res[key][0]).replace('  ', ' ')
                send(' ||| '.join(('SCORE', reference, hypothesis)))
                stats.append(receive())
            send('EVAL' + ''.join(' ||| ' + stat for stat in stats))
            for _ in stats:
                float(receive())
            return float(receive())
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
            proc.stdin.close()
            thread.join(timeout=10)
            proc.stdout.close()


def _compute(gts, res, identity):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    bleu, _ = Bleu(4).compute_score(gts, res)
    scores = dict(zip(METRICS[:4], map(float, bleu)))
    scores['METEOR'] = _meteor(gts, res, identity)
    scores['ROUGE-L'] = float(Rouge().compute_score(gts, res)[0])
    scores['CIDEr'] = float(Cider().compute_score(gts, res)[0])
    if set(scores) != set(METRICS) or not all(math.isfinite(v) for v in scores.values()):
        raise RuntimeError('Scorers returned incomplete or non-finite metrics')
    return scores


def score_rows(rows, expected_count=16817):
    rows = list(rows)
    if not isinstance(expected_count, int) or expected_count < 1 or len(rows) != expected_count:
        raise ValueError(f'Expected {expected_count} rows, received {len(rows)}')
    samples, evals = set(), set()
    gts, res = {}, {}
    for row in rows:
        sample, eval_id = row['sample_id'], row['eval_id']
        if not isinstance(sample, str) or not sample or isinstance(eval_id, bool) or not isinstance(eval_id, int):
            raise ValueError('Each row requires a nonempty sample_id and integer eval_id')
        if sample in samples or eval_id in evals:
            raise ValueError('Duplicate sample_id or eval_id')
        if not isinstance(row['reference'], str) or not isinstance(row['prediction'], str):
            raise ValueError('Reference and prediction must be strings')
        samples.add(sample)
        evals.add(eval_id)
        gts[eval_id] = [row['reference']]
        res[eval_id] = [row['prediction']]
    identity = _identity()
    metrics = _compute(gts, res, identity)
    diagnostics = {'prediction_count': len(rows), 'expected_count': expected_count,
                   'unique_sample_count': len(samples), 'unique_eval_count': len(evals),
                   'empty_prediction_count': sum(not row['prediction'] for row in rows),
                   'meteor_sanitized_reference_count': sum(_meteor_text(row['reference']) != row['reference'] for row in rows),
                   'meteor_sanitized_prediction_count': sum(_meteor_text(row['prediction']) != row['prediction'] for row in rows)}
    return {'metrics': metrics, 'metrics_display': {k: round(v * 100, 6) for k, v in metrics.items()},
            'diagnostics': diagnostics, 'scoring_identity': identity}


def metric_preflight():
    identity = _identity()
    rows = [dict(sample_id='smoke-1', eval_id=1, reference='a red car drives along the road', prediction='a red car drives along the road'),
            dict(sample_id='smoke-2', eval_id=2, reference='a blue bus stops near a tree', prediction='a blue bus stops near a tree')]
    result = score_rows(rows, expected_count=2)
    return {**identity, 'scorer_smoke': result['metrics'], 'ok': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, default=16817)
    args = parser.parse_args(argv)
    if args.preflight:
        result = metric_preflight()
    else:
        if args.input is None:
            parser.error('--input is required unless --preflight is selected')
        with args.input.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        result = score_rows(rows, expected_count=args.expected_count)
    from smolvlm_attention.sample_store import atomic_json
    atomic_json(args.output, result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
