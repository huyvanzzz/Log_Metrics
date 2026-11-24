"""Paired answer evaluation and publication plots for the configured variants."""

import csv
from io import StringIO
import json
import math

from smolvlm_attention.mvpruner_ablation.data import atomic_output, file_sha256, write_json

from .design import DEEP, MERGE_SEEDS, MIDDLE
from .runtime import _load_frame
from .suites import DEEP_LAYERS, MIDDLE_CONTROLS, MIDDLE_FULL_RANGE, suite_variants


def _yes_no(text):
    return text.strip().casefold().rstrip('.!')


def _import_baseline(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        matches = [row for row in csv.DictReader(stream) if row.get('variant', '').strip() == 'baseline']
    if len(matches) != 1:
        raise ValueError('The deep score file must contain exactly one baseline row')
    row = matches[0]
    metrics = {key: float(row[key]) for key in ('bleu4', 'meteor', 'rouge_l', 'cider', 'overall', 'yes_no_accuracy')}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError('The imported baseline contains a non-finite metric')
    expected = (metrics['bleu4'] + metrics['meteor'] + metrics['rouge_l'] + metrics['cider'] / 10) / 4
    if not math.isclose(metrics['overall'], expected, abs_tol=1e-5):
        raise ValueError('Imported Overall does not match the metric formula used by this runner')
    return dict(metrics, removed_visual_mass_share=None, fallback_share=None,
                truncated=int(row['truncated']) if row.get('truncated') else None)


def score(output, baseline_scores=None):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    manifest = json.loads((output / 'manifest.json').read_bytes())
    middle_full = manifest.get('suite') == 'middle-full'
    if middle_full != (baseline_scores is not None):
        raise ValueError('Supply --baseline-scores for the middle-full suite only')
    results = {'baseline': _import_baseline(baseline_scores)} if middle_full else {}
    variants = suite_variants(manifest)
    frames = manifest['frames']
    texts = {row['id']: [{'caption': row['answer']}] for frame in frames.values() for row in frame['rows']}
    binary = {row['id']: _yes_no(row['answer'])
              for frame in frames.values() for row in frame['rows'] if row['yes_no']}
    if not binary:
        raise ValueError('The selected questions contain no Yes/No answers')
    for variant in variants:
        if not (output / 'predictions' / variant.name / 'complete.json').is_file():
            raise ValueError(f'Complete {variant.name} before scoring the matrix')

    tokenizer = PTBTokenizer()
    references = tokenizer.tokenize(texts)
    bleu, meteor, rouge, cider = Bleu(4), Meteor(), Rouge(), Cider()
    for index, variant in enumerate(variants, 1):
        directory = output / 'predictions' / variant.name
        rows = [row for frame_id, frame in frames.items()
                for row in _load_frame(directory / f'{frame_id}.json', frame['rows'])]
        if len(rows) != len(texts):
            raise ValueError(f'Incomplete predictions for {variant.name}')
        predictions = {row['id']: row['prediction'] for row in rows}
        tokenized = tokenizer.tokenize({key: [{'caption': value}] for key, value in predictions.items()})
        bleu4 = float(bleu.compute_score(references, tokenized)[0][-1]) * 100
        meteor_score = float(meteor.compute_score(references, tokenized)[0]) * 100
        rouge_l = float(rouge.compute_score(references, tokenized)[0]) * 100
        cider_score = float(cider.compute_score(references, tokenized)[0]) * 100
        original = sum(row['visual_mass'] for row in rows)
        weights = sum(row['attention_rows'] for row in rows)
        results[variant.name] = dict(
            bleu4=bleu4, meteor=meteor_score, rouge_l=rouge_l, cider=cider_score,
            overall=(bleu4 + meteor_score + rouge_l + cider_score / 10) / 4,
            yes_no_accuracy=100 * sum(_yes_no(predictions[key]) == value for key, value in binary.items()) / len(binary),
            removed_visual_mass_share=sum(row['removed_mass'] for row in rows) / original if original else None,
            fallback_share=sum(row['fallback_rows'] for row in rows) / weights if weights else None,
            truncated=sum(row['truncated'] for row in rows),
        )
        print(f'{index}/{len(variants)} {variant.name} | Overall {results[variant.name]["overall"]:.3f}', flush=True)

    baseline = results['baseline']
    for result in results.values():
        result['delta_overall'] = result['overall'] - baseline['overall']
        result['delta_yes_no'] = result['yes_no_accuracy'] - baseline['yes_no_accuracy']
    payload = dict(samples=len(texts), yes_no_samples=len(binary), results=results)
    if 'suite' in manifest:
        payload['suite'] = manifest['suite']
    if baseline_scores is not None:
        payload['baseline_source'] = dict(file=baseline_scores.name, sha256=file_sha256(baseline_scores))
    write_json(output / 'scores.json', payload)
    buffer = StringIO(newline='')
    writer = csv.DictWriter(buffer, ['variant', *baseline.keys()])
    writer.writeheader()
    writer.writerows(dict(variant=label, **result) for label, result in results.items())
    with atomic_output(output / 'scores.csv') as stream:
        stream.write(buffer.getvalue().encode('utf-8'))


def plot(output):
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    from smolvlm_attention.plotting import _STYLE, _export, plt

    payload = json.loads((output / 'scores.json').read_bytes())
    results = payload['results']
    folder = output / 'figures'

    def extract(names, key='delta_overall'):
        return np.array([results[name][key] for name in names])

    shallow = ('shallow-top10', 'shallow-keep10', 'shallow-none')
    merge = tuple(f'shallow-merge-{seed}' for seed in MERGE_SEEDS)
    methods = ('top10', 'bottom10', 'keep5', 'top10-mass', 'bottom10-mass')
    middle = {f'{start}–{stop}': tuple(f'middle-{start}-{stop}-{name}' for name in methods)
              for start, stop in MIDDLE}
    middle_labels = ('Mask top 10%', 'Mask bottom 10%', 'Keep top 5%', 'Mask top 10%\n+ preserve mass',
                     'Mask bottom 10%\n+ preserve mass')
    colors = ('#185782', '#a24d17')
    written = []

    def finish(fig, ax, name, ylabel='Δ Overall vs. baseline'):
        ax.axhline(0, color='#3b4248', linewidth=0.8)
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel)
        written.extend(_export(fig, folder, name, 300))

    def deep(layers, name, key, ylabel, modes=('skip', 'discard')):
        fig, ax = plt.subplots(figsize=(8.5, 4.6), layout='constrained')
        width = 0.7 / len(modes)
        for i, mode in enumerate(modes):
            names = [f'deep-{mode}-{layer}' for layer in layers]
            ax.bar(np.arange(len(layers)) + (i - (len(modes) - 1) / 2) * width,
                   extract(names, key), width, label=mode.capitalize(), color=colors[mode == 'discard'])
        ax.set_xticks(range(len(layers)), layers)
        ax.set_xlabel('Intervention layer')
        ax.legend()
        finish(fig, ax, name, ylabel)

    with plt.rc_context(_STYLE):
        if payload.get('suite') == 'middle-full':
            start, stop = MIDDLE_FULL_RANGE
            prefix = f'middle-{start}-{stop}-'
            for key, name, ylabel in (
                ('delta_overall', 'middle_full_preserve_mass', 'Δ Overall vs. baseline'),
                ('delta_yes_no', 'middle_full_preserve_mass_yes_no', 'Δ Yes/No accuracy vs. baseline (pp)'),
            ):
                fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
                x = np.arange(len(MIDDLE_CONTROLS))
                for i, (suffix, label) in enumerate((('', 'Mask only'), ('-mass', 'Mask + preserve mass'))):
                    names = [prefix + mode + suffix for mode, _ in MIDDLE_CONTROLS]
                    ax.bar(x + (i - 0.5) * 0.36, extract(names, key), 0.36, label=label, color=colors[i])
                ax.set_xticks(x, [title.replace('Chỉ giữ', 'Keep') for _, title in MIDDLE_CONTROLS])
                ax.set_title(f'Layers {start}–{stop}')
                ax.legend()
                finish(fig, ax, name, ylabel)
            return written

        if payload.get('suite') == 'deep-full':
            deep(DEEP_LAYERS, 'deep_discard', 'delta_overall', 'Δ Overall vs. baseline', ('discard',))
            deep(DEEP_LAYERS, 'deep_discard_yes_no', 'delta_yes_no',
                 'Δ Yes/No accuracy vs. baseline (pp)', ('discard',))
            return written

        fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
        seeds = extract(merge)
        ax.bar(range(4), [*extract(shallow), seeds.mean()], color=[colors[0]] * 3 + [colors[1]])
        ax.errorbar(3, seeds.mean(), yerr=seeds.std(ddof=1), color='#202020', capsize=5, fmt='none')
        ax.scatter(3 + np.linspace(-0.15, 0.15, len(seeds)), seeds, color='#202020', s=22, zorder=3)
        ax.set_xticks(range(4), ['Mask top 10%', 'Keep top 10%', 'Mask all', 'Merge\n3 recipient seeds'])
        finish(fig, ax, 'shallow_interventions')

        fig, ax = plt.subplots(figsize=(11, 4.6), layout='constrained')
        x = np.arange(len(methods))
        for i, (layers, names) in enumerate(middle.items()):
            ax.bar(x + (i - 0.5) * 0.35, extract(names), 0.35, label=f'Layers {layers}', color=colors[i])
        ax.set_xticks(x, middle_labels)
        ax.legend()
        finish(fig, ax, 'middle_masking_controls')

        deep(DEEP, 'deep_skip_vs_discard', 'delta_overall', 'Δ Overall vs. baseline')

        fig, ax = plt.subplots(figsize=(11, 4.6), layout='constrained')
        for i, (layers, names) in enumerate(middle.items()):
            ax.bar(x + (i - 0.5) * 0.35, extract(names, 'removed_visual_mass_share'),
                   0.35, label=f'Layers {layers}', color=colors[i])
        ax.set_xticks(x, middle_labels)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
        finish(fig, ax, 'middle_removed_visual_mass', 'Share of visual mass removed before redistribution')
    return written
