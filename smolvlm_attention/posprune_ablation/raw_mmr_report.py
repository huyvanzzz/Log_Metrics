"""Report paired raw-attention and raw-MMR results at matched token budgets."""

import json
from math import isqrt
from pathlib import Path

from smolvlm_attention.plotting import _STYLE, _export, plt
from .report import _csv, _summarize


LABELS = {'raw_mmr': 'Raw MMR', 'intra_camera_mmr': 'Intra-camera MMR',
          'spatial_aware_mmr': 'Spatial-aware MMR', 'attention_mmr': 'Attention-pattern MMR'}


def _add_historical(rows, historical):
    if [row['sample_id'] for row in rows] != historical['evaluation_ids']:
        raise ValueError('Historical raw_96 predictions are from a different question cohort')
    for row, saved in zip(rows, historical['predictions'], strict=True):
        if 'raw_96' in row['variants']:
            raise ValueError('A current raw_96 result would be overwritten')
        row['variants']['raw_96'] = saved


def _selections(rows, config):
    per_view = config['per_view']
    side = isqrt(per_view)
    entries = []
    for row in rows:
        for variant in config['variants']:
            result = row['variants'][variant['name']].get('selection')
            if result is None:
                continue
            raw = row['variants'][f"raw_{variant['keep']}"].get('selection')
            raw_picks = set(raw['picks']) if raw is not None else None
            weight = None if variant['method'] == 'raw' else variant.get('mmr_lambda', config.get('weight'))
            for rank, index in enumerate(result['picks'], 1):
                camera, local = divmod(index, per_view)
                y, x = divmod(local, side)
                entries.append(dict(sample_id=row['sample_id'], variant=variant['name'], source=variant['source'],
                    method=variant['method'], mmr_lambda=weight,
                    keep=variant['keep'], selection_rank=rank, visual_index=index, camera=config['cameras'][camera],
                    grid_row=y, grid_col=x, raw_score=result['raw'][index],
                    relevance=result.get('relevance', result['raw'])[index],
                    also_selected_by_raw_topk=None if raw_picks is None else index in raw_picks,
                    max_redundancy=None if rank == 1 or result['penalties'] is None else result['penalties'][rank - 1]))
    return entries


def _differences(summary):
    named = {row['variant']: row for row in summary}
    output = []
    for mmr in summary:
        keep = mmr['kept_tokens']
        method = mmr['method']
        if method == 'raw':
            continue
        raw = named[f'raw_{keep}']
        deltas = {f'delta_{key}': (None if raw[key] is None or mmr[key] is None else mmr[key] - raw[key])
                  for key in ('overall', 'bleu4', 'meteor', 'rouge_l', 'cider', 'yes_no_accuracy',
                              'median_first_token_step_ms', 'median_latency_ms')}
        output.append(dict(method=method, kept_tokens=keep, raw_source=raw['source'],
                           attention_normalization=mmr['attention_normalization'], mmr_lambda=mmr['mmr_lambda'],
                           raw_overall=raw['overall'],
                           mmr_overall=mmr['overall'], **deltas, answer_changed_pct=mmr['answer_changed_pct']))
    return output


def _plot(differences, output, *, sweep=False):
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.7) if sweep else (8, 4), layout='constrained')
        if sweep:
            series = dict.fromkeys((row['method'], row['kept_tokens']) for row in differences)
            for method, keep in series:
                points = sorted((row['mmr_lambda'], row['delta_overall']) for row in differences
                                if row['method'] == method and row['kept_tokens'] == keep)
                weights, values = zip(*points)
                ax.plot(weights, values, marker='o', label=f'{LABELS[method]} · keep {keep}')
            ax.set(xlabel='MMR λ', xticks=sorted({row['mmr_lambda'] for row in differences}))
            ax.legend()
        else:
            labels = [f"{LABELS[row['method']]}\nKeep {row['kept_tokens']}" for row in differences]
            values = [row['delta_overall'] for row in differences]
            ax.bar(labels, values, width=.5, color=['#267862' if value >= 0 else '#b56542' for value in values])
        ax.axhline(0, color='0.5', linewidth=1)
        ax.margins(y=.2)
        ax.set(title='Raw attention + MMR vs. raw top-k', ylabel='Change in Overall score')
        return _export(fig, output, 'raw_mmr_vs_raw', 180)


def report_raw_mmr(output, historical=None):
    output = Path(output)
    config = json.loads((output / 'config.json').read_bytes())
    with (output / 'predictions.jsonl').open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    if config.get('experiment') != 'raw_mmr' or [row['sample_id'] for row in rows] != config['evaluation_ids']:
        raise ValueError('Raw MMR generation must finish before reporting')
    variants = {variant['name']: variant for variant in config['variants']}
    sources = {name: variant['source'] for name, variant in variants.items()}
    current = {name for name, source in sources.items() if source == 'current'}
    if any(set(row['variants']) != current for row in rows):
        raise ValueError('A question is missing a current variant or has an unexpected variant')
    if sources.get('raw_96') == 'historical':
        if historical is None:
            raise ValueError('Historical raw_96 predictions are required for this run')
        _add_historical(rows, historical)
    summary = _summarize(rows, config)
    for row in summary:
        variant = variants[row['variant']]
        topk = variant['method'] == 'raw'
        row.update(source=sources[row['variant']],
                   method=variant['method'],
                   attention_normalization='none' if topk else config.get('attention_normalization', 'none'),
                   mmr_lambda=None if topk else variant.get('mmr_lambda', config.get('weight', 0.9)))
    differences = _differences(summary)
    _csv(output / 'comparison.csv', summary)
    _csv(output / 'mmr_vs_raw.csv', differences)
    _csv(output / 'indexed_selections.csv', _selections(rows, config))
    if config['question_type_distribution']:
        _csv(output / 'question_type_distribution.csv', config['question_type_distribution'])
    _csv(output / 'paired_answers.csv', [dict(sample_id=row['sample_id'], question=row['question'],
        question_type=row.get('question_type'), ground_truth=row['ground_truth'],
        **{name: row['variants'][name]['prediction'] for name in sources}) for row in rows])
    return _plot(differences, output, sweep=len(config.get('mmr_lambdas', ())) > 1)
