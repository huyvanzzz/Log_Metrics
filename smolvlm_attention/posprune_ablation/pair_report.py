"""Export per-question correlations and paired output changes."""

import json
from pathlib import Path
from statistics import mean, median

import numpy as np

from smolvlm_attention.plotting import _STYLE, _export, plt
from .report import _csv


def _median(values):
    return median(values) if values else None


def report_pairs(output):
    output = Path(output)
    config = json.loads((output / 'config.json').read_bytes())
    with (output / 'results.jsonl').open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    if [row['sample_id'] for row in rows] != config['sample_ids']:
        raise ValueError('The merged run is incomplete or has different question IDs')
    layers, groups = config['layers'], config['groups']
    per_question = [dict(sample_id=row['sample_id'], question_type=row['question_type'], layer=layer,
                         **row['correlations'][str(layer)]) for row in rows for layer in layers]
    _csv(output / 'correlation_by_question.csv', per_question)
    correlation = []
    distributions = {'pearson': [], 'spearman': []}
    for layer in layers:
        summary = dict(layer=layer, questions=len(rows))
        entries = [row for row in per_question if row['layer'] == layer]
        for metric, values_by_layer in distributions.items():
            values = [item[metric] for item in entries if item[metric] is not None]
            values_by_layer.append(values)
            summary[f'{metric}_count'] = len(values)
            summary[f'{metric}_median'] = _median(values)
            summary[f'{metric}_mean'] = mean(values) if values else None
        correlation.append(summary)
    _csv(output / 'correlation_by_layer.csv', correlation)

    ablations = [dict(sample_id=row['sample_id'], question_type=row['question_type'], group=group,
                      baseline=row['baseline'], **result)
                 for row in rows for group, result in row['ablations'].items()]
    if ablations:
        _csv(output / 'ablation_by_question.csv', ablations)
    summaries = []
    for group in groups:
        values = [row for row in ablations if row['group'] == group]
        summaries.append(dict(group=group, questions=len(values), skipped_questions=len(rows) - len(values),
            median_first_token_jsd=_median([row['first_token_jsd'] for row in values]),
            first_token_changed_pct=100 * mean(row['first_token_changed'] for row in values) if values else None,
            answer_changed_pct=100 * mean(row['answer_changed'] for row in values) if values else None,
            median_dropped_raw_attention=_median([row['dropped_raw_attention'] for row in values]),
            same_camera_pct=100 * mean(row['dropped_camera'] == row['kept_camera'] for row in values) if values else None))
    _csv(output / 'ablation_by_group.csv', summaries)
    return _plot(output, layers, distributions, summaries, config['drop_after_layer'])


def _plot(output, layers, distributions, summaries, drop_layer):
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4.5), layout='constrained')
        for ax, (metric, values) in zip(axes, distributions.items()):
            ax.boxplot([items or [np.nan] for items in values], showfliers=False)
            ax.axhline(0, color='0.6', linewidth=1)
            ax.set(xticks=range(1, len(layers) + 1), xticklabels=layers, xlabel='Decoder layer',
                   title=metric.capitalize(), ylim=(-1.02, 1.02))
        axes[0].set_ylabel('Per-question correlation')
        fig.suptitle('Hidden-state cosine vs. incoming-attention cosine')
        paths = _export(fig, output, 'pair_correlations', 180)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
        labels = [row['group'].replace('_hidden_', ' hidden\n').replace('_attention', ' attention') for row in summaries]
        for ax, (key, label) in zip(axes, [('median_first_token_jsd', 'Median first-token JSD (nats)'),
                                          ('answer_changed_pct', 'Changed answers (%)')]):
            values = [row[key] if row[key] is not None else np.nan for row in summaries]
            ax.bar(labels, values, color='#4677a2')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', labelsize=9)
        fig.suptitle(f'Remove one visual token after layer {drop_layer}')
        return paths + _export(fig, output, 'single_token_ablation', 180)
