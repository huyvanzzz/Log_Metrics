"""Export calibration, indexed token selections, and paired quality and latency results."""

import csv
import json
from math import isqrt
from pathlib import Path
from statistics import mean, median

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from smolvlm_attention.plotting import _STYLE, _export, plt


def _quality(rows, names):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    def binary(text):
        return text.strip().casefold().rstrip('.!')

    tokenizer = PTBTokenizer()
    def captions(answers):
        return {i: [{'caption': answer}] for i, answer in enumerate(answers)}

    answers = [str(row['ground_truth']) for row in rows]
    references = tokenizer.tokenize(captions(answers))
    yes_no = [i for i, answer in enumerate(answers) if binary(answer) in ('yes', 'no')]
    bleu, meteor, rouge, cider = Bleu(4), Meteor(), Rouge(), Cider()
    results = {}
    for name in names:
        answers_for_model = [row['variants'][name]['prediction'] for row in rows]
        predictions = tokenizer.tokenize(captions(answers_for_model))
        bleu4 = 100 * float(bleu.compute_score(references, predictions)[0][-1])
        meteor_score = 100 * float(meteor.compute_score(references, predictions)[0])
        rouge_l = 100 * float(rouge.compute_score(references, predictions)[0])
        cider_score = 100 * float(cider.compute_score(references, predictions)[0])
        overall = (bleu4 + meteor_score + rouge_l + cider_score / 10) / 4
        accuracy = (100 * mean(binary(answers_for_model[i]) == binary(answers[i]) for i in yes_no)
                    if yes_no else None)
        results[name] = dict(bleu4=bleu4, meteor=meteor_score, rouge_l=rouge_l,
                             cider=cider_score, overall=overall, yes_no_accuracy=accuracy)
    return results


def _csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selection_rows(rows, config):
    per_view = config['per_view']
    side = isqrt(per_view)
    if side ** 2 != per_view:
        raise ValueError('Selection report requires square camera token grids')
    selections = []
    for row in rows:
        sample = row['variants']
        for variant in config['variants']:
            selected = sample[variant['name']].get('selection')
            if selected is None:
                continue
            count = variant['keep']
            raw = set(sample[f'raw_{count}']['selection']['picks'])
            corrected = set(sample[f'corrected_{count}']['selection']['picks'])
            for rank, index in enumerate(selected['picks'], 1):
                camera, local = divmod(index, per_view)
                grid_row, grid_col = divmod(local, side)
                membership = ('both' if index in raw and index in corrected else 'raw only' if index in raw
                              else 'corrected only' if index in corrected else 'neither')
                selections.append(dict(sample_id=row['sample_id'], variant=variant['name'], keep=count,
                    selection_rank=rank, visual_index=index, camera=config['cameras'][camera],
                    grid_row=grid_row, grid_col=grid_col, raw_score=selected['raw'][index],
                    corrected_score=selected['corrected'][index], top_k_membership=membership,
                    max_cosine=None if selected['penalties'] is None or rank == 1 else selected['penalties'][rank - 1]))
    return selections


def _summarize(rows, config):
    names = {'baseline': config['visual_count'], **{v['name']: v['keep'] for v in config['variants']}}
    quality = _quality(rows, names)
    summary = []
    for name, count in names.items():
        samples = [row['variants'][name] for row in rows]
        first_steps = [sample['first_beam_step_ms'] for sample in samples if sample['first_beam_step_ms'] is not None]
        scores = quality[name]
        record = dict(variant=name, kept_tokens=count, **scores, examples=len(samples),
                      mean_generated_tokens=mean(sample['generated_tokens'] for sample in samples),
                      median_first_beam_step_ms=median(first_steps) if first_steps else None,
                      measured_first_beam_steps=len(first_steps),
                      median_latency_ms=median(sample['latency_ms'] for sample in samples),
                      delta_overall=scores['overall'] - quality['baseline']['overall'],
                      answer_changed_pct=100 * mean(row['variants']['baseline']['prediction'] != sample['prediction']
                                                   for row, sample in zip(rows, samples)))
        for key in ('mean_abs_weighted_relevance', 'mean_abs_weighted_similarity'):
            values = [sample[key] for sample in samples if key in sample]
            record[key] = mean(values) if values else None
        for i, camera in enumerate(config['cameras']):
            record[f'mean_kept_{camera}'] = (config['per_view'] if name == 'baseline'
                                            else mean(sample['camera_counts'][i] for sample in samples))
        summary.append(record)
    return summary


def _bias_figures(config, curve, output):
    per_view = config['per_view']
    cameras = config['cameras']
    fitted = np.asarray(curve['fitted'])
    fig, ax = plt.subplots(figsize=(11, 4.6), layout='constrained')
    ax.plot(curve['calibration_observed'], color='#4677a2', lw=.9, label='Calibration mean')
    ax.plot(fitted, color='#c36719', lw=2.1, label='Cubic fit')
    for boundary in range(per_view, len(fitted), per_view):
        ax.axvline(boundary - .5, color='0.8', lw=.7)
    ax.set(title='Positional Bias Fitting Function', xlabel='Original visual index (zero-based)',
           ylabel=f'Attention difference (layer {config["shallow"]} − layer {config["deep"]})')
    ax.secondary_xaxis('top').set_xticks(np.arange(len(cameras)) * per_view + (per_view - 1) / 2, cameras)
    ax.legend()
    paths = _export(fig, output, 'positional_bias_fit', 180)

    fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
    residuals = (np.asarray(curve['evaluation_observed']) - fitted).reshape(len(cameras), per_view)
    ax.boxplot(residuals.T, showmeans=True)
    ax.set_xticks(range(1, len(cameras) + 1), cameras)
    ax.axhline(0, color='0.45', linestyle='--', lw=1)
    ax.set(title='Positional bias residuals by camera on held-out questions', ylabel='Observed difference − fitted difference')
    return paths + _export(fig, output, 'residuals_by_camera', 180)


def _selection_figure(selections, config, output):
    first = selections[0]['sample_id']
    example = [row for row in selections if row['sample_id'] == first]
    budgets = sorted({row['keep'] for row in example}, reverse=True)
    colors = {'both': '#366e57', 'raw only': '#4677a2', 'corrected only': '#c36719', 'neither': '#8a7aab'}
    fig, axes = plt.subplots(len(budgets), 1, figsize=(11, 2.7 * len(budgets) + 1),
                             sharex=True, squeeze=False, layout='constrained')
    labels = ['Raw attention', 'Corrected attention', 'Corrected + MMR']
    for budget, ax in zip(budgets, axes[:, 0]):
        for y, method in enumerate(('raw', 'corrected', 'mmr')):
            picks = [row for row in example if row['variant'] == f'{method}_{budget}']
            for code, color in colors.items():
                indices = [row['visual_index'] for row in picks if row['top_k_membership'] == code]
                ax.scatter(indices, [y] * len(indices), s=13, color=color)
        for boundary in range(config['per_view'], config['visual_count'], config['per_view']):
            ax.axvline(boundary - .5, color='0.8', lw=.7, zorder=0)
        ax.set(yticks=[0, 1, 2], yticklabels=labels, ylim=(-.5, 2.5), title=f'Keep {budget} visual tokens')
    legend = [Line2D([], [], marker='o', linestyle='none', color=color, label=code.capitalize())
              for code, color in colors.items()]
    fig.legend(handles=legend, ncol=len(colors), loc='outside lower center')
    axis = axes[-1, 0]
    axis.set(xlabel='Original visual index (zero-based)', xlim=(-1, config['visual_count']))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=13, integer=True))
    fig.suptitle(f'Token selection · {first}')
    return _export(fig, output, 'indexed_token_selections', 180)


def _comparison_figure(summary, output):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout='constrained')
    labels = [row['variant'].replace('_', '\n') for row in summary]
    colors = {'baseline': '#555555', 'raw': '#4677a2', 'corrected': '#c36719', 'mmr': '#267862'}
    for ax, key, title in zip(axes, ('delta_overall', 'median_first_beam_step_ms', 'median_latency_ms'),
            ('Change in Overall', 'First beam step\nmedian (ms)', 'Generation latency\nmedian (ms)')):
        baseline = summary[0][key]
        ax.bar(labels, [np.nan if row[key] is None else row[key] for row in summary],
               color=[colors[row['variant'].split('_')[0]] for row in summary])
        if baseline is not None:
            ax.axhline(baseline, color='0.35', linestyle='--', lw=1)
        ax.set_ylabel(title)
    fig.suptitle('Unpruned baseline and matched-budget pruning variants')
    return _export(fig, output, 'baseline_vs_pruned', 180)


def report(output):
    output = Path(output)
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    curve = json.loads((output / 'bias_curve.json').read_text(encoding='utf-8'))
    with (output / 'predictions.jsonl').open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    if [row['sample_id'] for row in rows] != config['evaluation_ids']:
        raise ValueError('Paired generation must finish before reporting')
    selections = _selection_rows(rows, config)
    _csv(output / 'indexed_selections.csv', selections)
    summary = _summarize(rows, config)
    _csv(output / 'comparison.csv', summary)
    fields = ('sample_id', 'question', 'ground_truth')
    predictions = [dict({key: row[key] for key in fields},
                        **{result['variant']: row['variants'][result['variant']]['prediction'] for result in summary})
                   for row in rows]
    _csv(output / 'paired_answers.csv', predictions)
    with plt.rc_context(_STYLE):
        return (_bias_figures(config, curve, output) + _selection_figure(selections, config, output)
                + _comparison_figure(summary, output))
