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


PREDICTOR_LABELS = dict(mmr='Spline + MMR', predictor_topk='MLP Top-32', predictor_mmr='MLP + MMR')


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


def _selection_reference(config):
    if config.get('experiment') != 'spline_vs_predictor':
        return ('raw', 'corrected'), ('raw only', 'corrected only')
    if any(v['method'] == 'predictor_topk' for v in config['variants']):
        return ('predictor_topk', 'predictor_mmr'), ('MLP Top-32 only', 'MLP + MMR only')
    return ('mmr', 'predictor_mmr'), ('spline only', 'MLP only')


def _selection_rows(rows, config):
    per_view = config['per_view']
    side = isqrt(per_view)
    if side ** 2 != per_view:
        raise ValueError('Selection report requires square camera token grids')
    selections = []
    predictor_comparison = config.get('experiment') == 'spline_vs_predictor'
    (first_method, second_method), (first_only, second_only) = _selection_reference(config)
    for row in rows:
        sample = row['variants']
        for variant in config['variants']:
            selected = sample[variant['name']].get('selection')
            if selected is None:
                continue
            count = variant['keep']
            first = set(sample[f'{first_method}_{count}']['selection']['picks'])
            second = set(sample[f'{second_method}_{count}']['selection']['picks'])
            for rank, index in enumerate(selected['picks'], 1):
                camera, local = divmod(index, per_view)
                grid_row, grid_col = divmod(local, side)
                membership = ('both' if index in first and index in second else first_only if index in first
                              else second_only if index in second else 'neither')
                selection = dict(sample_id=row['sample_id'], variant=variant['name'], keep=count,
                    selection_rank=rank, visual_index=index, camera=config['cameras'][camera],
                    grid_row=grid_row, grid_col=grid_col,
                    raw_score=None if selected['raw'] is None else selected['raw'][index],
                    corrected_score=None if selected['corrected'] is None else selected['corrected'][index],
                    top_k_membership=membership,
                    max_cosine=None if selected['penalties'] is None or rank == 1 else selected['penalties'][rank - 1])
                if predictor_comparison:
                    selection['relevance'] = selected['relevance'][index]
                selections.append(selection)
    return selections


def _summarize(rows, config):
    matched_raw = config.get('experiment') in ('raw_mmr', 'camera_posprune')
    predictor_comparison = config.get('experiment') == 'spline_vs_predictor'
    names = {} if matched_raw or predictor_comparison else {'baseline': config['visual_count']}
    names.update({v['name']: v['keep'] for v in config['variants']})
    quality = _quality(rows, names)
    has_predictor_topk = predictor_comparison and any(v['method'] == 'predictor_topk' for v in config['variants'])
    batches = {row.get('batch_id', i): row for i, row in enumerate(rows)}
    summary = []
    for name, count in names.items():
        reference = f'mmr_{count}' if predictor_comparison else f'raw_{count}' if matched_raw else 'baseline'
        samples = [row['variants'][name] for row in rows]
        batch_results = [row['variants'][name] for row in batches.values()]
        first_steps = [result.get('first_token_step_ms', result.get('first_beam_step_ms')) for result in batch_results]
        first_steps = [value for value in first_steps if value is not None]
        latencies = [result['latency_ms'] for result in batch_results if result.get('latency_ms') is not None]
        total_ms = sum(latencies)
        scores = quality[name]
        record = dict(variant=name, kept_tokens=count, **scores, examples=len(samples),
                      measured_batches=len(latencies), mean_batch_size=len(samples) / len(latencies) if latencies else None,
                      mean_generated_tokens=mean(sample['generated_tokens'] for sample in samples),
                      median_first_token_step_ms=median(first_steps) if first_steps else None,
                      measured_first_token_steps=len(first_steps),
                      median_latency_ms=median(latencies) if latencies else None,
                      inference_questions_per_gpu_second=1000 * len(samples) / total_ms if total_ms else None,
                      delta_overall=scores['overall'] - quality[reference]['overall'],
                      answer_changed_pct=100 * mean(row['variants'][reference]['prediction'] != sample['prediction']
                                                   for row, sample in zip(rows, samples)))
        if has_predictor_topk:
            topk = f'predictor_topk_{count}'
            record.update(mmr_lambda=None if name == topk else config['weight'],
                          delta_overall_vs_mlp_topk=scores['overall'] - quality[topk]['overall'],
                          answer_changed_pct_vs_mlp_topk=100 * mean(row['variants'][topk]['prediction'] != sample['prediction']
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
    camera_fits = curve.get('camera_fits')
    fit_label = 'Smoothing spline' if curve.get('fit_method') == 'smoothing_spline' else 'Cubic fit'
    segments = range(0, len(fitted), per_view) if camera_fits else (0,)
    for start in segments:
        stop = start + per_view if camera_fits else len(fitted)
        ax.plot(np.arange(start, stop), fitted[start:stop], color='#c36719', lw=2.1,
                label=fit_label if start == 0 else None)
    for boundary in range(per_view, len(fitted), per_view):
        ax.axvline(boundary - .5, color='0.8', lw=.7)
    ax.set(title='Positional Bias Fitting Function', xlabel='Original visual index (zero-based)',
           ylabel=f'Attention difference (layer {config["shallow"]} − layer {config["deep"]})')
    ax.secondary_xaxis('top').set_xticks(np.arange(len(cameras)) * per_view + (per_view - 1) / 2, cameras)
    ax.legend()
    paths = _export(fig, output, 'positional_bias_fit', 180)
    if camera_fits:
        fig, axes = plt.subplots((len(cameras) + 2) // 3, 3, figsize=(12, 6.5), sharex=True, sharey=True,
                                 squeeze=False, layout='constrained')
        for camera, ax in zip(camera_fits, axes.flat):
            start = camera['first_visual_index']
            segment = slice(start, start + per_view)
            ax.plot(curve['calibration_observed'][segment], color='#4677a2', lw=.9, label='Calibration mean')
            ax.plot(curve['evaluation_observed'][segment], color='0.45', lw=.8, linestyle='--', label='Held-out mean')
            ax.plot(fitted[segment], color='#c36719', lw=1.8, label=fit_label)
            ax.set_title(camera['camera'])
        for ax in list(axes.flat)[len(cameras):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=3)
        fig.supxlabel('Camera-local visual index (zero-based)')
        fig.supylabel(f'Attention difference (layer {config["shallow"]} − layer {config["deep"]})')
        paths += _export(fig, output, 'positional_bias_fit_by_camera', 180)

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
    predictor_comparison = config.get('experiment') == 'spline_vs_predictor'
    _, (first_only, second_only) = _selection_reference(config)
    colors = {'both': '#366e57', first_only: '#4677a2', second_only: '#c36719', 'neither': '#8a7aab'}
    fig, axes = plt.subplots(len(budgets), 1, figsize=(11, 2.7 * len(budgets) + 1),
                             sharex=True, squeeze=False, layout='constrained')
    method_labels = dict(raw='Raw attention', corrected='Corrected attention', mmr='Corrected + MMR',
                         raw_mmr='Raw + MMR')
    if predictor_comparison:
        method_labels.update(PREDICTOR_LABELS)
    methods = tuple(dict.fromkeys(variant['method'] for variant in config['variants']))
    labels = [method_labels[method] for method in methods]
    for budget, ax in zip(budgets, axes[:, 0]):
        for y, method in enumerate(methods):
            picks = [row for row in example if row['variant'] == f'{method}_{budget}']
            for code, color in colors.items():
                indices = [row['visual_index'] for row in picks if row['top_k_membership'] == code]
                ax.scatter(indices, [y] * len(indices), s=13, color=color)
        for boundary in range(config['per_view'], config['visual_count'], config['per_view']):
            ax.axvline(boundary - .5, color='0.8', lw=.7, zorder=0)
        ax.set(yticks=range(len(methods)), yticklabels=labels, ylim=(-.5, len(methods) - .5),
               title=f'Keep {budget} visual tokens')
    legend = [Line2D([], [], marker='o', linestyle='none', color=color, label=code.capitalize())
              for code, color in colors.items()]
    fig.legend(handles=legend, ncol=len(colors), loc='outside lower center')
    axis = axes[-1, 0]
    axis.set(xlabel='Original visual index (zero-based)', xlim=(-1, config['visual_count']))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=13, integer=True))
    fig.suptitle(f'Token selection · {first}')
    return _export(fig, output, 'indexed_token_selections', 180)


def _comparison_figure(summary, output, *, matched_raw=False, predictor_comparison=False):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout='constrained')
    labels = [row['variant'].replace('_', '\n') for row in summary]
    colors = {'baseline': '#555555', 'raw': '#4677a2', 'corrected': '#c36719', 'mmr': '#267862', 'raw_mmr': '#8a7aab'}
    if predictor_comparison:
        labels = [PREDICTOR_LABELS[row['variant'].rsplit('_', 1)[0]] for row in summary]
        colors.update(predictor_topk='#4677a2', predictor_mmr='#8a7aab')
    for ax, key, title in zip(axes, ('delta_overall', 'median_first_token_step_ms', 'median_latency_ms'),
            ('Change in Overall', 'First token step\nbatch median (ms)', 'Generation time\nbatch median (ms)')):
        baseline = (0 if key == 'delta_overall' else None) if matched_raw else summary[0][key]
        ax.bar(labels, [np.nan if row[key] is None else row[key] for row in summary],
               color=[colors[row['variant'].rsplit('_', 1)[0]] for row in summary])
        if baseline is not None:
            ax.axhline(baseline, color='0.35', linestyle='--', lw=1)
        ax.set_ylabel(title)
    fig.suptitle('Spline and MLP pruning at 32 tokens' if predictor_comparison else
                 'Pruning variants with matched-budget raw references' if matched_raw else
                 'Unpruned baseline and matched-budget pruning variants')
    return _export(fig, output, 'pruning_comparison' if matched_raw or predictor_comparison else 'baseline_vs_pruned', 180)


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
    distribution = config.get('question_type_distribution')
    if distribution:
        _csv(output / 'question_type_distribution.csv', distribution)
        fields += ('question_type',)
    predictions = [dict({key: row[key] for key in fields},
                        **{result['variant']: row['variants'][result['variant']]['prediction'] for result in summary})
                   for row in rows]
    _csv(output / 'paired_answers.csv', predictions)
    if 'camera_fits' in curve:
        columns = ['camera', 'held_out_rmse', 'global_fit_held_out_rmse']
        if curve.get('fit_method') == 'smoothing_spline':
            columns.extend(('spline_smoothing', 'per_camera_cubic_held_out_rmse'))
        _csv(output / 'camera_fit_metrics.csv', [{key: fit[key] for key in columns} for fit in curve['camera_fits']])
    with plt.rc_context(_STYLE):
        return (_bias_figures(config, curve, output) + _selection_figure(selections, config, output)
                + _comparison_figure(summary, output, matched_raw=config.get('experiment') == 'camera_posprune',
                                     predictor_comparison=config.get('experiment') == 'spline_vs_predictor'))
