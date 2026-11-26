"""Sample the pinned test set by its annotated question types."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from smolvlm_inference.cache import digest


QUESTION_TYPES_PATH = Path(__file__).with_name('test_question_types.json')


def _sample_groups(groups, available, target, seed):
    total = sum(available.values())
    quotas = {name: divmod(target * size, total) for name, size in available.items()}
    selected = {name: quota for name, (quota, _) in quotas.items()}
    # Use the largest remainders so integer allocations add up to the requested size.
    left = target - sum(selected.values())
    for name in sorted(quotas, key=lambda name: (-quotas[name][1], name))[:left]:
        selected[name] += 1
    rng = random.Random(seed)
    chosen = {i: name for name in sorted(groups) for i in rng.sample(groups[name], selected[name])}
    return chosen, selected


def stratified_subset(evaluation, distribution, target, seed):
    """Sample only from the selected cohort, keeping its annotated question-type mix."""
    if type(target) is not int or not 1 <= target <= len(evaluation) or not distribution:
        raise ValueError('The subset size must fit an annotated evaluation cohort')
    groups = defaultdict(list)
    for i, row in enumerate(evaluation):
        groups[row['question_type']].append(i)
    available = {row['question_type']: row['selected_count'] for row in distribution}
    observed = {name: len(groups.get(name, ())) for name in available}
    if observed != available or not set(groups).issubset(available):
        raise ValueError('Question type counts do not match the evaluation cohort')

    chosen, selected = _sample_groups(groups, available, target, seed)
    subset = [evaluation[i] for i in sorted(chosen)]
    updated = [dict(row, parent_count=row['selected_count'], selected_count=selected[row['question_type']],
                    selected_share=selected[row['question_type']] / target) for row in distribution]
    return subset, updated


def stratified_test(test, prepared_test, fraction, seed, *, labels_path=QUESTION_TYPES_PATH):
    if not 0 < fraction <= 1 or (target := int(len(test) * fraction)) < 1:
        raise ValueError('Evaluation fraction must be within (0, 1] and select at least one question')
    with labels_path.open(encoding='utf-8') as stream:
        index = json.load(stream)
    labels, types = index['labels_by_eval_id'], index['question_types']
    if (digest(prepared_test) != index['prepared_sha256'] or len(labels) != len(test)
            or any(row['eval_id'] != i for i, row in enumerate(test))):
        raise ValueError('Question-type labels do not match the pinned prepared test split')

    groups = defaultdict(list)
    for i, code in enumerate(labels):
        groups[types[int(code)]].append(i)
    available = Counter({name: len(rows) for name, rows in groups.items()})
    if available != index['counts']:
        raise ValueError('Question-type label counts do not match the label index')

    chosen, selected = _sample_groups(groups, available, target, seed)
    evaluation = [dict(test[i], question_type=chosen[i]) for i in sorted(chosen)]
    distribution = [dict(question_type=name, full_count=available[name], selected_count=selected[name],
                         full_share=available[name] / len(test), selected_share=selected[name] / target)
                    for name in sorted(groups)]
    return evaluation, distribution
