"""Experiment matrix and reproducible selection for the DriveLM test split."""

from collections import defaultdict
from dataclasses import dataclass
import random


SEED = 42
SAMPLE_COUNT = 4_204
MERGE_SEEDS = (42, 43, 44)
MIDDLE = ((9, 15), (18, 24))
DEEP = (20, 23, 26)


@dataclass(frozen=True)
class Variant:
    name: str
    title: str
    layers: tuple[int, ...] = ()
    mode: str = 'baseline'
    preserve_mass: bool = False
    seed: int | None = None


def _middle(start, stop, mode, title, *, preserve=False):
    layers = tuple(range(start, stop + 1))
    suffix = '-mass' if preserve else ''
    description = ' và bảo toàn visual mass' if preserve else ''
    return Variant(f'middle-{start}-{stop}-{mode}{suffix}',
                   f'Layer {start}–{stop} · {title}{description}', layers, mode, preserve)


VARIANTS = (
    Variant('baseline', 'Baseline · SmolVLM với đủ token ảnh'),
    Variant('shallow-top10', 'Layer 1–2 · Mask top 10%', (1, 2), 'top10'),
    Variant('shallow-keep10', 'Layer 1–2 · Chỉ giữ top 10%', (1, 2), 'keep10'),
    Variant('shallow-none', 'Layer 1 · Mask toàn bộ attention lên ảnh', (1,), 'mask_all'),
    *(Variant(f'shallow-merge-{seed}', f'Layer 1 · Merge attention với seed {seed}', (1,), 'merge', seed=seed)
      for seed in MERGE_SEEDS),
    *(_middle(start, stop, mode, title, preserve=preserve)
      for start, stop in MIDDLE
      for mode, title, preserve in (('top10', 'Mask top 10%', False),
                                    ('bottom10', 'Mask bottom 10%', False),
                                    ('keep5', 'Chỉ giữ top 5%', False),
                                    ('top10', 'Mask top 10%', True),
                                    ('bottom10', 'Mask bottom 10%', True))),
    *(Variant(f'deep-{mode}-{layer}', f'Layer {layer} · {mode.capitalize()} token ảnh', (layer,), mode)
      for layer in DEEP for mode in ('skip', 'discard')),
)
BY_NAME = {variant.name: variant for variant in VARIANTS}
RANK_LAYERS = (1, 2, *(layer for start, stop in MIDDLE for layer in range(start, stop + 1)))


def yes_no(text):
    return text.strip().casefold().rstrip('.!') in {'yes', 'no'}


def select_indices(records, count=SAMPLE_COUNT):
    groups = defaultdict(list)
    for i, (qa, _) in enumerate(records):
        groups[qa['question_type'], yes_no(qa['A'])].append(i)
    if not 0 < count <= len(records):
        raise ValueError('Sample count must be within the test split')
    quotas = {key: len(group) * count // len(records) for key, group in groups.items()}
    remaining = count - sum(quotas.values())
    for key in sorted(groups, key=lambda k: (-(len(groups[k]) * count % len(records)), k))[:remaining]:
        quotas[key] += 1
    rng = random.Random(SEED)
    return sorted(index for key in sorted(groups) for index in rng.sample(groups[key], quotas[key]))


def merge_positions():
    """Choose three different cameras and one patch in each horizontal band."""
    cameras = random.Random(SEED).sample(range(6), len(MERGE_SEEDS))
    bands = ((0, 3), (3, 5), (5, 8))
    positions = {}
    for seed, camera, (start, stop) in zip(MERGE_SEEDS, cameras, bands):
        rng = random.Random(seed)
        row, col = rng.randrange(start, stop), rng.randrange(8)
        positions[seed] = dict(camera=camera, row=row, col=col, token=camera * 64 + row * 8 + col)
    return positions
