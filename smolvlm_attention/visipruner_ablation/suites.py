"""Choose the original matrix or the full-test deep layer sweep."""

from .design import VARIANTS, Variant, _middle


DEEP_LAYERS = tuple(range(24, 31))
DEEP_SWEEP = (
    VARIANTS[0],
    *(Variant(f'deep-discard-{layer}', f'Layer {layer} · Discard token ảnh', (layer,), 'discard')
      for layer in DEEP_LAYERS),
)
MIDDLE_CONTROLS = (('keep5', 'Chỉ giữ top 5%'), ('keep10', 'Chỉ giữ top 10%'),
                   ('top10', 'Mask top 10%'), ('top5', 'Mask top 5%'))
MIDDLE_FULL_RANGE = (20, 24)
MIDDLE_SWEEP = tuple(
    _middle(*MIDDLE_FULL_RANGE, mode, title, preserve=preserve)
    for mode, title in MIDDLE_CONTROLS for preserve in (False, True)
)
SUITES = {'original': VARIANTS, 'deep-full': DEEP_SWEEP, 'middle-full': MIDDLE_SWEEP}
RUN_VARIANTS = {variant.name: variant for variants in SUITES.values() for variant in variants}


def suite_variants(manifest):
    variants = SUITES[manifest.get('suite', 'original')]
    if 'variants' in manifest and [entry['name'] for entry in manifest['variants']] != [v.name for v in variants]:
        raise ValueError('Manifest has a different experiment matrix; prepare a new output directory')
    return variants
