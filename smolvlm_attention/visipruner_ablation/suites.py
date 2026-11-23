"""Choose the original matrix or the full-test deep layer sweep."""

from .design import VARIANTS, Variant


DEEP_LAYERS = tuple(range(24, 31))
DEEP_SWEEP = (
    VARIANTS[0],
    *(Variant(f'deep-discard-{layer}', f'Layer {layer} · Discard token ảnh', (layer,), 'discard')
      for layer in DEEP_LAYERS),
)
SUITES = {'original': VARIANTS, 'deep-full': DEEP_SWEEP}
RUN_VARIANTS = {variant.name: variant for variants in SUITES.values() for variant in variants}


def suite_variants(manifest):
    variants = SUITES[manifest.get('suite', 'original')]
    if 'variants' in manifest and [entry['name'] for entry in manifest['variants']] != [v.name for v in variants]:
        raise ValueError('Manifest has a different experiment matrix; prepare a new output directory')
    return variants
