"""Find the pinned feature cache already mounted as a Kaggle notebook input."""

import json
from pathlib import Path

from smolvlm_inference.cache import FEATURE_FILE


def _matching_caches(roots, fingerprint):
    checked, matched = set(), set()
    for root in roots:
        for payload in root.rglob(FEATURE_FILE):
            folder = payload.parent.resolve()
            manifest = folder / 'manifest.json'
            if (folder in checked or not payload.is_file() or not manifest.is_file()
                    or not (folder / 'index.json').is_file()):
                continue
            checked.add(folder)
            try:
                metadata = json.loads(manifest.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if isinstance(metadata, dict) and metadata.get('fingerprint') == fingerprint:
                matched.add(folder)
    return matched, checked


def resolve_cache_input(input_root, explicit_dir, *, dataset, fingerprint):
    root = Path(explicit_dir or input_root).expanduser()
    owner, name, _, version = dataset.split('/')
    attach = (f'Attach {owner}/{name} version {version} as a saved notebook input before '
              'starting a non-interactive Kaggle run. Leave CACHE_INPUT_DIR blank for auto-detection.')
    if explicit_dir and not root.is_dir():
        raise FileNotFoundError(f'CACHE_INPUT_DIR is not a directory: {root}. {attach}')

    matched, checked = set(), set()
    if not explicit_dir:
        mounts = (root / name, root / 'datasets' / owner / name)
        matched, checked = _matching_caches(mounts, fingerprint)
    if not matched:
        matched, scanned = _matching_caches((root,), fingerprint)
        checked.update(scanned)
    if len(matched) == 1:
        return matched.pop()
    if len(matched) > 1:
        paths = ', '.join(str(path) for path in sorted(matched))
        raise ValueError(f'Multiple matching cache directories: {paths}. Set CACHE_INPUT_DIR to one of them.')
    if checked:
        raise ValueError(f'Found {len(checked)} cache folder(s) under {root}, but none match the pinned cache. {attach}')
    raise FileNotFoundError(f'No complete feature cache found under {root}. {attach}')
