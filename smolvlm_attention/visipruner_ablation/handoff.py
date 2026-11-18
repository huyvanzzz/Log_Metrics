"""Verify and copy the completed first half into the second Kaggle session."""

import json
import re
import shutil
from pathlib import Path

from filelock import FileLock

from smolvlm_attention.mvpruner_ablation.data import atomic_output, file_sha256

from .design import VARIANTS

PART1 = VARIANTS[:12]


def _finished(root):
    return (root / 'manifest.json').is_file() and (root / 'run.json').is_file() and all(
        (root / 'predictions' / variant.name / 'complete.json').is_file() for variant in PART1
    )


def _required_files(root):
    manifest_path = root / 'manifest.json'
    identity = json.loads((root / 'run.json').read_bytes())
    if identity['manifest'] != file_sha256(manifest_path):
        raise ValueError(f'Run metadata does not match the manifest in {root}')
    manifest = json.loads(manifest_path.read_bytes())
    frames = manifest['frames']
    if not frames or any(not re.fullmatch(r'[0-9a-f]{64}', frame) for frame in frames):
        raise ValueError('Invalid frame IDs in Part 1 manifest')
    files = [Path('manifest.json'), Path('run.json')]
    for frame in frames:
        files.extend((Path('rankings') / f'{frame}.npz', Path('projected') / f'{frame}.npy'))
    for variant in PART1:
        folder = Path('predictions') / variant.name
        marker = json.loads((root / folder / 'complete.json').read_bytes())
        if marker != dict(samples=manifest['samples'], variant=variant.name):
            raise ValueError(f'Invalid completion marker for {variant.name}')
        files.extend(folder / f'{frame}.json' for frame in frames)
        files.append(folder / 'complete.json')
    for relative in files:
        if not (root / relative).is_file():
            raise ValueError(f'Missing Part 1 artifact: {root / relative}')
    return files


def handoff(output, source=None, *, input_root=Path('/kaggle/input')):
    output = output.resolve()
    if source is None:
        if _finished(output):
            source = output
        else:
            candidates = sorted({path.parent.resolve() for path in input_root.rglob('manifest.json')
                                 if _finished(path.parent)})
            if len(candidates) != 1:
                details = ', '.join(map(str, candidates)) or 'none'
                raise ValueError('Attach the saved Part 1 notebook output as Kaggle Input, '
                                 f'or set PART1_OUTPUT to its result directory. Candidates: {details}')
            source = candidates[0]
    source = source.resolve()
    if not _finished(source):
        raise ValueError(f'Part 1 is incomplete in {source}; finish experiments 1–12 first')
    files = _required_files(source)
    if source == output:
        print(f'Part 1 verified in {output}')
        return

    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / 'run.lock'), timeout=0):
        # Check conflicts before writing any file. Existing Part 2 predictions remain in place.
        for relative in files:
            destination = output / relative
            if destination.exists() and file_sha256(destination) != file_sha256(source / relative):
                raise ValueError(f'Existing Part 1 artifact differs from the source: {destination}')
        logs = [Path('logs') / f'{variant.name}.log' for variant in PART1]
        for relative in [*files, *(path for path in logs if (source / path).is_file())]:
            destination = output / relative
            if not destination.exists():
                with (source / relative).open('rb') as existing, atomic_output(destination) as restored:
                    shutil.copyfileobj(existing, restored)
    print(f'Restored Part 1 from {source} to {output}')
