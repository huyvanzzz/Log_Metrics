"""Commands for the Kaggle notebook and local manifest checks."""

import argparse
import os
from pathlib import Path

from .data import fetch_hf_test, prepare_manifest, resolve_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest='step', required=True)
    fetch = steps.add_parser('fetch-hf', help='Download the pinned DriveLM test split from Hugging Face')
    fetch.add_argument('--output', type=Path, required=True, help='Directory for the test JSON and images')
    for step in ('prepare', 'measure', 'plot'):
        command = steps.add_parser(step)
        command.add_argument('--image-root', type=Path, default=os.environ.get('DATA_DIR') or None,
                             help='Directory containing nuscenes/samples')
        command.add_argument('--output', type=Path, required=True)
        if step == 'prepare':
            command.add_argument('--test', type=Path, default=os.environ.get('TEST_JSON') or None)
            command.add_argument('--example-id', type=int, help='Zero-based test row for the heatmaps')
        if step == 'measure':
            command.add_argument('--batch-size', type=int, default=8)
            command.add_argument('--checkpoint', type=Path, help='Trusted local copy of the base checkpoint')

    args = parser.parse_args()
    if args.step == 'fetch-hf':
        print(f'Test file: {fetch_hf_test(args.output)}')
        return
    try:
        args.image_root, args.test = resolve_inputs(
            args.image_root, getattr(args, 'test', None), needs_test=args.step == 'prepare',
        )
    except ValueError as exc:
        parser.error(str(exc))
    manifest_path = args.output / 'manifest.json'
    if args.step == 'prepare':
        from filelock import FileLock

        print(f'Test file: {args.test}\nImage root: {args.image_root}', flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        with FileLock(str(args.output / 'measure.lock'), timeout=0):
            try:
                prepare(args, manifest_path)
            except FileNotFoundError as exc:
                parser.error(f'Missing input: {exc.filename}. Check the mounted test dataset and image tree.')
    elif args.step == 'measure':
        from .runtime import measure

        measure(manifest_path, args.image_root, args.output,
                checkpoint=args.checkpoint, batch_size=args.batch_size)
    else:
        from .plotting import plot

        for path in plot(manifest_path, args.image_root, args.output):
            print(path)


def prepare(args, manifest_path):
    from .data import load_manifest

    if (args.output / 'measurement.json').exists():
        manifest = load_manifest(manifest_path)
        from .data import file_sha256

        if (manifest['test_sha256'] != file_sha256(args.test)
                or (args.example_id is not None and manifest['illustration']['id'] != args.example_id)):
            raise ValueError('The measured dataset or illustration differs. Use a new output directory.')
        print('Reusing the manifest from the existing measurement.')
    else:
        manifest = prepare_manifest(args.test, args.image_root, manifest_path, example_id=args.example_id)
    print(f"{len(manifest['rows']):,} questions · {len(manifest['frames']):,} frames · "
          f"{len(manifest['assets']):,} images")
    print(f"Label counts: {manifest['counts']}")
    print(f"Illustration: test row {manifest['illustration']['id']}")


if __name__ == '__main__':
    main()
