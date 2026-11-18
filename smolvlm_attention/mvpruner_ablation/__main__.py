"""Commands for the Kaggle notebook and local manifest checks."""

import argparse
from pathlib import Path

from .data import prepare_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest='step', required=True)
    for step in ('prepare', 'measure', 'plot'):
        command = steps.add_parser(step)
        command.add_argument('--image-root', type=Path, required=True,
                             help='Directory containing nuscenes/samples')
        command.add_argument('--output', type=Path, required=True)
        if step == 'prepare':
            command.add_argument('--test', type=Path, required=True)
            command.add_argument('--example-id', type=int, help='Zero-based test row for the heatmaps')
        if step == 'measure':
            command.add_argument('--batch-size', type=int, default=8)
            command.add_argument('--checkpoint', type=Path, help='Trusted local copy of the base checkpoint')

    args = parser.parse_args()
    manifest_path = args.output / 'manifest.json'
    if args.step == 'prepare':
        from filelock import FileLock

        args.output.mkdir(parents=True, exist_ok=True)
        with FileLock(str(args.output / 'measure.lock'), timeout=0):
            prepare(args, manifest_path)
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
