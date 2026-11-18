"""Kaggle command line for the VisiPruner ablation."""

import argparse
import os
from pathlib import Path

from .design import BY_NAME, SAMPLE_COUNT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'handoff', 'score', 'plot'):
        command = commands.add_parser(name)
        command.add_argument('--output', type=Path, required=True)
        if name in ('prepare', 'run'):
            command.add_argument('--image-root', type=Path, required=True)
        if name == 'prepare':
            command.add_argument('--test', type=Path, required=True)
            command.add_argument('--samples', type=int, default=SAMPLE_COUNT)
        if name == 'run':
            command.add_argument('--variant', choices=BY_NAME, required=True)
            command.add_argument('--batch-size', type=int, default=8)
            command.add_argument('--max-new-tokens', type=int, default=128)
            command.add_argument('--checkpoint', type=Path)
        if name == 'handoff':
            command.add_argument('--source', type=Path, default=os.environ.get('PART1_OUTPUT') or None)
    args = parser.parse_args()
    if args.command == 'prepare':
        from .runtime import prepare

        prepare(args.test, args.image_root, args.output, args.samples)
    elif args.command == 'run':
        from .runtime import run

        run(args.output, args.image_root, BY_NAME[args.variant], batch_size=args.batch_size,
            limit=args.max_new_tokens, checkpoint=args.checkpoint)
    elif args.command == 'handoff':
        from .handoff import handoff

        try:
            handoff(args.output, args.source)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        from .report import plot, score

        if args.command == 'score':
            score(args.output)
        else:
            for path in plot(args.output):
                print(path)


if __name__ == '__main__':
    main()
