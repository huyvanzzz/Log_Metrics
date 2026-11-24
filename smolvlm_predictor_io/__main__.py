"""Internal projected-cache worker entrypoint."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build-worker'])
    parser.add_argument('--spec', required=True)
    parser.add_argument('--rank', required=True, type=int)
    args = parser.parse_args()
    from .orchestration import build_worker
    build_worker(args.spec, args.rank)


if __name__ == '__main__':
    main()
