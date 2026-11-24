"""Post-block-2 last-prompt-query predictor commands; old CLIs remain unchanged."""
import argparse
import json
from pathlib import Path
import signal


def _worker_signals():
    # Let worker finally blocks join loader pools when the supervisor stops a peer.
    def interrupted(signum, frame):
        raise InterruptedError(f'Predictor worker received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('validate-inputs', help='Validate complete targets/cache and prepare a fresh or resumed run')
    prepare.add_argument('--config', default=str(Path(__file__).resolve().parents[1]/'configs/smolvlm_map_predictor_after2_lastquery_index22.json'))
    prepare.add_argument('--targets-input', required=True)
    prepare.add_argument('--cache-input', default='')
    prepare.add_argument('--prepared-dir')
    prepare.add_argument('--base-checkpoint')
    prepare.add_argument('--output')
    prepare.add_argument('--resume', action='store_true')
    for name in ('diagnose', 'train', 'evaluate', 'verify-resume-checkpoint', 'bundle',
                 'diagnose-worker', 'train-worker', 'evaluate-worker'):
        item = commands.add_parser(name)
        item.add_argument('--run', required=True)
        if name in ('train', 'train-worker'):
            item.add_argument('--resume')
        if name in ('evaluate', 'evaluate-worker', 'verify-resume-checkpoint'):
            item.add_argument('--predictor', required=True)
        if name == 'evaluate-worker':
            item.add_argument('--rank', type=int, choices=(0,1), required=True)
    return root


def verify_resume(run, path):
    from smolvlm_inference.model import load_model, CAMERA_ORDER
    from .training import make_predictor
    from .config import recipe
    from .checkpoint import read
    base = load_model(run['backbone_checkpoint'], device='cpu', expected_prompt_camera_order=CAMERA_ORDER)
    predictor = make_predictor(base, run['config'])
    value = read(path, predictor, run['identity'], training=True, recipe=recipe(run['config']),
                 world_size=run['config']['training']['world_size'])
    predictor.load_state_dict(value['model_state_dict'], strict=True)
    optimizer = __import__('torch').optim.AdamW(predictor.parameters())
    optimizer.load_state_dict(value['optimizer_state_dict'])
    scheduler = __import__('torch').optim.lr_scheduler.ExponentialLR(optimizer, gamma=run['config']['training']['scheduler_gamma'])
    scheduler.load_state_dict(value['scheduler_state_dict'])
    result = dict(ok=True, scope='read_only_training_state_load', metadata=value['metadata'])
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    from . import runtime
    if args.command == 'validate-inputs':
        from .config import load_config
        cfg = load_config(args.config)
        return runtime.prepare(cfg, args.targets_input, args.output or cfg['output_dir'],
            cache_input=args.cache_input, prepared_dir=args.prepared_dir,
            backbone_checkpoint=args.base_checkpoint, resume=args.resume)
    if args.command == 'bundle':
        result = runtime.bundle(args.run)
        print(result, flush=True)
        return result
    run = runtime.read_run(args.run, include_targets=args.command not in ('evaluate', 'evaluate-worker'))
    if args.command == 'verify-resume-checkpoint':
        return verify_resume(run, args.predictor)
    if args.command in ('train', 'diagnose', 'evaluate'):
        return runtime.launch(args.run, args.command, resume=getattr(args, 'resume', None),
                              predictor=getattr(args, 'predictor', None))
    _worker_signals()
    try:
        if args.command == 'train-worker':
            from .training import train
            return train(run['config'], run['receipt'], run['backbone_checkpoint'], run['output_dir'], resume=args.resume)
        if args.command == 'diagnose-worker':
            from .diagnostics import diagnose
            return diagnose(run['config'], run['receipt'], run['backbone_checkpoint'], run['output_dir'])
        if args.command == 'evaluate-worker':
            from .evaluation import evaluate_worker
            return evaluate_worker(run, args.rank, args.predictor)
    except BaseException as exc:
        import os
        from smolvlm_attention.sample_store import atomic_json
        rank = getattr(args, 'rank', int(os.environ.get('RANK', '0')))
        atomic_json(Path(run['output_dir'])/'logs'/f'{args.command}_rank_{rank}_failure.json',
                    dict(rank=rank, exception=type(exc).__name__, message=str(exc), resources=runtime.resource_snapshot()))
        raise
    raise AssertionError('Unreachable command')
