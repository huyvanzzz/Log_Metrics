"""CLI for separate test-only MMR runs."""
import argparse
from . import runtime
from .config import load_config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','evaluate','worker','bundle'))
    parser.add_argument('--config'); parser.add_argument('--variant',choices=('before','after2'),default='before')
    parser.add_argument('--run'); parser.add_argument('--output'); parser.add_argument('--predictor')
    parser.add_argument('--projected-cache-input',default=''); parser.add_argument('--projected-cache-dir',default='')
    parser.add_argument('--cache-input',default=''); parser.add_argument('--prepared-dir'); parser.add_argument('--base-checkpoint')
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--rank',type=int)
    args=parser.parse_args()
    if args.command=='prepare':
        cfg=load_config(args.config,variant=args.variant)
        if not args.predictor: parser.error('prepare requires --predictor checkpoint')
        return runtime.prepare(cfg,args.output or cfg['output_dir'],args.predictor,
            projected_input=args.projected_cache_input,projected_dir=args.projected_cache_dir,cache_input=args.cache_input,
            prepared_dir=args.prepared_dir,base_checkpoint=args.base_checkpoint,resume=args.resume)
    if not args.run: parser.error('command requires --run')
    if args.command=='evaluate': return runtime.launch(args.run)
    if args.command=='bundle': return runtime.bundle(args.run)
    if args.rank not in (0,1): parser.error('worker requires --rank 0 or 1')
    return runtime.worker(runtime.read_run(args.run),args.rank)

if __name__=='__main__': main()
