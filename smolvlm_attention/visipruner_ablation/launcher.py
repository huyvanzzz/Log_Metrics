"""Run independent intervention variants on separate visible GPUs."""

import json
from pathlib import Path
import subprocess
import sys

from .suites import suite_variants


def run_pair(variants, *, output, image_root, batch_size, limit, checkpoint):
    import torch

    if not 1 <= len(variants) <= 2 or len(set(variants)) != len(variants):
        raise ValueError('Specify one or two distinct intervention variants')
    if 'baseline' in variants:
        manifest = json.loads((Path(output) / 'manifest.json').read_bytes())
        if (len(variants) != 2 or manifest.get('suite') != 'deep-full'
                or not set(variants) <= {variant.name for variant in suite_variants(manifest)}):
            raise ValueError('Baseline can run in a pair only with a Discard variant in the deep-full suite')
    gpu_count = torch.cuda.device_count()
    if not gpu_count:
        raise ValueError('Enable a Kaggle GPU before running interventions')
    base = [sys.executable, '-u', '-m', 'smolvlm_attention.visipruner_ablation', 'run',
            '--output', str(output), '--image-root', str(image_root), '--batch-size', str(batch_size),
            '--max-new-tokens', str(limit)]
    if checkpoint is not None:
        base.extend(('--checkpoint', str(checkpoint)))

    def command(variant, gpu):
        print(f'Launching {variant} on cuda:{gpu}', flush=True)
        return [*base, '--variant', variant, '--gpu', str(gpu)]

    if gpu_count == 1 or len(variants) == 1:
        for variant in variants:
            subprocess.run(command(variant, 0), check=True)
        return

    processes = {}
    try:
        for gpu, variant in enumerate(variants):
            processes[variant] = subprocess.Popen(command(variant, gpu))
        failures = {variant: code for variant, process in processes.items() if (code := process.wait()) != 0}
        if failures:
            details = ', '.join(f'{name}: exit {status}' for name, status in failures.items())
            raise RuntimeError(f'Intervention worker failed ({details}). See the variant log and rerun the cell.')
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            process.wait()
