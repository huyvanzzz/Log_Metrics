import os
import subprocess
import socket
import sys
import pytest
import torch


def test_two_rank_cpu_training_resume_and_validation(tmp_path):
    from map_support import make_inputs
    assert torch.distributed.is_gloo_available(), 'PyTorch Gloo backend is required for this integration test'
    model, receipt = make_inputs(tmp_path)
    torch.save(model, tmp_path/'fixture_model.pt')
    environment = os.environ.copy()
    environment.update(PYTHONPATH=os.pathsep.join([os.getcwd(), os.path.join(os.getcwd(), 'tests')]),
                       USE_TF='0', USE_LIBUV='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if os.name == 'nt':
        processes = []
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = str(sock.getsockname()[1])
        try:
            for rank in range(2):
                env = dict(environment, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2',
                           MASTER_ADDR='127.0.0.1', MASTER_PORT=port)
                processes.append(subprocess.Popen([sys.executable, 'tests/map_ddp_worker.py', str(tmp_path)],
                                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
            results = [process.communicate(timeout=180)[0] for process in processes]
            assert all(process.returncode == 0 for process in processes), '\n'.join(results)
        finally:
            for process in processes:
                if process.poll() is None: process.terminate()
            for process in processes:
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
    else:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   'tests/map_ddp_worker.py', str(tmp_path)]
        run = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
        assert run.returncode == 0, run.stdout + '\n' + run.stderr
    full = torch.load(tmp_path/'full'/'checkpoints'/'predictor_latest.pt', weights_only=False)
    resumed = torch.load(tmp_path/'resumed'/'checkpoints'/'predictor_latest.pt', weights_only=False)
    assert full['metadata']['optimizer_step'] == resumed['metadata']['optimizer_step'] == 2
    assert full['metadata']['completed_epochs'] == resumed['metadata']['completed_epochs'] == 2
    assert full['metadata']['val_loss'] == resumed['metadata']['val_loss']
    for name, value in full['model_state_dict'].items():
        torch.testing.assert_close(value, resumed['model_state_dict'][name], rtol=0, atol=0)
    import json
    assert [json.loads((tmp_path/f'worker_{rank}.json').read_text())['val_loss'] for rank in (0, 1)] == [full['metadata']['val_loss']]*2
    events = [json.loads(line) for line in (tmp_path/'full'/'debug'/'train_progress.jsonl').read_text().splitlines()]
    assert all(row['sample_count'] == 3 for row in events if row['event'] == 'validation_end')
