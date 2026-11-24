"""Optional supervisor diagnostics without changing the hashed teacher protocol.

Run with: python -m smolvlm_attention.teacher_targets.diagnostics run --run RUN.json
No targets, run identities, schedules or extraction settings are changed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

from smolvlm_attention.sample_store import atomic_json

_popen = subprocess.Popen


def resource_snapshot():
    from .loader import memory_snapshot
    value = memory_snapshot()
    for name in ('memory.events', 'memory.events.local', 'memory.stat'):
        path = Path('/sys/fs/cgroup') / name
        try:
            counters = {parts[0]: int(parts[1]) for line in path.read_text().splitlines()
                        if len(parts := line.split()) == 2}
            if name == 'memory.stat':
                counters = {key: counters[key] for key in
                            ('anon', 'file', 'shmem', 'inactive_file', 'active_file', 'kernel')
                            if key in counters}
            value[name.replace('.', '_')] = counters
        except (OSError, ValueError):
            value[name.replace('.', '_')] = None
    try:
        path = Path('/sys/fs/cgroup/memory/memory.oom_control')
        value['memory_oom_control'] = path.read_text().strip()
    except OSError:
        value['memory_oom_control'] = None
    return value


def launch(run_path):
    from . import runtime
    output = Path(json.loads(Path(run_path).read_text(encoding='utf-8'))['output_root'])
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    before = resource_snapshot()
    atomic_json(logs / 'diagnostics_before.json', before)
    children, failure = [], {}

    def statuses():
        rows = []
        for rank, child in enumerate(children):
            code = child.raw.returncode
            name = None
            if code is not None and code < 0:
                try:
                    name = signal.Signals(-code).name
                except ValueError:
                    name = {9: 'SIGKILL', 15: 'SIGTERM'}.get(-code, f'signal_{-code}')
            rows.append(dict(rank=rank, pid=getattr(child.raw, 'pid', None),
                             returncode=code, signal=name,
                             terminated_by_parent=child.terminated_by_parent))
        return rows

    def publish():
        events = failure['resources_at_failure'].get('memory_events') or {}
        previous = before.get('memory_events') or {}
        delta = (events['oom_kill'] - previous['oom_kill']
                 if 'oom_kill' in events and 'oom_kill' in previous else None)
        report = dict(failure, workers=statuses(), resources_before=before, oom_kill_delta=delta)
        atomic_json(logs / 'worker_failure.json', report)

    class TrackedProcess:
        def __init__(self, *args, **kwargs):
            self.raw = _popen(*args, **kwargs)
            self.terminated_by_parent = False
            children.append(self)

        def __getattr__(self, name):
            return getattr(self.raw, name)

        def poll(self):
            code = self.raw.poll()
            if code not in (None, 0) and not self.terminated_by_parent and not failure:
                failure.update(timestamp=datetime.now(timezone.utc).isoformat(),
                               first_failed_rank=children.index(self),
                               resources_at_failure=resource_snapshot())
                publish()
                print(json.dumps(dict(phase='teacher_worker_failure', rank=children.index(self),
                      returncode=code, diagnostics=str(logs / 'worker_failure.json'))), flush=True)
            return code

        def terminate(self):
            self.terminated_by_parent = True
            return self.raw.terminate()

        def kill(self):
            self.terminated_by_parent = True
            return self.raw.kill()

    original = runtime.subprocess
    # Only this launcher's dependency is replaced, not the global subprocess module.
    runtime.subprocess = SimpleNamespace(Popen=TrackedProcess, TimeoutExpired=subprocess.TimeoutExpired)
    try:
        return runtime.launch(run_path)
    except BaseException as exc:
        if not failure:
            failure.update(timestamp=datetime.now(timezone.utc).isoformat(), first_failed_rank=None,
                           resources_at_failure=resource_snapshot())
        failure.update(exception=type(exc).__name__, message=str(exc))
        publish()
        raise
    finally:
        runtime.subprocess = original
        if failure:
            publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run'])
    parser.add_argument('--run', required=True)
    launch(parser.parse_args().run)


if __name__ == '__main__':
    main()
