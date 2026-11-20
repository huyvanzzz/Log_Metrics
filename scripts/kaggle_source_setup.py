"""Standard-library-only Git setup, embedded verbatim in both Kaggle notebooks."""
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def ensure_no_workers(output_roots):
    for name in ('smolvlm_attention.runtime', 'smolvlm_attention.generated_runtime',
                 'dinov2_attention.runtime', 'smolvlm_probing.runtime',
                 'smolvlm_attention.sample_runtime', 'smolvlm_topk_benchmark.runtime'):
        module = sys.modules.get(name)
        if module is not None and any(process.poll() is None for process in getattr(module, '_PROCESSES', {}).values()):
            raise RuntimeError('A measurement worker is still running. Wait for it to finish before pulling source.')
    # The canonical probing output is shared by all notebooks. Also inspect it
    # after a kernel restart, when the new runtime is no longer in sys.modules.
    roots = set(map(Path, output_roots)) | {Path('/kaggle/working/smolvlm_probing_outputs'),
                                          Path('/kaggle/working/smolvlm_prompt_attention_outputs'),
                                          Path('/kaggle/working/smolvlm_topk_benchmark_outputs')}
    for root in roots:
        for path in (Path(root) / 'logs').glob('*.pid.json'):
            pid = json.loads(path.read_text(encoding='utf-8'))['pid']
            try:
                os.kill(pid, 0)
            except (OSError, ProcessLookupError):
                continue
            raise RuntimeError(f'A measurement worker PID {pid} may still be running ({path}). Finish it before pulling.')


def sync_source(repository, branch, directory, packages=('smolvlm_inference', 'smolvlm_attention')):
    directory = Path(directory).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()
    if directory.exists():
        if not (directory / '.git').exists():
            raise RuntimeError(f'{directory} is not a Git checkout. Choose a new SOURCE directory.')
        if git('remote', 'get-url', 'origin').rstrip('/') != str(repository).rstrip('/'):
            raise RuntimeError('Existing source checkout has a different origin; choose a new SOURCE directory.')
        if git('status', '--porcelain', '--untracked-files=no'):
            raise RuntimeError('Source checkout has local changes. Preserve them before pulling; no files were reset.')
        if git('branch', '--show-current') != branch:
            raise RuntimeError('Checkout branch differs from SOURCE_BRANCH. Use the matching branch or a new SOURCE directory.')
        subprocess.run(['git', '-C', str(directory), 'pull', '--ff-only', 'origin', branch], check=True)
    else:
        directory.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--depth', '1', '--single-branch', '--branch', branch,
                        str(repository), str(directory)], check=True)
    # Keep the historical source_archive_sha256 field reproducible without embedding stale source.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for package in packages:
            folder = directory / package
            if not folder.is_dir():
                raise RuntimeError(f'Missing source package after pull: {package}')
            for path in sorted(folder.iterdir()):
                if path.suffix in ('.py', '.txt'):
                    entry = zipfile.ZipInfo(path.relative_to(directory).as_posix(), (2026, 1, 1, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(entry, path.read_bytes())
    return dict(commit=git('rev-parse', 'HEAD'), sha256=hashlib.sha256(buffer.getvalue()).hexdigest())


def activate_source(directory, packages=('smolvlm_inference', 'smolvlm_attention')):
    directory = Path(directory).resolve()
    for package in packages:
        for path in (directory / package).glob('*.py'):
            Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)
    for name in list(sys.modules):
        if any(name == package or name.startswith(package + '.')
               for package in packages):
            del sys.modules[name]
    sys.path[:] = [str(directory)] + [entry for entry in sys.path if entry != str(directory)]
    importlib.invalidate_caches()
