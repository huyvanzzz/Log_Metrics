import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


def _module():
    spec = importlib.util.spec_from_file_location('kaggle_source_setup', 'scripts/kaggle_source_setup.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def test_clone_then_pull_updates_entire_source_and_records_identity(tmp_path):
    helper = _module()
    upstream, target = tmp_path / 'upstream', tmp_path / 'source'
    upstream.mkdir()
    _git(upstream, 'init', '-b', 'main')
    _git(upstream, 'config', 'user.email', 'test@example.invalid')
    _git(upstream, 'config', 'user.name', 'Test')
    for package in ('smolvlm_attention', 'smolvlm_inference'):
        (upstream / package).mkdir()
        (upstream / package / '__init__.py').write_text('VERSION = 1\n')
    _git(upstream, 'add', '.')
    _git(upstream, 'commit', '-m', 'initial')
    identity = helper.sync_source(str(upstream), 'main', target)
    assert identity['commit'] == _git(upstream, 'rev-parse', 'HEAD')
    assert len(identity['sha256']) == 64
    (upstream / 'smolvlm_attention' / 'new.py').write_text('VERSION = 2\n')
    _git(upstream, 'add', '.')
    _git(upstream, 'commit', '-m', 'update')
    updated = helper.sync_source(str(upstream), 'main', target)
    assert (target / 'smolvlm_attention' / 'new.py').read_text() == 'VERSION = 2\n'
    assert updated['commit'] != identity['commit'] and updated['sha256'] != identity['sha256']
    (target / 'smolvlm_attention' / '__init__.py').write_text('LOCAL = True\n')
    with pytest.raises(RuntimeError, match='local changes'):
        helper.sync_source(str(upstream), 'main', target)


def test_sync_refuses_non_checkout_and_live_workers(tmp_path, monkeypatch):
    helper = _module()
    target = tmp_path / 'source'
    target.mkdir()
    with pytest.raises(RuntimeError, match='Git checkout'):
        helper.sync_source('unused', 'main', target)
    module = ModuleType('smolvlm_attention.runtime')
    class Process:
        def poll(self):
            return None
    module._PROCESSES = {'worker': Process()}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match='worker'):
        helper.ensure_no_workers([])


def test_activate_source_removes_old_package_imports(tmp_path, monkeypatch):
    helper = _module()
    monkeypatch.setitem(sys.modules, 'smolvlm_attention.old_test', ModuleType('smolvlm_attention.old_test'))
    previous = list(sys.path)
    try:
        helper.activate_source(tmp_path)
        assert 'smolvlm_attention.old_test' not in sys.modules
        assert sys.path[0] == str(tmp_path.resolve())
    finally:
        sys.path[:] = previous


def test_dino_worker_guard_and_optional_activation(tmp_path,monkeypatch):
    helper=_module()
    module=ModuleType('dinov2_attention.runtime')
    class Live:
        def poll(self): return None
    module._PROCESSES={0:Live()}
    monkeypatch.setitem(sys.modules,module.__name__,module)
    with pytest.raises(RuntimeError,match='worker'):
        helper.ensure_no_workers([])
    module._PROCESSES={}
    previous=list(sys.path)
    try:
        helper.activate_source(tmp_path,packages=('dinov2_attention',))
        assert module.__name__ not in sys.modules
    finally:
        sys.path[:]=previous
