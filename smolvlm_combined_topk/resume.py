"""Restore operational artifact paths without changing a measurement identity."""
import json
from pathlib import Path
import tempfile

from filelock import FileLock
from .store import atomic_json, digest, pid_is_alive


def _stat(path):
    value=Path(path).stat()
    return {'size':value.st_size,'mtime_ns':value.st_mtime_ns}


def rebase_compact_store(path, old_root, new_root, projected_directory):
    """Return a verified candidate; do not publish any file during validation."""
    path=Path(path); old_root=Path(old_root); new_root=Path(new_root)
    directory=Path(projected_directory)
    value=json.loads(path.read_text(encoding='utf-8'))
    receipt=json.loads((new_root/'inputs'/'input_identity.json').read_text(encoding='utf-8'))
    hashes=receipt.get('compact_content_sha256',{})
    for relative, expected in hashes.items():
        candidate=(new_root/'inputs'/relative).resolve()
        if not candidate.is_relative_to((new_root/'inputs').resolve()):
            raise ValueError('Compact content hash path escapes inputs directory')
        if digest(candidate)!=expected:
            raise ValueError(f'Compact input checksum mismatch: {relative}')
    def relocate(source):
        source=Path(source)
        try: return new_root/source.relative_to(old_root)
        except ValueError: return source
    compact_files={source for split in value['splits'].values() for source in split.values() if source}
    updated={}
    for source in compact_files:
        target=relocate(source)
        current=_stat(target)
        relative=target.relative_to(new_root/'inputs').as_posix() if target.is_relative_to(new_root/'inputs') else str(target)
        expected=hashes.get(relative,hashes.get(source))
        changed=str(target)!=source or current!=value['files'].get(source)
        if expected:
            if digest(target)!=expected: raise ValueError(f'Compact input checksum mismatch: {relative}')
        elif changed:
            raise ValueError('Compact input paths/stats changed but saved content hashes are unavailable; restore the original output paths/files or use a new OUTPUT.')
        updated[str(target)]=current
    value['splits']={split:{key:str(relocate(source)) for key,source in files.items()}
                     for split,files in value['splits'].items()}
    value['projected_manifest']=str(directory/'manifest.json')
    for name in ('manifest.json','projected.npy'):
        updated[str(directory/name)]=_stat(directory/name)
    value['files']=updated
    return value


def resume_artifacts(run, projected_input='', prepared_dir=None, checkpoint=None):
    """Resolve missing artifacts, verify provenance, then atomically refresh paths."""
    from .projected import (discover_projected_cache, open_projected_receipt,
                            validate_projected_cache)
    from smolvlm_predictor_io.projected import ProjectedCache
    from smolvlm_predictor_io.setup import resolve_artifacts
    root=Path(run['output_root']).resolve(); path=root/'combined_run.json'
    logs=root/'logs'; logs.mkdir(parents=True,exist_ok=True)
    with FileLock(str(logs/'launch.lock'),timeout=0):
        for pidfile in logs.glob('*.pid.json'):
            if pid_is_alive(json.loads(pidfile.read_text(encoding='utf-8'))['pid']):
                raise RuntimeError('Attention/benchmark worker still running; cannot refresh artifact paths.')
        old_store=Path(run['store']); old_root=old_store.parent.parent.parent
        store_path=root/'inputs'/'compact'/'store.json'
        receipt_path=root/'validation'/'projected_receipt.json'
        if not receipt_path.is_file() and Path(run['projected_receipt']).is_file():
            receipt_path=Path(run['projected_receipt'])
        previous=json.loads(receipt_path.read_text(encoding='utf-8'))
        explicit=projected_input or (previous['directory'] if Path(previous['directory']).is_dir() else '')
        directory=discover_projected_cache(explicit)
        cache=ProjectedCache(directory,full=False)
        identity=run['identity']
        if cache.manifest['fingerprint']!=identity['projected_fingerprint']:
            raise ValueError('Resume projected-cache fingerprint mismatch; choose the original cache.')
        existing=run['artifacts']['model']['checkpoint']
        checkpoint=checkpoint or (existing if Path(existing).is_file() else None)
        if prepared_dir is None:
            previous_prepared=Path(run['artifacts'].get('prepared_dir',root/'prepared_data'))
            prepared_dir=(root/'prepared_data' if (root/'prepared_data').is_dir()
                          else previous_prepared if previous_prepared.is_dir() else None)
        artifacts=resolve_artifacts(root,prepared_dir=prepared_dir,backbone_checkpoint=checkpoint,
                                    projected_input=str(directory))
        if digest(artifacts['model']['checkpoint'])!=identity['checkpoint_sha256']:
            raise ValueError('Resume checkpoint SHA-256 mismatch')
        for name,expected in previous['prepared_hashes'].items():
            if digest(Path(artifacts['prepared_dir'])/name)!=expected:
                raise ValueError(f'Resume prepared-data hash mismatch: {name}')
        candidate_store=rebase_compact_store(store_path,old_root,root,directory)
        reusable=False
        if str(directory)==previous['directory']:
            try:
                open_projected_receipt(receipt_path); reusable=True
            except (OSError,ValueError): pass
        destination=root/'validation'/'projected_receipt.json'
        if reusable:
            candidate_receipt=previous
        else:
            from smolvlm_inference.model import load_model, CAMERA_ORDER
            model=load_model(artifacts['model']['checkpoint'],device='cpu',expected_prompt_camera_order=CAMERA_ORDER)
            destination.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
                validated=validate_projected_cache(directory,model,artifacts['model']['checkpoint'],
                    artifacts['prepared_dir'],Path(temporary)/'receipt.json')
                candidate_receipt=json.loads(Path(validated).read_text(encoding='utf-8'))
            del model
        # All semantic checks finish before any operational file is replaced.
        atomic_json(destination,candidate_receipt)
        atomic_json(store_path,candidate_store)
        updated=dict(run,artifacts=artifacts,projected_receipt=str(destination),store=str(store_path))
        updated.pop('config_path',None)
        atomic_json(path,updated)
    return path
