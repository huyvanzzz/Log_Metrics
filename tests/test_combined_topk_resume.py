import json
from pathlib import Path
import pytest
from smolvlm_combined_topk.resume import rebase_compact_store
from smolvlm_combined_topk.store import digest
from types import SimpleNamespace


def fixture(tmp_path):
    old=tmp_path/'old'; new=tmp_path/'new'; directory=new/'inputs'/'compact'; directory.mkdir(parents=True)
    payload=directory/'test.tokens.bin'; payload.write_bytes(b'123')
    oldfile=old/'inputs'/'compact'/payload.name
    store=dict(version=2,splits={'test':{'tokens':str(oldfile)}},files={str(oldfile):{'size':3,'mtime_ns':0}},
               projected_manifest=str(old/'cache'/'manifest.json'))
    path=directory/'store.json'; path.write_text(json.dumps(store))
    receipt=new/'inputs'/'input_identity.json'
    receipt.write_text(json.dumps({'compact_content_sha256':{'compact/test.tokens.bin':digest(payload)}}))
    cache=new/'cache'; cache.mkdir(); (cache/'manifest.json').write_text('{}'); (cache/'projected.npy').write_bytes(b'x')
    return old,new,path,cache


def test_rebase_requires_content_and_updates_only_operational_paths(tmp_path):
    old,new,path,cache=fixture(tmp_path)
    result=rebase_compact_store(path,old,new,cache)
    assert result['splits']['test']['tokens']==str(new/'inputs'/'compact'/'test.tokens.bin')
    assert result['projected_manifest']==str(cache/'manifest.json')
    assert json.loads(path.read_text())['splits']['test']['tokens'].startswith(str(old))


def test_rebase_rejects_corrupt_or_unverifiable_files_before_writing(tmp_path):
    old,new,path,cache=fixture(tmp_path); original=path.read_bytes()
    (new/'inputs'/'compact'/'test.tokens.bin').write_bytes(b'456')
    with pytest.raises(ValueError,match='checksum'): rebase_compact_store(path,old,new,cache)
    assert path.read_bytes()==original
    (new/'inputs'/'input_identity.json').write_text('{}')
    with pytest.raises(ValueError,match='content hashes'): rebase_compact_store(path,old,new,cache)


def test_resume_redownloads_missing_checkpoint_and_preserves_identity(tmp_path,monkeypatch):
    from smolvlm_combined_topk.resume import resume_artifacts
    from smolvlm_combined_topk import projected
    import smolvlm_predictor_io.projected as neutral
    import smolvlm_predictor_io.setup as setup
    old,new,path,cache=fixture(tmp_path)
    checkpoint=new/'checkpoint.pt'; checkpoint.write_bytes(b'weights')
    prepared=new/'prepared_data'; prepared.mkdir(); (prepared/'test.jsonl').write_text('data')
    receipt=new/'validation'/'projected_receipt.json'; receipt.parent.mkdir()
    receipt.write_text(json.dumps(dict(directory=str(cache),prepared_hashes={'test.jsonl':digest(prepared/'test.jsonl')})))
    identity=dict(projected_fingerprint='fp',checkpoint_sha256=digest(checkpoint))
    run=dict(output_root=str(new),store=str(old/'inputs'/'compact'/'store.json'),projected_receipt=str(receipt),
        identity=identity,source_commit='original',artifacts={'model':{'checkpoint':str(old/'gone.pt')}})
    called=[]
    monkeypatch.setattr(projected,'discover_projected_cache',lambda value:cache)
    monkeypatch.setattr(projected,'open_projected_receipt',lambda value:None)
    monkeypatch.setattr(neutral,'ProjectedCache',lambda *a,**k:SimpleNamespace(manifest={'fingerprint':'fp'}))
    def resolve(*args,**kwargs):
        called.append(kwargs)
        return dict(model={'checkpoint':str(checkpoint)},prepared_dir=str(prepared))
    monkeypatch.setattr(setup,'resolve_artifacts',resolve)
    result=resume_artifacts(run)
    assert called[0]['backbone_checkpoint'] is None
    saved=json.loads(result.read_text())
    assert saved['identity']==identity and saved['source_commit']=='original'
    assert saved['store']==str(path)
    assert json.loads(path.read_text())['projected_manifest']==str(cache/'manifest.json')


def test_resume_mismatched_checkpoint_never_updates_files(tmp_path,monkeypatch):
    from smolvlm_combined_topk.resume import resume_artifacts
    from smolvlm_combined_topk import projected
    import smolvlm_predictor_io.projected as neutral
    import smolvlm_predictor_io.setup as setup
    old,new,path,cache=fixture(tmp_path)
    checkpoint=new/'wrong.pt'; checkpoint.write_bytes(b'wrong')
    receipt=new/'validation'/'projected_receipt.json'; receipt.parent.mkdir()
    receipt.write_text(json.dumps(dict(directory=str(cache),prepared_hashes={})))
    run=dict(output_root=str(new),store=str(old/'inputs'/'compact'/'store.json'),projected_receipt=str(receipt),
        identity=dict(projected_fingerprint='fp',checkpoint_sha256='expected'),artifacts={'model':{'checkpoint':str(checkpoint)}})
    monkeypatch.setattr(projected,'discover_projected_cache',lambda value:cache)
    monkeypatch.setattr(neutral,'ProjectedCache',lambda *a,**k:SimpleNamespace(manifest={'fingerprint':'fp'}))
    monkeypatch.setattr(setup,'resolve_artifacts',lambda *a,**k:dict(model={'checkpoint':str(checkpoint)},prepared_dir=str(new)))
    original=path.read_bytes()
    with pytest.raises(ValueError,match='checkpoint SHA'): resume_artifacts(run)
    assert path.read_bytes()==original and not (new/'combined_run.json').exists()
