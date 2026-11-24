"""Predictor-only artifact resolution; projected Input never resolves raw cache."""
from pathlib import Path

def resolve_artifacts(output, cache_input='', prepared_dir=None, backbone_checkpoint=None, *, projected_input='', projected_dir=''):
    from smolvlm_attention.config import MODELS, PREPARED_REPO, PREPARED_REVISION
    from huggingface_hub import hf_hub_download
    complete = bool(projected_input) or bool(projected_dir and (Path(projected_dir)/'manifest.json').is_file())
    if not complete:
        if backbone_checkpoint:
            if not cache_input or not prepared_dir:
                raise ValueError('Local base requires local raw cache/prepared data when building projected cache')
            return dict(model=dict(checkpoint=str(Path(backbone_checkpoint).resolve())),
                        cache_dir=str(Path(cache_input).resolve()),prepared_dir=str(Path(prepared_dir).resolve()))
        from smolvlm_attention.sample_runtime import resolve_artifacts as original
        return original(output,cache_input,prepared_dir)
    prepared=Path(prepared_dir).resolve() if prepared_dir else Path(output)/'prepared_data'
    for name in ('manifest.json','train.jsonl','val.jsonl','test.jsonl'):
        if not (prepared/name).is_file():
            hf_hub_download(repo_id=PREPARED_REPO,repo_type='dataset',revision=PREPARED_REVISION,
                            filename=name,local_dir=prepared,token=False)
    checkpoint=str(Path(backbone_checkpoint).resolve()) if backbone_checkpoint else hf_hub_download(
        repo_id=MODELS['base']['repo'],revision=MODELS['base']['revision'],filename='finetune_best.pt',token=False)
    return dict(model=dict(checkpoint=checkpoint),prepared_dir=str(prepared),cache_dir=None)
