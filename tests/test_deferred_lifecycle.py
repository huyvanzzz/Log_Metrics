"""Real cached native inference, durable paired resume, scoring and archive smoke."""
import copy
import json
import zipfile

import numpy as np
import torch

from smolvlm_inference.model import CachedSmolVLM
from smolvlm_inference.text import CachedBatchCollator
from smolvlm_topk_benchmark.generation import (fuse_batch, compact_prompt, SharedCausalLM,
    greedy_config, generate_compacted)
from smolvlm_topk_benchmark.runtime import generation_versions
from smolvlm_topk_benchmark.scoring import METRICS
from smolvlm_topk_deferred.generation import DeferredGreedyDecoder
from smolvlm_topk_deferred.runtime import (select_scores, schedule_batches, load_config,
    sources, score, bundle)
from smolvlm_topk_deferred.state import (CONFIGS, initialize_run, read_run, input_rows,
    write_selection, load_selection, save_batch, load_batches, finalize_config, validate_result)
from test_smolvlm_model import tiny_components


def test_native384_paired_interrupt_resume_metrics_archive(tmp_path):
    torch.manual_seed(42)
    native,processor=tiny_components()
    native.config.vision_config.image_size=64
    native.config.text_config.max_position_embeddings=2048
    native.model.text_model.layers=torch.nn.ModuleList([
        copy.deepcopy(native.model.text_model.layers[i%2]) for i in range(30)])
    native.config.text_config.num_hidden_layers=30
    for i,layer in enumerate(native.model.text_model.layers):
        layer.self_attn.layer_idx=i
    model=CachedSmolVLM(native,processor).eval()
    assert model.visual_token_len==384
    collator=CachedBatchCollator(processor,image_seq_len=64,max_length=2048,
                                 max_new_tokens=3,generation=True)
    values=np.linspace(-.3,.3,3*6*256*8,dtype=np.float32).reshape(3,6,256,8)
    values.astype(np.float16).tofile(tmp_path/'features.float16.memmap')
    cache=np.memmap(tmp_path/'features.float16.memmap',dtype=np.float16,mode='r',shape=values.shape)
    questions=['What is visible?','What changed? A person.','Where?']
    items=[dict(question=q,answer='A car.' if i!=1 else 'A person.',sample_id=f'smoke-{i}',
                eval_id=100+i,visual_features=torch.from_numpy(cache[i].astype(np.float32)))
           for i,q in enumerate(questions)]
    all_batch=collator(items)
    rows=[dict(row_index=i,sample_id=item['sample_id'],eval_id=item['eval_id'],answer=item['answer'],
               input_ids=all_batch['input_ids'][i][all_batch['attention_mask'][i].bool()].tolist(),
               image_token_id=native.config.image_token_id,attention_row=[2,0,1][i])
          for i,item in enumerate(items)]
    attention=np.zeros((3,30,384),dtype=np.float32)
    for source in range(3):
        attention[source,22]=np.roll(np.arange(384,dtype=np.float32),source*45)/100000
    selection=select_scores(attention,rows)
    identity=dict(batch_size=64,protocol=load_config(),sources=sources(),versions=generation_versions(),
                  test_fixture='real-native384-paired')
    path=initialize_run(tmp_path/'run',identity,rows,{})
    run=read_run(path)
    write_selection(run,selection)
    before={key:value.clone() for key,value in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    config=greedy_config(model,3)
    adapters=[SharedCausalLM(native),DeferredGreedyDecoder(model,2)]
    def generate(name,selected):
        batch=collator([items[row['row_index']] for row in selected])
        fused=fuse_batch(model,batch)
        keep=load_selection(run)[[row['row_index'] for row in selected]].astype(int).tolist()
        if name==CONFIGS[0]:
            values=generate_compacted(adapters[0],compact_prompt(fused,keep),processor.tokenizer,config)
        else:
            values=adapters[1].generate(fused,keep,config,debug=True)
            assert adapters[1].last_debug['visual_counts_by_layer']==[[384]*3]*3+[[32]*3]*27
            assert values==adapters[1].generate_reference(fused,keep,config)
        return [dict(value,sample_id=row['sample_id'],eval_id=row['eval_id'],reference=row['answer'])
                for row,value in zip(selected,values)]
    hook=native.model.vision_model.register_forward_pre_hook(
        lambda *args: (_ for _ in ()).throw(AssertionError('cache inference called vision')))
    try:
        # First worker finished before simulated session interruption; second has no commits.
        schedule_batches(rows,set(),lambda batch:generate(CONFIGS[0],batch),
                         lambda start,records:save_batch(run,CONFIGS[0],start,records))
        assert set(load_batches(run,CONFIGS[0]))=={0}
        assert load_batches(run,CONFIGS[1])=={}
        run=read_run(path)
        for name in CONFIGS:
            completed=set(load_batches(run,name))
            schedule_batches(input_rows(run),completed,lambda batch:generate(name,batch),
                             lambda start,records:save_batch(run,name,start,records))
            info,records=validate_result(finalize_config(run,name),run)
            assert info['count']==3 and len(records)==3
    finally:
        hook.remove()
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(value,before[key]) for key,value in model.state_dict().items())
    summary=score(path)
    assert set(summary['delta_after2_vs_input'])==set(METRICS)
    assert all(np.isfinite(result[metric]) for result in summary['results'] for metric in METRICS)
    archive=bundle(path)
    with zipfile.ZipFile(archive) as packed:
        names=packed.namelist()
        assert 'bundle_manifest.json' in names
        assert all('checkpoint' not in name and 'memmap' not in name and 'attention.npy' not in name for name in names)
        assert all(f'results/{name}/predictions.jsonl' in names for name in CONFIGS)
        manifest=json.loads(packed.read('bundle_manifest.json'))
        assert manifest['experiment_fingerprint']==run['fingerprint']
