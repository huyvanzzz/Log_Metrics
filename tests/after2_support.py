"""Real tiny native fixtures for the separate post-block-2 experiment."""
import json


def make_inputs(root):
    from map_support import make_inputs as original_inputs
    from smolvlm_map_predictor_after2.data import prepare_inputs
    root.mkdir(parents=True,exist_ok=True)
    model,common = original_inputs(root)
    identity = json.loads(common.read_text(encoding='utf-8'))['identity']
    receipt = prepare_inputs(root/'targets',root/'cache',root/'prepared',model,root/'after2_inputs',
        expected_counts={'train':3,'val':3,'test':3},checkpoint_sha256=identity['checkpoint_sha256'])
    return model,receipt


def tiny_config():
    from smolvlm_map_predictor_after2.config import load_config
    cfg=load_config()
    cfg['architecture'].update(hidden_dim=8,projection_dim=4)
    cfg['training'].update(world_size=1,batch_size=2,epochs=2,num_workers=0,pin_memory=False,
                           persistent_workers=False,progress_log_every_steps=1)
    cfg['evaluation'].update(batch_size=2,max_new_tokens=2)
    return cfg
