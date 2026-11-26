"""MMR-only adapters around the unchanged hard-prune generation paths."""
import torch
from .selector import MMRSelector


def _portable(value):
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value,dict): return {key:_portable(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)): return [_portable(item) for item in value]
    return value


class Decoder:
    def __init__(self,backbone,predictor,variant,keep_tokens=32):
        if variant not in ('before','after2'): raise ValueError('Unknown predictor variant')
        self.backbone,self.predictor,self.variant=backbone,predictor,variant
        self.keep_tokens=keep_tokens
        self.selector=MMRSelector(weight=.9)
        self.last_debug={}
        if variant=='after2':
            from smolvlm_map_predictor_after2.generation import PredictorGreedyDecoder
            self.engine=PredictorGreedyDecoder(backbone,predictor,keep_tokens,selection_fn=self.selector)

    def _validate(self):
        if self.predictor.training or self.backbone.training:
            raise ValueError('MMR generation requires eval models')
        device=next(self.predictor.parameters()).device.type
        if (torch.is_autocast_enabled(device) or
            any(p.dtype!=torch.float32 for p in self.predictor.parameters()) or
            any(p.is_floating_point() and p.dtype!=torch.float32 for p in self.backbone.parameters())):
            raise ValueError('MMR generation requires FP32 without autocast')

    def _debug(self,debug):
        self.last_debug={}
        if debug:
            base=self.engine.last_debug if self.variant=='after2' else dict(predictor_calls=1)
            self.last_debug=_portable(dict(base,mmr={k:v for k,v in self.selector.last_debug.items() if k!='relevance'}))

    @torch.inference_mode()
    def generate(self,batch,max_new_tokens=512,debug=False):
        self._validate()
        if self.variant=='after2':
            records=self.engine.generate(batch,max_new_tokens,debug)
        else:
            from smolvlm_map_predictor.evaluation import predict_batch
            records=predict_batch(self.backbone,self.predictor,batch,keep_tokens=self.keep_tokens,
                                  max_new_tokens=max_new_tokens,selection_fn=self.selector)
        self._debug(debug)
        return records

    @torch.inference_mode()
    def generate_reference(self,batch,max_new_tokens=512,debug=False):
        self._validate()
        if self.variant=='after2':
            records=self.engine.generate_reference(batch,max_new_tokens,debug)
        else:
            from smolvlm_map_predictor.evaluation import predict_batch
            from smolvlm_topk_benchmark.generation import SharedCausalLM
            class NoCacheAdapter(SharedCausalLM):
                def generate(self,*args,**kwargs):
                    kwargs['use_cache']=False
                    return super().generate(*args,**kwargs)
            adapter=NoCacheAdapter(self.backbone.native)
            records=predict_batch(self.backbone,self.predictor,batch,keep_tokens=self.keep_tokens,
                max_new_tokens=max_new_tokens,selection_fn=self.selector,adapter=adapter)
        self._debug(debug)
        return records
