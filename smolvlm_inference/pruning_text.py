"""Question-only masks on the unchanged native cache prompt."""
import torch
from .text import CachedBatchCollator, chat_messages
from .model import CAMERA_ORDER


class PruningBatchCollator(CachedBatchCollator):
    def __init__(self,processor,*,image_seq_len,max_length,generation=True,max_new_tokens=512,prompt_camera_order=CAMERA_ORDER):
        if list(prompt_camera_order)!=list(CAMERA_ORDER):
            raise ValueError('pruning requires canonical camera order')
        super().__init__(processor,image_seq_len=image_seq_len,max_length=max_length,generation=generation,max_new_tokens=max_new_tokens,prompt_camera_order=prompt_camera_order)

    def _encode_question(self,item):
        question=str(item['question'])
        if str(self.processor.image_token) in question:
            raise ValueError('question cannot contain image placeholders')
        if not question.strip():
            raise ValueError('question must not be empty')
        sentinel='__PRUNING_QUESTION_SPAN__'
        while sentinel in question:
            sentinel+='X'
        marked=self.processor.apply_chat_template(chat_messages(sentinel),tokenize=False,add_generation_prompt=True)
        if marked.count(sentinel)!=1:
            raise ValueError('native template must preserve question sentinel exactly once')
        before,after=marked.split(sentinel)
        before,after=self._expand(before+sentinel+after).split(sentinel)
        rendered=self._expand(self.processor.apply_chat_template(chat_messages(question),tokenize=False,add_generation_prompt=True))
        if rendered!=before+question+after:
            raise ValueError('native template changes question characters')
        encoded=self.processor.tokenizer(rendered,add_special_tokens=True,return_offsets_mapping=True)
        ids=encoded['input_ids']
        if len(ids)+self.max_new_tokens > self.max_length:
            raise ValueError('expanded chat exceeds context budget; truncation is disabled')
        image_id=self.processor.tokenizer.convert_tokens_to_ids(str(self.processor.image_token))
        if ids.count(image_id)!=6*self.image_seq_len:
            raise ValueError('native chat has incorrect expanded image token count')
        lo,hi=len(before),len(before)+len(question)
        overlap=[b>lo and a<hi and b>a and token!=image_id
                 for (a,b),token in zip(encoded['offset_mapping'],ids)]
        if not any(overlap):
            raise ValueError('question has no valid text tokens')
        return encoded,overlap

    def prompt_length(self,row):
        """Validate and measure native prompt tokens without reading visual features."""
        encoded,_=self._encode_question(row)
        return len(encoded['input_ids'])

    def __call__(self,items):
        clean=[{**item,'answer':''} for item in items]
        batch=super().__call__(clean)
        batch['answers']=[item.get('answer','') for item in items]
        ids=batch['input_ids']
        question_mask=torch.zeros_like(ids,dtype=torch.bool)
        positions=[]
        for row,item in enumerate(items):
            encoded,overlap=self._encode_question(item)
            start=int((batch['attention_mask'][row]==0).sum())
            length=len(encoded['input_ids'])
            if encoded['input_ids']!=ids[row,start:start+length].tolist():
                raise ValueError('offset tokenization differs from native prompt')
            overlap=torch.tensor(overlap,dtype=torch.bool)
            question_mask[row,start:start+length]=overlap
            image_id=self.processor.tokenizer.convert_tokens_to_ids(str(self.processor.image_token))
            image_positions=((ids[row]==image_id)&batch['attention_mask'][row].bool()).nonzero().flatten()
            positions.append(image_positions)
            question_mask[row,image_positions]=False
        question_mask &= batch['attention_mask'].bool()
        if not question_mask.any(1).all():
            raise ValueError('question has no valid text tokens')
        batch.update(question_mask=question_mask,image_positions=torch.stack(positions),
                     image_camera_ids=torch.arange(6).repeat_interleave(self.image_seq_len).expand(len(items),-1),
                     source_position_ids=(batch['attention_mask'].cumsum(1)-1).clamp_min(0))
        return batch
