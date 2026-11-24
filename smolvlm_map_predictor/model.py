"""Small question-conditioned predictor over native visual tokens."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, rotate_half


class MapPredictor(nn.Module):
    def __init__(self, text_config, hidden_dim=128, projection_dim=64, visual_tokens=384):
        super().__init__()
        if projection_dim <= 0 or projection_dim % 2 or visual_tokens <= 0:
            raise ValueError('projection dimension must be positive and even; visual tokens positive')
        self.input_dim = int(text_config.hidden_size)
        self.hidden_dim, self.projection_dim, self.visual_tokens = hidden_dim, projection_dim, visual_tokens
        def mlp():
            return nn.Sequential(nn.Linear(self.input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.visual_mlp, self.question_mlp = mlp(), mlp()
        self.Wq = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, projection_dim, bias=False)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        config = copy.deepcopy(text_config)
        config.hidden_size = projection_dim
        config.num_attention_heads = 1
        if hasattr(config, 'head_dim'):
            config.head_dim = projection_dim
        self.rotary = LlamaRotaryEmbedding(config=config)
        self._rope = dict(rope_theta=getattr(text_config, 'rope_theta', 10000.0),
                          rope_scaling=getattr(text_config, 'rope_scaling', None),
                          max_position_embeddings=text_config.max_position_embeddings)

    @property
    def architecture_signature(self):
        return dict(input_dim=self.input_dim, hidden_dim=self.hidden_dim, projection_dim=self.projection_dim,
                    visual_tokens=self.visual_tokens, rope=self._rope, architecture='map_predictor_v1')

    def forward(self, visual, question, question_mask, visual_positions, question_positions):
        if (visual.ndim != 3 or question.ndim != 3 or visual.shape[0] != question.shape[0]
                or visual.shape[1:] != (self.visual_tokens, self.input_dim)
                or question.shape[2] != self.input_dim or question.shape[1] == 0
                or question_mask.shape != question.shape[:2] or visual_positions.shape != visual.shape[:2]
                or question_positions.shape != question.shape[:2] or not question_mask.bool().any(dim=1).all()
                or not torch.isfinite(visual).all() or not torch.isfinite(question).all()
                or (visual_positions < 0).any() or (question_positions < 0).any()):
            raise ValueError('invalid visual, question, mask, or position inputs')
        q = self.Wq(self.question_mlp(question))
        k = self.Wk(self.visual_mlp(visual))
        def rotate(x, positions):
            cos, sin = self.rotary(x, positions)
            return x * cos.to(x.dtype) + rotate_half(x) * sin.to(x.dtype)
        q, k = rotate(q, question_positions), rotate(k, visual_positions)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.projection_dim)
        per_question = scores.log_softmax(dim=-1).masked_fill(~question_mask.bool().unsqueeze(-1), -torch.inf)
        return torch.logsumexp(per_question, dim=1) - question_mask.sum(dim=1).to(scores.dtype).log().unsqueeze(-1)


def kl_per_sample(log_pred, log_target):
    dtype = torch.float64 if log_pred.dtype == torch.float64 or log_target.dtype == torch.float64 else torch.float32
    return F.kl_div(log_pred.to(dtype), log_target.to(dtype), reduction='none', log_target=True).sum(dim=-1)


def topk_indices(logscores, k=32):
    if logscores.ndim != 2 or not 0 < k <= logscores.shape[1] or not torch.isfinite(logscores).all():
        raise ValueError('invalid scores or k')
    return torch.argsort(logscores, dim=-1, descending=True, stable=True)[:, :k].sort(dim=-1).values


@torch.no_grad()
def frozen_features(model, batch):
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('native model must be frozen')
    device = next(model.native.model.connector.parameters()).device
    ids = batch['input_ids'].to(device)
    attention = batch['attention_mask'].to(device)
    mask = batch['question_mask'].to(device).bool()
    positions = batch['source_position_ids'].to(device)
    image_indices = batch['image_positions'].to(device)
    visual = model._project(batch['visual_features'].to(device=device, dtype=torch.float32), ids.shape[0]).reshape(ids.shape[0], -1, model.geometry['text_dim'])
    embedded = model.native.get_input_embeddings()(ids)
    counts = mask.sum(1)
    if not (counts > 0).all():
        raise ValueError('empty question')
    width = int(counts.max())
    question = embedded.new_zeros((ids.shape[0], width, embedded.shape[-1]))
    question_positions = positions.new_zeros((ids.shape[0], width))
    question_mask = torch.arange(width, device=device)[None, :] < counts[:, None]
    for i in range(ids.shape[0]):
        question[i, :counts[i]] = embedded[i, mask[i]]
        question_positions[i, :counts[i]] = positions[i, mask[i]]
    return dict(visual=visual, question=question, question_mask=question_mask,
                visual_positions=positions.gather(1, image_indices), question_positions=question_positions)
