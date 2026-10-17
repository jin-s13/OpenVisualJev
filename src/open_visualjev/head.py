import math
import torch
from torch import nn

class CandidateFFN(nn.Module):
    """Capacity control with no cross-candidate computation."""
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,6*dim),nn.GELU(),nn.Linear(6*dim,dim))
    def forward(self,x,src_key_padding_mask=None):
        return x+self.net(x)

class SetPointer(nn.Module):
    """Position-free candidate interaction and shared scoring; questions are batch rows."""
    def __init__(self, input_dim, dim=256, heads=4, layers=2, interaction="attention"):
        super().__init__()
        if dim % heads: raise ValueError('dim must be divisible by heads')
        self.input_projection=nn.Linear(input_dim,dim)
        if interaction not in ['attention','ffn']:raise ValueError('Unknown interaction')
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(dim,heads,dim*4,dropout=0.0,activation='gelu',batch_first=True,norm_first=True) if interaction=='attention' else CandidateFFN(dim) for _ in range(layers)])
        self.query=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,dim,bias=False))
        self.key=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim,bias=False))
        self.dim=dim
    def forward(self, query, candidates, valid):
        if query.ndim!=2 or candidates.ndim!=3 or valid.shape!=candidates.shape[:2]:
            raise ValueError('Expected query[B,D], candidates[B,K,D], valid[B,K]')
        if valid.dtype!=torch.bool: raise TypeError('valid must be boolean')
        if query.shape[0]!=candidates.shape[0] or query.shape[1]!=candidates.shape[2]:
            raise ValueError('Query/candidate dimension mismatch')
        if not valid.any(dim=1).all(): raise ValueError('Every question requires a valid candidate')
        if not torch.isfinite(query).all() or not torch.isfinite(candidates).all():
            raise ValueError('Non-finite embeddings')
        h=self.input_projection(candidates)
        for block in self.blocks: h=block(h,src_key_padding_mask=~valid)
        logits=torch.einsum('bd,bkd->bk',self.query(query),self.key(h))/math.sqrt(self.dim)
        return logits.masked_fill(~valid,float('-inf'))
