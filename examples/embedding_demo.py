"""CPU structure demonstration with random embeddings, not a trained VLM result."""
import torch
from open_visualjev import SetPointer
torch.manual_seed(42)
head=SetPointer(16,dim=16,heads=4,layers=2).eval()
q=torch.randn(2,16);c=torch.randn(2,4,16);valid=torch.tensor([[True,True,True,True],[True,True,False,False]])
with torch.inference_mode():
    scores=head(q,c,valid);p=scores.softmax(-1)
assert p.shape==(2,4) and torch.allclose(p.sum(-1),torch.ones(2)) and torch.equal(p[1,2:],torch.zeros(2))
print('Random-embedding structure demo; no model-quality claim. Probabilities:',p.tolist())
