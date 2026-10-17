import pytest
import torch
from open_visualjev import SetPointer
@pytest.fixture
def fixture():
    torch.manual_seed(73)
    return SetPointer(32,dim=16,heads=4,layers=2).eval(),torch.randn(2,32),torch.randn(2,10,32)
def test_permutation_and_padding(fixture):
    model,q,c=fixture
    mask=torch.ones(2,10,dtype=torch.bool); mask[0,7:]=False
    order=torch.tensor([9,0,5,2,8,1,6,4,3,7])
    original=model(q,c,mask).softmax(-1)
    permuted=model(q,c[:,order],mask[:,order]).softmax(-1)
    torch.testing.assert_close(permuted,original[:,order],rtol=1e-5,atol=1e-6)
    assert original[0,7:].eq(0).all()
    torch.testing.assert_close(original.sum(-1),torch.ones(2))
def test_question_isolation(fixture):
    model,q,c=fixture; mask=torch.ones(2,10,dtype=torch.bool)
    first=model(q[:1],c[:1],mask[:1])
    c[1]*=20; q[1]*=-10
    torch.testing.assert_close(first,model(q,c,mask)[:1],rtol=1e-5,atol=1e-6)
def test_padding_does_not_change_valid_logits(fixture):
    model,q,c=fixture
    short=model(q,c[:,:4],torch.ones(2,4,dtype=torch.bool))
    mask=torch.zeros(2,10,dtype=torch.bool); mask[:,:4]=True
    torch.testing.assert_close(short,model(q,c,mask)[:,:4],rtol=1e-5,atol=1e-6)
def test_gradient_and_invalid_input(fixture):
    model,q,c=fixture; mask=torch.ones(2,10,dtype=torch.bool)
    torch.nn.functional.cross_entropy(model(q,c,mask),torch.tensor([0,3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    mask[0]=False
    with pytest.raises(ValueError,match='valid candidate'): model(q,c,mask)
    with pytest.raises(TypeError): model(q,c,mask.float())
