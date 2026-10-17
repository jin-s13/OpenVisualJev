import torch
from open_visualjev import SetPointer

def test_cross_question_and_padding_jacobian_zero():
 torch.manual_seed(19);m=SetPointer(8,dim=8,heads=2,layers=2).double().eval();q=torch.randn(2,8,dtype=torch.double,requires_grad=True);c=torch.randn(2,4,8,dtype=torch.double,requires_grad=True);v=torch.tensor([[True,True,True,False],[True,True,True,True]])
 score=m(q,c,v)[0,0];gq,gc=torch.autograd.grad(score,(q,c));assert torch.count_nonzero(gq[1])==0 and torch.count_nonzero(gc[1])==0 and torch.count_nonzero(gc[0,3])==0
 assert gc[0,:3].abs().sum()>0

def test_independent_old_logit_odds_unchanged_on_extension():
 torch.manual_seed(20);m=SetPointer(8,dim=8,heads=2,layers=0).double().eval();q=torch.randn(1,8,dtype=torch.double);c=torch.randn(1,10,8,dtype=torch.double)
 s=m(q,c[:,:4],torch.ones(1,4,dtype=torch.bool));t=m(q,c,torch.ones(1,10,dtype=torch.bool));torch.testing.assert_close(s,t[:,:4],rtol=1e-12,atol=1e-12)
 torch.testing.assert_close((s.log_softmax(-1)[0,0]-s.log_softmax(-1)[0,1]),(t.log_softmax(-1)[0,0]-t.log_softmax(-1)[0,1]),rtol=1e-12,atol=1e-12)
 assert t.softmax(-1)[0,0]<s.softmax(-1)[0,0]

def test_head_only_k64_no_claim_on_vlm_interface():
 torch.manual_seed(21);m=SetPointer(8,dim=8,heads=2,layers=2).double().eval();q=torch.randn(1,8,dtype=torch.double);c=torch.randn(1,64,8,dtype=torch.double);v=torch.ones(1,64,dtype=torch.bool);p=torch.randperm(64)
 s=m(q,c,v);t=m(q,c[:,p],v[:,p]);assert s.shape==(1,64);torch.testing.assert_close(t,s[:,p],rtol=1e-10,atol=1e-11)
