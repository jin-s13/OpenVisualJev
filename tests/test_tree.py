import pytest
import torch
from open_visualjev.tree import pack_tree

def data():
 rows=[torch.tensor(r) for r in [[1,2,3,9],[1,2,3,5,9],[1,2,3,6],[1,2,4,9],[1,2,4,7,9],[1,2,4,6]]]
 pos=[torch.arange(len(r)).repeat(3,1) for r in rows]
 return rows,pos,[(0,1,2),(3,1,5)],[3,3]
def test_tree_attention_exact_ancestor_paths():
 t=pack_tree(*data(),2);a=t['allowed'];assert t['packed_tokens']==10
 # image[0,1], q0[2], query0[3], candidate0[4,5], q1[6], query1[7], candidate1[8,9]
 expected=[[0],[0,1],[0,1,2],[0,1,2,3],[0,1,2,4],[0,1,2,4,5],[0,1,6],[0,1,6,7],[0,1,6,8],[0,1,6,8,9]]
 for i,want in enumerate(expected):assert torch.where(a[i])[0].tolist()==want
 assert t['leaf_ends']==[[3,5],[7,9]]
 assert t['position_ids'][0,0].tolist()==[0,1,2,3,3,4,2,3,3,4]

def test_dense_budget_is_explicit_failure():
 with pytest.raises(ValueError,match='no truncation'):pack_tree(*data(),2,max_tokens=9)

def test_mismatched_question_positions_rejected():
 r,p,s,e=data();p[1][0,2]=99
 with pytest.raises(ValueError,match='positions differ'):pack_tree(r,p,s,e,2)

def test_invalid_partition_rejected():
 r,p,s,e=data();s[1]=(2,1,4)
 with pytest.raises(ValueError,match='partition'):pack_tree(r,p,s,e,2)

def test_sdpa_mask_blocks_sibling_values_and_gradients():
 t=pack_tree(*data(),2);n=t['packed_tokens'];allowed=t['allowed'];values=torch.arange(n,dtype=torch.float32).reshape(1,1,n,1).requires_grad_();q=torch.zeros((1,1,n,1));mask=torch.zeros(n,n).masked_fill(~allowed,torch.finfo(torch.float32).min)
 output=torch.nn.functional.scaled_dot_product_attention(q,q,values,attn_mask=mask[None,None],is_causal=False)
 # First candidate's endpoint sees root0,1, question2 and candidate4,5 only.
 target=output[0,0,5,0];assert torch.allclose(target,torch.tensor((0+1+2+4+5)/5))
 target.backward();assert torch.where(values.grad.reshape(-1)!=0)[0].tolist()==[0,1,2,4,5]

def test_tree_predictor_dispatch(monkeypatch):
    from types import SimpleNamespace
    from open_visualjev.predictor import VisualChoicePredictor
    import open_visualjev.tree as module
    predictor=VisualChoicePredictor.__new__(VisualChoicePredictor)
    predictor.model=SimpleNamespace()
    questions=[{'id':'q','question':'Which?', 'candidates':[{'id':'a','text':'red'},{'id':'b','text':'blue'}]}]
    calls=[]
    def score(model,image,items,**kwargs):
        calls.append((image,items,kwargs))
        return [torch.tensor([1.,2.])],{}
    monkeypatch.setattr(module,'score_tree_questions',score)
    out=predictor.predict('image.png',questions,execution='tree',crop_bottom_pixels=20,max_tree_tokens=99)
    assert len(out)==1
    assert calls==[('image.png',[{'question':'Which?','candidates':['red','blue']}],{'max_tokens':99,'crop_bottom_pixels':20})]
    with pytest.raises(ValueError,match='execution'):
        predictor.predict('image.png',questions,execution='unknown')


def test_fixed_boundary_rejects_token_mismatch():
    from open_visualjev.tree_preprocessing import fixed_question_boundary
    q=torch.tensor([1,2,3,9]);p=torch.tensor([1,2,3,6])
    assert fixed_question_boundary(q,p,[q,torch.tensor([1,2,3,5])],2)==3
    with pytest.raises(ValueError,match='tokenization'):
        fixed_question_boundary(q,p,[torch.tensor([1,2,4,5])],2)
