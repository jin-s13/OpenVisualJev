import copy
import pytest
from open_visualjev.contract import validate_questions,decision

def question():return dict(id='q',question='What?',candidates=[dict(id='x',text=''),dict(id='y',text='a'),dict(id='z',text='a')])
def test_duplicate_text_and_tie_are_explicit():
    q=question();r=decision(q,[0.,1.,1.]);assert r['selected_id']=='y';assert r['selected_text_probability']==sum(x['probability'] for x in r['candidates'][1:])
def test_ids_never_affect_scoring_mapping():
    q=question();a=decision(q,[0.,1.,2.]);q['id']='new';q['candidates'][2]['id']='external';b=decision(q,[0.,1.,2.]);assert b['selected_id']=='external';assert a['selected_text_probability']==b['selected_text_probability']
@pytest.mark.parametrize('mode',['extra_label','duplicate_id','bad_text','one_choice','extra_candidate_field'])
def test_bad_contract_rejected(mode):
    q=question()
    if mode=='extra_label':q['gold']=1
    if mode=='duplicate_id':q['candidates'][1]['id']='x'
    if mode=='bad_text':q['candidates'][0]['text']=None
    if mode=='one_choice':q['candidates']=q['candidates'][:1]
    if mode=='extra_candidate_field':q['candidates'][0]['label']=True
    with pytest.raises(ValueError):validate_questions([q])
def test_bad_logits_rejected():
    with pytest.raises(ValueError):decision(question(),[0.,float('nan'),0.])
