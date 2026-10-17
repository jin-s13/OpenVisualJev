"""Strict image + independent questions + externally named candidates contract."""
import math

def validate_questions(questions):
    if not isinstance(questions,list) or not questions:raise ValueError('questions must be a nonempty list')
    seen=set()
    for q in questions:
        if not isinstance(q,dict) or set(q)!={'id','question','candidates'}:raise ValueError('Question requires exactly id/question/candidates; labels are not accepted')
        if not isinstance(q['id'],str) or not q['id'] or q['id'] in seen:raise ValueError('Question IDs must be nonempty and unique')
        seen.add(q['id'])
        if not isinstance(q['question'],str) or not q['question'].strip():raise ValueError('Question text must be nonempty')
        cs=q['candidates']
        if not isinstance(cs,list) or not 2<=len(cs)<=20:raise ValueError('Current VLM interface supports 2..20 candidate slots')
        ids=set()
        for c in cs:
            if not isinstance(c,dict) or set(c)!={'id','text'}:raise ValueError('Candidate requires exactly id/text')
            if not isinstance(c['id'],str) or not c['id'] or c['id'] in ids:raise ValueError('Candidate IDs must be nonempty and unique within each question')
            if not isinstance(c['text'],str):raise ValueError('Candidate text must be a literal string; blank and duplicate texts are preserved')
            ids.add(c['id'])
    return questions

def decision(q,logits):
    validate_questions([q])
    if len(logits)!=len(q['candidates']) or any(not math.isfinite(x) for x in logits):raise ValueError('Expected one finite logit per candidate')
    weights=[math.exp(x-max(logits)) for x in logits];total=math.fsum(weights);p=[x/total for x in weights];best=max(range(len(logits)),key=lambda i:logits[i]);chosen=q['candidates'][best]
    return dict(question_id=q['id'],selected_id=chosen['id'],selected_text=chosen['text'],candidates=[dict(c,logit=float(z),probability=v) for c,z,v in zip(q['candidates'],logits,p)],selected_text_probability=math.fsum(v for c,v in zip(q['candidates'],p) if c['text']==chosen['text']))
