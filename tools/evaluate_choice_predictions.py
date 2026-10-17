"""Standard-library offline scorer for literal-text Choice events, no model calls."""
import argparse,hashlib,json,math
from pathlib import Path

def exact(obj,keys,where):
 if not isinstance(obj,dict) or set(obj)!=set(keys):raise ValueError(where+': unexpected/missing fields')
def number(x,where):
 if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError(where+': expected finite number')
 return x
def identity(x,where):
 if not isinstance(x,str) or not x:raise ValueError(where+': expected nonempty string')
def close(a,b,where):
 if not math.isclose(number(a,where),b,rel_tol=0.,abs_tol=1e-8):raise ValueError(where+': inconsistent with logits')
def lse(xs):
 m=max(xs);return m+math.log(math.fsum(math.exp(x-m) for x in xs))
def load_jsonl(path):
 def pairs(items):
  obj={}
  for k,v in items:
   if k in obj:raise ValueError('Duplicate JSON key: '+k)
   obj[k]=v
  return obj
 def constant(x):raise ValueError('Nonstandard JSON number: '+x)
 raw=Path(path).read_bytes();rows=[]
 for i,line in enumerate(raw.decode('utf-8').splitlines(),1):
  if not line.strip():raise ValueError(str(path)+': blank line '+str(i))
  rows.append(json.loads(line,object_pairs_hook=pairs,parse_constant=constant))
 if not rows:raise ValueError(str(path)+': empty input')
 return rows,hashlib.sha256(raw).hexdigest()
def evaluate(references,predictions):
 if not references or not predictions:raise ValueError('Empty reference or prediction inventory')
 refs={};preds={}
 for row in references:
  exact(row,['question_id','candidates','correct_id'],'reference');qid=row['question_id'];identity(qid,'question_id')
  if qid in refs:raise ValueError('Duplicate reference question: '+qid)
  cs=row['candidates']
  if not isinstance(cs,list) or not 2<=len(cs)<=20:raise ValueError('Expected 2..20 reference candidate slots')
  ids=[]
  for c in cs:
   exact(c,['id','text'],'reference candidate');identity(c['id'],'candidate ID')
   if not isinstance(c['text'],str):raise ValueError('Candidate text must be literal string')
   ids.append(c['id'])
  if len(set(ids))!=len(ids):raise ValueError('Duplicate candidate IDs')
  identity(row['correct_id'],'correct_id')
  if row['correct_id'] not in ids:raise ValueError('Gold ID absent from candidates')
  refs[qid]=row
 for row in predictions:
  exact(row,['question_id','selected_id','selected_text','candidates','selected_text_probability'],'prediction');qid=row['question_id'];identity(qid,'prediction question_id')
  if qid in preds:raise ValueError('Duplicate prediction question: '+qid)
  preds[qid]=row
 if set(refs)!=set(preds):raise ValueError('Reference/prediction question inventories differ')
 records=[]
 for qid,ref in refs.items():
  pred=preds[qid];cs=ref['candidates'];pcs=pred['candidates']
  if not isinstance(pcs,list) or len(pcs)!=len(cs):raise ValueError(qid+': candidate count mismatch')
  z=[]
  for c,p in zip(cs,pcs):
   exact(p,['id','text','logit','probability'],'prediction candidate')
   if (p['id'],p['text'])!=(c['id'],c['text']):raise ValueError(qid+': candidate identity/text/order mismatch')
   z.append(number(p['logit'],'logit'))
  centered=[x-max(z) for x in z];normalizer=lse(centered);logp=[x-normalizer for x in centered];probs=[math.exp(x) for x in logp]
  for p,w in zip(pcs,probs):close(p['probability'],w,'slot probability')
  slot=max(range(len(z)),key=lambda i:z[i]);selected=cs[slot]
  if pred['selected_id']!=selected['id'] or pred['selected_text']!=selected['text']:raise ValueError(qid+': selection must be first maximum-logit slot')
  texts=list(dict.fromkeys(c['text'] for c in cs));event_lp={t:lse([logp[i] for i,c in enumerate(cs) if c['text']==t]) for t in texts};event_p={t:math.exp(v) for t,v in event_lp.items()}
  gold=next(c['text'] for c in cs if c['id']==ref['correct_id']);confidence=event_p[selected['text']];close(pred['selected_text_probability'],confidence,'selected text probability')
  correct=selected['text']==gold;nll=-event_lp[gold];brier=math.fsum((v-float(t==gold))**2 for t,v in event_p.items());number(nll,'NLL');number(brier,'Brier')
  records.append(dict(question_id=qid,correct=correct,designated_id_correct=selected['id']==ref['correct_id'],nll=nll,brier=brier,selected_text_confidence=confidence))
 n=len(records);bins=[[] for _ in range(15)]
 for r in records:bins[min(int(r['selected_text_confidence']*15),14)].append(r)
 ece=math.fsum(abs(math.fsum(r['selected_text_confidence']-float(r['correct']) for r in group)) for group in bins)/n
 return dict(schema_version=1,count=n,correct=sum(r['correct'] for r in records),accuracy=sum(r['correct'] for r in records)/n,nll=math.fsum(r['nll'] for r in records)/n,brier=math.fsum(r['brier'] for r in records)/n,ece15=ece,records=records,scope='Offline raw-logit metrics. First maximum slot predicts; literal duplicate-text mass is summed for probability metrics. No normalization of text, temperature fitting, confidence interval or calibrated-risk guarantee.')
def main():
 p=argparse.ArgumentParser();p.add_argument('--references',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 refs,rh=load_jsonl(a.references);preds,ph=load_jsonl(a.predictions);result=evaluate(refs,preds);result['input_sha256']={'references':rh,'predictions':ph}
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if __name__=='__main__':main()
