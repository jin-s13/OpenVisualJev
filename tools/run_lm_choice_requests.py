"""Dynamic-candidate direct LM baseline in the supplied candidate order.
Repository tool, not included in the existing pointer wheel. Raw probabilities.
"""
import argparse,json,math
from pathlib import Path
from open_visualjev.contract import validate_questions,decision
from run_choice_requests import validate_requests,write_predictions,digest

def prompt_and_order(question,ordering):
    validate_questions([question])
    cs=question['candidates']
    k=len(cs)
    if ordering!='native':raise ValueError('Only the supplied candidate order is supported')
    order=list(range(k))
    texts=[cs[i]['text'] for i in order]
    prompt=question['question']+'\nChoices:\n'+'\n'.join(f'{s}. {c}' for s,c in zip('ABCDEFGHIJKLMNOPQRST',texts))+'\nAnswer with the letter from the given choices directly.'
    return prompt,order

def align_logits(logits,order):
    if not 2<=len(order)<=20 or any(type(i)!=int for i in order) or sorted(order)!=list(range(len(order))):raise ValueError('Expected bijective candidate-slot order')
    if len(logits)!=len(order) or any(type(x) not in (int,float) or not math.isfinite(x) for x in logits):raise ValueError('Expected one finite logit per candidate')
    result=[None]*len(order)
    for new,original in enumerate(order):result[original]=logits[new]
    return result

def snapshot(directory):
    root=Path(directory).resolve()
    if not root.is_dir():raise ValueError('Missing local model/adapter directory')
    files={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink() and not p.resolve().is_relative_to(root):raise ValueError('Model file escapes supplied directory: '+str(p))
        if p.is_file():files[str(p.relative_to(root))]=digest(p)
    if not files:raise ValueError('Empty model/adapter directory')
    return files

class DirectLMChoicePredictor:
    def __init__(self,base,adapter,ordering):
        import torch
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        if ordering!='native':raise ValueError('Only native ordering is supported')
        self.ordering=ordering;self.torch=torch
        self.processor=AutoProcessor.from_pretrained(base,min_pixels=65536,max_pixels=262144,local_files_only=True)
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(base,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        if adapter is not None:
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,adapter,is_trainable=False,local_files_only=True)
        self.model=self.model.to('cuda').eval();self.model.requires_grad_(False)
        ids=[self.processor.tokenizer.encode(s,add_special_tokens=False) for s in 'ABCDEFGHIJKLMNOPQRST']
        if any(len(z)!=1 for z in ids) or len({z[0] for z in ids})!=20:raise ValueError('Candidate labels must map to twenty distinct single token IDs')
        self.symbol_ids=[z[0] for z in ids]
    def predict(self,image,questions):
        from open_visualjev.images import image_content
        validate_questions(questions)
        compiled=[prompt_and_order(q,self.ordering) for q in questions]
        prepared=image_content(image,0);results=[]
        with self.torch.inference_mode():
            for q,(prompt,order) in zip(questions,compiled):
                conversation=[dict(role='user',content=[dict(type='image',image=prepared),dict(type='text',text=prompt)])]
                inputs=self.processor.apply_chat_template(conversation,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
                scores=self.model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1,self.symbol_ids[:len(q['candidates'])]].float().cpu().tolist()
                results.append(decision(q,align_logits(scores,order)))
        return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--requests',required=True);p.add_argument('--images',required=True);p.add_argument('--output',required=True);p.add_argument('--base');p.add_argument('--adapter');p.add_argument('--ordering',choices=['native'],default='native');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise ValueError('Output exists; no implicit overwrite/resume')
    if a.validate_only and (a.base or a.adapter):raise ValueError('Input-only validation does not validate weights; omit base/adapter')
    if not a.validate_only and not a.base:raise ValueError('Local base model is required')
    rows,paths,preflight=validate_requests(a.requests,a.images)
    for row in rows:
        for q in row['questions']:prompt_and_order(q,a.ordering)
    preflight.update(ordering=a.ordering,baseline='dynamic_candidate_direct_lm')
    if a.validate_only:
        out.mkdir(parents=True,exist_ok=False);(out/'preflight.json').write_text(json.dumps(preflight,indent=2));print(json.dumps(preflight));return
    before={'base':snapshot(a.base),'adapter':snapshot(a.adapter) if a.adapter is not None else None}
    if 'config.json' not in before['base']:raise ValueError('Missing base config.json')
    if a.adapter is not None and not {'adapter_config.json','adapter_model.safetensors'}.issubset(before['adapter']):raise ValueError('Missing adapter configuration or weights')
    model=DirectLMChoicePredictor(a.base,a.adapter,a.ordering)
    out.mkdir(parents=True,exist_ok=False);(out/'preflight.json').write_text(json.dumps(preflight,indent=2));(out/'model-files.json').write_text(json.dumps(before,indent=2))
    count=write_predictions(rows,paths,out/'predictions.jsonl',model)
    if count!=preflight['questions']:raise ValueError('Incomplete inference inventory')
    after={'base':snapshot(a.base),'adapter':snapshot(a.adapter) if a.adapter is not None else None}
    if after!=before:raise ValueError('Model/adapter files changed during inference')
    complete=dict(status='inference_complete',baseline='dynamic_candidate_direct_lm',ordering=a.ordering,adapter_used=a.adapter is not None,questions=count,requests_sha256=preflight['requests_sha256'],predictions_sha256=digest(out/'predictions.jsonl'),model_files_sha256=digest(out/'model-files.json'),runner_sha256=digest(__file__),scope='Direct candidate-letter logits, aligned to original slots, raw probabilities. Supports 2–20 slots. No generation or temperature fitting. User-supplied adapter identity and training provenance require separate verification.')
    (out/'complete.json').write_text(json.dumps(complete,indent=2));print(json.dumps(complete))
if __name__=='__main__':main()
