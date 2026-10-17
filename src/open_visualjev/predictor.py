"""Pointer inference with ordinary batching or explicit dense tree execution."""
import hashlib,json
from pathlib import Path
import torch
from peft import PeftModel
from .backbone import QwenSetPointer
from .contract import validate_questions,decision
from .images import image_content

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

class VisualChoicePredictor:
    def __init__(self,base_model,bundle,device='cuda'):
        base_model=Path(base_model);bundle=Path(bundle);cfg=json.loads((bundle/'bundle.json').read_text())
        if set(cfg)!={'schema_version','arm','layers','max_pixels','crop_bottom_pixels','base_files','files','source'} or cfg['schema_version']!=1:raise ValueError('Unsupported bundle schema')
        if cfg['arm'] not in ['set','independent'] or cfg['layers']!={'set':2,'independent':0}[cfg['arm']]:raise ValueError('Unsupported pointer architecture')
        if cfg['max_pixels']!=262144 or cfg['crop_bottom_pixels']!=0:raise ValueError('This release supports the uncropped 262144-pixel protocol only')
        actual={str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file() and p.name!='bundle.json'}
        if set(cfg['files'])!=actual or not {'readout.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json'}.issubset(actual):raise ValueError('Incomplete or extra checkpoint files')
        for prefix,files in [(bundle,cfg['files']),(base_model,cfg['base_files'])]:
            if not files:raise ValueError('Missing content hashes')
            for name,h in files.items():
                if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Bundle paths must be relative')
                if digest(prefix/name)!=h:raise ValueError('Artifact hash mismatch: '+name)
        self.model=QwenSetPointer(base_model,device=device,lora=False,layers=cfg['layers'],max_pixels=cfg['max_pixels'],crop_bottom_pixels=0)
        self.model.backbone=PeftModel.from_pretrained(self.model.backbone,bundle/'adapter',is_trainable=False)
        self.model.readout.load_state_dict(torch.load(bundle/'readout.pt',map_location=device,weights_only=True),strict=True)
        self.model.requires_grad_(False);self.model.eval();self.config=cfg

    @torch.inference_mode()
    def predict(self,image,questions,*,crop_bottom_pixels=0,execution="ordinary",max_tree_tokens=4096):
        # Explicit per-request preprocessing; never mutate model crop state.
        validate_questions(questions)
        if execution not in ('ordinary','tree'):
            raise ValueError("execution must be 'ordinary' or 'tree'")
        if execution == 'tree':
            from .tree import score_tree_questions
            items=[dict(question=q['question'],candidates=[c['text'] for c in q['candidates']]) for q in questions]
            logits,_=score_tree_questions(self.model,image,items,max_tokens=max_tree_tokens,crop_bottom_pixels=crop_bottom_pixels)
            if len(logits)!=len(questions):raise ValueError('Tree result count mismatch')
            return [decision(q,z.cpu().tolist()) for q,z in zip(questions,logits)]
        prepared=image_content(image,crop_bottom_pixels);model=self.model;results=[]
        for q in questions:
            texts=[q['question']]+[q['question']+'\nCandidate: '+c['text'] for c in q['candidates']]
            conversations=[[dict(role='user',content=[dict(type='image',image=prepared),dict(type='text',text=t)])] for t in texts]
            inputs=model.processor.apply_chat_template(conversations,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True).to(model.device_name)
            out=model.backbone(**inputs,use_cache=False);mask=inputs['attention_mask'];last=torch.arange(mask.shape[1],device=mask.device).expand_as(mask).masked_fill(mask==0,-1).max(-1).values
            if (last<0).any():raise ValueError('Empty candidate/query branch')
            h=out.last_hidden_state[torch.arange(mask.shape[0],device=mask.device),last].float();k=len(q['candidates'])
            if h.shape[0]!=k+1:raise ValueError('Wrong encoded branch inventory')
            logits=model.readout(h[:1],h[1:].unsqueeze(0),torch.ones((1,k),dtype=torch.bool,device=h.device))[0]
            results.append(decision(q,logits.cpu().tolist()))
        return results
