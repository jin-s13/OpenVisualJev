from pathlib import Path
import torch
from torch import nn
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model
from .head import SetPointer
from .images import image_content

class QwenSetPointer(nn.Module):
    def __init__(self, model_path, device='cuda', max_pixels=512*512, lora=True, layers=2, interaction="attention", crop_bottom_pixels=0):
        super().__init__()
        if type(crop_bottom_pixels)!=int or crop_bottom_pixels not in [0,20]:raise ValueError("Invalid crop protocol")
        self.crop_bottom_pixels=crop_bottom_pixels
        self.processor=AutoProcessor.from_pretrained(model_path,min_pixels=256*256,max_pixels=max_pixels)
        full=Qwen3VLForConditionalGeneration.from_pretrained(model_path,dtype=torch.bfloat16,attn_implementation='sdpa')
        backbone=full.model
        backbone.requires_grad_(False)
        if lora:
            config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,bias='none',target_modules=r'language_model.layers.\d+.self_attn.(q_proj|k_proj|v_proj|o_proj)')
            backbone=get_peft_model(backbone,config)
        self.backbone=backbone.to(device)
        self.readout=SetPointer(full.config.text_config.hidden_size,layers=layers,interaction=interaction).to(device)
        self.device_name=device
        del full
    def encode_question(self,image,question,candidates):
        if not candidates or len(set(candidates))!=len(candidates): raise ValueError('Empty or duplicate candidates')
        if not Path(image).is_file(): raise FileNotFoundError(image)
        texts=[question]+[question+'\nCandidate: '+candidate for candidate in candidates]
        prepared_image=image_content(image,self.crop_bottom_pixels)
        conversations=[[{'role':'user','content':[{'type':'image','image':prepared_image},{'type':'text','text':text}]}] for text in texts]
        inputs=self.processor.apply_chat_template(conversations,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True).to(self.device_name)
        self.last_input_tokens=int(inputs['attention_mask'].sum())
        outputs=self.backbone(**inputs,use_cache=False)
        mask=inputs['attention_mask']
        last=torch.arange(mask.shape[1],device=mask.device).expand_as(mask).masked_fill(mask==0,-1).max(-1).values
        if (last<0).any(): raise ValueError('Empty token sequence')
        representations=outputs.last_hidden_state[torch.arange(mask.shape[0],device=mask.device),last].float()
        return representations[:1],representations[1:].unsqueeze(0)
    def forward(self,image,question,candidates):
        query,encoded=self.encode_question(image,question,candidates)
        valid=torch.ones(1,len(candidates),device=query.device,dtype=torch.bool)
        return self.readout(query,encoded,valid)
