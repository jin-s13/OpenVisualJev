"""Dense tree-mask single-forward inference; frozen weights and existing readout.

One root image prefix, one prefix per question, isolated causal query/candidate
leaves. MRoPE positions are gathered from each original full branch, never from
packed offsets. Dense masked attention is not sparse computation.
"""
from pathlib import Path
import torch
from .tree_preprocessing import fixed_question_boundary
from .images import image_content
from .tree_preprocessing import PreprocessOnceModel


def pack_tree(rows, positions, spans, question_ends, image_end, *, max_tokens=4096):
    if not rows or len(rows)!=len(positions) or len(spans)!=len(question_ends) or not spans:
        raise ValueError('Inconsistent tree inputs')
    if type(image_end)!=int or image_end<1 or type(max_tokens)!=int or max_tokens<1:raise ValueError('Invalid tree bounds')
    expected=0
    for (start,k,probe),end in zip(spans,question_ends):
        if start!=expected or type(k)!=int or k<1 or probe!=start+k+1 or probe>=len(rows):raise ValueError('Invalid branch partition')
        expected=probe+1
        if not image_end<end<min(len(rows[i]) for i in range(start,probe+1)):raise ValueError('Empty question prefix or leaf')
        if any(not torch.equal(rows[i][:end],rows[start][:end]) or not torch.equal(positions[i][:,:end],positions[start][:,:end]) for i in range(start,probe+1)):raise ValueError('Question prefix tokens/positions differ')
    if expected!=len(rows):raise ValueError('Unassigned input branches')
    for row,pos in zip(rows,positions):
        if row.ndim!=1 or pos.shape!=(3,len(row)):raise ValueError('Invalid token/position shape')
        if not torch.equal(row[:image_end],rows[0][:image_end]) or not torch.equal(pos[:,:image_end],positions[0][:,:image_end]):raise ValueError('Image prefix tokens/positions differ')
    tokens=[rows[0][:image_end]];coords=[positions[0][:,:image_end]];segments=[(0,image_end,[])];leaves=[];offset=image_end
    for (s,k,_),end in zip(spans,question_ends):
        qstart=offset;qlength=end-image_end;tokens.append(rows[s][image_end:end]);coords.append(positions[s][:,image_end:end]);offset+=qlength
        root=list(range(image_end));segments.append((qstart,offset,root));ancestors=root+list(range(qstart,offset));local=[]
        for i in range(s,s+k+1):
            length=len(rows[i])-end;tokens.append(rows[i][end:]);coords.append(positions[i][:,end:]);segments.append((offset,offset+length,ancestors));offset+=length;local.append(offset-1)
        leaves.append(local)
    if offset>max_tokens:raise ValueError(f'Dense tree has {offset} tokens, exceeding explicit limit {max_tokens}; no truncation applied')
    allowed=torch.zeros((offset,offset),dtype=torch.bool,device=rows[0].device)
    for start,end,ancestors in segments:
        allowed[start:end,start:end]=torch.ones((end-start,end-start),dtype=torch.bool,device=allowed.device).tril()
        if ancestors:allowed[start:end,ancestors]=True
    return dict(input_ids=torch.cat(tokens).unsqueeze(0),position_ids=torch.cat(coords,dim=1).unsqueeze(1),allowed=allowed,leaf_ends=leaves,segments=segments,packed_tokens=offset)


@torch.inference_mode()
def score_tree_questions(model,image,questions,*,max_tokens=4096,crop_bottom_pixels=0):
    if model.training:raise ValueError('Tree inference requires eval mode')
    if not Path(image).is_file():raise FileNotFoundError(image)
    if not isinstance(questions,list) or not questions:raise ValueError('Expected nonempty questions')
    texts=[];spans=[]
    for item in questions:
        if not isinstance(item,dict) or set(item)!={'question','candidates'}:raise ValueError('Expected question and candidates')
        q=item['question'];c=item['candidates']
        if not isinstance(q,str) or not q.strip() or not isinstance(c,list) or not c or not all(isinstance(x,str) and x.strip() for x in c) or len(set(c))!=len(c):raise ValueError('Invalid question or candidates')
        s=len(texts);texts.extend([q]+[q+'\nCandidate: '+x for x in c]);probe=len(texts);texts.append(q+'\nCandidate: ');spans.append((s,len(c),probe))
    view=PreprocessOnceModel(model);prepared=image_content(image,crop_bottom_pixels)
    messages=[[{'role':'user','content':[{'type':'image','image':prepared},{'type':'text','text':t}]}] for t in texts]
    inputs=view.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True).to(model.device_name)
    if set(inputs)!={'input_ids','attention_mask','pixel_values','image_grid_thw'}:raise ValueError('Unsupported tree processor schema')
    ids=inputs['input_ids'];mask=inputs['attention_mask'];grids=inputs['image_grid_thw'];pixels=inputs['pixel_values'];n=len(texts)
    if grids.shape!=(n,3) or not torch.equal(grids,grids[:1].expand_as(grids)):raise ValueError('Expected same image per branch')
    patches=int(grids[0].prod())
    if pixels.shape[0]!=n*patches or not torch.equal(pixels.reshape(n,patches,-1),pixels[:patches].unsqueeze(0).expand(n,-1,-1)):raise ValueError('Image patches differ')
    base=model.backbone.get_base_model() if hasattr(model.backbone,'get_base_model') else model.backbone
    if base.language_model.config._attn_implementation!='sdpa':raise ValueError('Tree prototype requires SDPA with explicit4D mask')
    positions,_=base.get_rope_index(ids,grids,attention_mask=mask)
    rows=[r[m.bool()] for r,m in zip(ids,mask)];coords=[positions[:,i,m.bool()] for i,m in enumerate(mask)]
    ends=torch.where(rows[0]==base.config.vision_end_token_id)[0]
    if len(ends)!=1:raise ValueError('Expected one image boundary')
    image_end=int(ends[0])+1
    if any(bool((row[image_end:]==base.config.image_token_id).any()) for row in rows):raise ValueError('Image token outside shared root')
    boundaries=[fixed_question_boundary(rows[s],rows[p],rows[s:s+k+1],image_end) for s,k,p in spans]
    tree=pack_tree(rows,coords,spans,boundaries,image_end,max_tokens=max_tokens)
    dtype=base.get_input_embeddings().weight.dtype
    attention=torch.zeros(tree['allowed'].shape,dtype=dtype,device=ids.device).masked_fill(~tree['allowed'],torch.finfo(dtype).min)[None,None]
    output=model.backbone(input_ids=tree['input_ids'],position_ids=tree['position_ids'],attention_mask=attention,pixel_values=pixels[:patches],image_grid_thw=grids[:1],use_cache=False)
    scores=[]
    for leaf in tree['leaf_ends']:
        reps=output.last_hidden_state[0,leaf].float();k=len(leaf)-1
        z=model.readout(reps[:1],reps[1:].unsqueeze(0),torch.ones((1,k),dtype=torch.bool,device=ids.device))[0]
        if z.shape!=(k,) or not torch.isfinite(z).all():raise ValueError('Invalid tree logits')
        scores.append(z)
    return scores,dict(packed_tokens=tree['packed_tokens'],dense_attention_cells=tree['packed_tokens']**2,unshared_tokens=sum(len(rows[s+i]) for s,k,_ in spans for i in range(k+1)),image_prefix_tokens=image_end,question_prefix_tokens=[e-image_end for e in boundaries],backbone_calls=1,scope='Dense tree mask, original MRoPE, one image/deepstack injection; no persistent cache; preprocessing once with repeated tensor contract')
