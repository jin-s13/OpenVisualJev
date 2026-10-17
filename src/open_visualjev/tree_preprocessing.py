"""Request-local identical-image preprocessing reuse for frozen shared paths.

The original processor still renders/tokenizes/expands multimodal tokens. Only
its image processor is replaced in a shallow, private copy. No persistent cache,
no mutation of the caller's processor, no truncation or token approximation.
Pixel tensors are repeated to preserve the existing encoder input contract.
"""
import copy
import torch
from transformers.feature_extraction_utils import BatchFeature

class SameImageProcessor:
    def __init__(self,original):self.original=original
    def __getattr__(self,name):return getattr(self.original,name)
    def __call__(self,*,images,**kwargs):
        if not isinstance(images,list) or not images or any(not isinstance(x,list) or len(x)!=1 for x in images):
            raise ValueError('Preprocess-once requires a nonempty batch with exactly one image per branch')
        first=images[0][0]
        for group in images:
            item=group[0]
            if item is not first and not (isinstance(first,str) and isinstance(item,str) and item==first):
                raise ValueError('Preprocess-once requires identical image paths or the same image object')
        result=self.original(images=images[:1],**kwargs)
        if set(result)!={'pixel_values','image_grid_thw'}:
            raise ValueError('Unsupported image processor output schema: '+str(sorted(result)))
        pixels=result['pixel_values'];grid=result['image_grid_thw']
        if not isinstance(pixels,torch.Tensor) or pixels.ndim!=2 or not isinstance(grid,torch.Tensor) or grid.shape!=(1,3) or pixels.shape[0]!=int(grid[0].prod()):
            raise ValueError('Expected torch image patches and one valid image grid')
        return BatchFeature(data=dict(pixel_values=pixels.repeat(len(images),1),image_grid_thw=grid.repeat(len(images),1)))

class PreprocessOnceModel:
    """Read-only model view accepted by shared and hierarchical scoring functions."""
    def __init__(self,model):
        if model.training:raise ValueError('Preprocess-once model view requires eval mode')
        self.original=model
        self.processor=copy.copy(model.processor)
        self.processor.image_processor=SameImageProcessor(model.processor.image_processor)
    def __getattr__(self,name):return getattr(self.original,name)


def fixed_question_boundary(query, probe, branches, image_end):
    """Boundary from fixed templates, independent of runtime candidate contents."""
    width=min(len(query),len(probe))
    different=torch.where(query[:width]!=probe[:width])[0]
    end=int(different[0]) if len(different) else width
    if end<=image_end or end>=len(query) or end>=len(probe):
        raise ValueError('Template references require a nonempty question prefix and suffix')
    if not branches or any(len(row)<=end or not torch.equal(row[:end],query[:end]) for row in branches):
        raise ValueError('Candidate tokenization does not preserve fixed question prefix')
    return end
