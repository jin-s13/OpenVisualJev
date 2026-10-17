from pathlib import Path
from PIL import Image

def image_content(path,bottom_crop=0):
    if type(bottom_crop)!=int or bottom_crop not in [0,20]:raise ValueError('Supported bottom crops are 0 or 20 pixels')
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    if bottom_crop==0:return str(path)
    with Image.open(path) as raw:
        image=raw.convert('RGB')
        if image.height<=bottom_crop:raise ValueError(f'Image is too short for bottom crop: {path}')
        return image.crop((0,0,image.width,image.height-bottom_crop))
