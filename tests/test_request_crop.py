"""CPU regression for the real predictor request path; no VLM quality claim."""
from types import SimpleNamespace
from PIL import Image
import pytest
import torch
from open_visualjev.predictor import VisualChoicePredictor


class Inputs(dict):
    def to(self, device):
        assert device == 'cpu'
        return self


def predictor():
    seen = []

    def prepare(conversations, **kwargs):
        heights = []
        for conversation in conversations:
            image = conversation[0]['content'][0]['image']
            if isinstance(image, str):
                with Image.open(image) as im:
                    heights.append(im.height)
            else:
                heights.append(image.height)
        seen.append(heights)
        return Inputs(attention_mask=torch.ones(len(heights), 1, dtype=torch.long),
                      heights=torch.tensor(heights, dtype=torch.float32))

    def backbone(attention_mask, heights, use_cache):
        assert use_cache is False
        return SimpleNamespace(last_hidden_state=heights[:, None, None])

    p = VisualChoicePredictor.__new__(VisualChoicePredictor)
    p.model = SimpleNamespace(processor=SimpleNamespace(apply_chat_template=prepare),
        device_name='cpu', backbone=backbone,
        readout=lambda q, c, valid: c[:, :, 0], crop_bottom_pixels=0)
    return p, seen


def questions():
    return [dict(id='q', question='Which object?',
                 candidates=[dict(id='a', text='cat'), dict(id='b', text='dog')])]


def test_actual_predict_path_crop_is_request_local(tmp_path):
    image = tmp_path / 'image.png'
    Image.new('RGB', (32, 48)).save(image)
    p, seen = predictor()
    first = p.predict(image, questions())
    cropped = p.predict(image, questions(), crop_bottom_pixels=20)
    after = p.predict(image, questions())
    assert seen == [[48] * 3, [28] * 3, [48] * 3]
    assert first == after
    assert [c['logit'] for c in cropped[0]['candidates']] == [28., 28.]
    assert p.model.crop_bottom_pixels == 0
    with Image.open(image) as original:
        assert original.size == (32, 48)


@pytest.mark.parametrize('crop', [True, -1, 1, 20., '20', None])
def test_invalid_crop_fails_before_encoding(tmp_path, crop):
    p, seen = predictor()
    with pytest.raises(ValueError, match='Supported bottom crops'):
        p.predict(tmp_path / 'absent.png', questions(), crop_bottom_pixels=crop)
    assert seen == []


def test_too_short_image_fails_before_encoding(tmp_path):
    image = tmp_path / 'short.png'
    Image.new('RGB', (32, 20)).save(image)
    p, seen = predictor()
    with pytest.raises(ValueError, match='too short'):
        p.predict(image, questions(), crop_bottom_pixels=20)
    assert seen == []
