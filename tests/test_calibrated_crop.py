"""Protocol and forwarding checks; no claim about fitted temperature quality."""
from types import SimpleNamespace
import pytest
from open_visualjev.calibrated_predictor import request_crop, CalibratedVisualChoicePredictor
from open_visualjev.contract import decision

MAPPING = dict(pets=0,cub=0,cars=0,dogs=0,flowers=0,aircraft=20)


def test_protocol_refuses_implicit_or_invalid_preprocessing():
    cfg = {'source': {'request_crop_bottom_pixels': MAPPING}}
    for ds, crop in MAPPING.items():
        assert request_crop(cfg, ds) == crop
    for ds in [None, '', 'unknown', True, []]:
        with pytest.raises(ValueError):
            request_crop(cfg, ds)
    assert request_crop({'source': {}}, None) == 0
    with pytest.raises(ValueError):
        request_crop({'source': {}}, 'aircraft')
    for mapping in [{}, {**MAPPING, 'aircraft': 0}, {**MAPPING, 'pets': False}]:
        with pytest.raises(ValueError):
            request_crop({'source': {'request_crop_bottom_pixels': mapping}}, 'pets')


def test_real_wrapper_forwards_request_local_crop():
    calls = []
    q = dict(id='q', question='Choose', candidates=[dict(id=str(i), text=str(i)) for i in range(4)])
    raw = decision(q, [0.,1.,2.,3.])
    def predict(image, questions, *, crop_bottom_pixels, execution, max_tree_tokens):
        assert execution == "ordinary" and max_tree_tokens == 4096
        calls.append(crop_bottom_pixels)
        return [raw]
    model = CalibratedVisualChoicePredictor.__new__(CalibratedVisualChoicePredictor)
    model.predictor = SimpleNamespace(config={'source': {'request_crop_bottom_pixels': MAPPING}}, predict=predict)
    model.calibration = SimpleNamespace(validate_questions=lambda qs:None,
        apply=lambda question, logits:dict(raw=decision(question,logits)))
    for ds in ['pets','aircraft','pets']:
        assert model.predict('unused.png',[q],dataset=ds) == [{'raw':raw}]
    assert calls == [0,20,0]
    with pytest.raises(ValueError):
        model.predict('unused.png',[q])
    assert calls == [0,20,0]


def test_calibrated_wrapper_forwards_tree_execution():
    calls=[]
    q=dict(id='q',question='Choose',candidates=[dict(id=str(i),text=str(i)) for i in range(2)])
    def predict(image,questions,**kwargs):
        calls.append(kwargs)
        return [decision(q,[0.,1.])]
    model=CalibratedVisualChoicePredictor.__new__(CalibratedVisualChoicePredictor)
    model.predictor=SimpleNamespace(config={'source':{'request_crop_bottom_pixels':MAPPING}},predict=predict)
    model.calibration=SimpleNamespace(validate_questions=lambda qs:None,apply=lambda question,logits:dict(raw=decision(question,logits)))
    model.predict('unused.png',[q],dataset='aircraft',execution='tree',max_tree_tokens=123)
    assert calls==[dict(crop_bottom_pixels=20,execution='tree',max_tree_tokens=123)]
