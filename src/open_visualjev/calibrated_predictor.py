"""Explicit calibrated wrapper; the ordinary predictor keeps raw semantics."""
from .calibration import TemperatureCalibration
from .predictor import VisualChoicePredictor
class CalibratedVisualChoicePredictor:
    def __init__(self,base_model,bundle,calibration,device='cuda'):
        self.calibration=TemperatureCalibration(calibration,bundle)
        self.predictor=VisualChoicePredictor(base_model,bundle,device=device)
    def predict(self,image,questions,*,dataset=None,execution="ordinary",max_tree_tokens=4096):
        self.calibration.validate_questions(questions)
        crop=request_crop(self.predictor.config,dataset)
        raw=self.predictor.predict(image,questions,crop_bottom_pixels=crop,execution=execution,max_tree_tokens=max_tree_tokens)
        if len(raw)!=len(questions):raise ValueError('Predictor result count mismatch')
        outputs=[]
        for q,r in zip(questions,raw):
            result=self.calibration.apply(q,[c['logit'] for c in r['candidates']])
            if result['raw']!=r:raise ValueError('Predictor result disagrees with raw calibration contract')
            outputs.append(result)
        return outputs


def request_crop(config,dataset):
    source=config['source']
    if 'request_crop_bottom_pixels' not in source:
        if dataset is not None:raise ValueError('Legacy calibrated bundles require the uncropped protocol; dataset override is unsupported')
        return 0
    expected={'pets':0,'cub':0,'cars':0,'dogs':0,'flowers':0,'aircraft':20}
    mapping=source['request_crop_bottom_pixels']
    if not isinstance(mapping,dict) or set(mapping)!=set(expected) or any(type(mapping[k]) is not int or mapping[k]!=v for k,v in expected.items()):raise ValueError('Invalid unified dataset crop protocol')
    if type(dataset) is not str or dataset not in mapping:raise ValueError('Unified calibrated prediction requires an explicit supported dataset')
    return mapping[dataset]
