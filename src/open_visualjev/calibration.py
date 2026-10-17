"""Explicit frozen four-slot temperature scaling, bound to one model bundle."""
import hashlib,json,math,re
from pathlib import Path
from .contract import decision,validate_questions
METHOD='slot_temperature_then_literal_text_sum'
SOURCE_FIELDS={'arm','seed','checkpoint_receipt_sha256','registration_sha256','calibration_manifest_sha256','temperature_artifact_sha256','audit_sha256','fit_count'}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_artifact(a,bundle):
    if not isinstance(a,dict) or set(a)!={'schema_version','method','candidate_count','temperature','bundle_sha256','source'}:raise ValueError('Calibration requires exact artifact schema')
    if type(a['schema_version']) is not int or a['schema_version']!=1 or a['method']!=METHOD:raise ValueError('Unsupported calibration method/version')
    if type(a['candidate_count']) is not int or a['candidate_count']!=4:raise ValueError('Only registered four-slot calibration is supported')
    t=a['temperature']
    if type(t) not in (int,float) or not math.isfinite(t) or t<=0:raise ValueError('Temperature must be finite and positive')
    src=a['source']
    if not isinstance(src,dict) or set(src)!=SOURCE_FIELDS:raise ValueError('Incomplete calibration source identity')
    for k in ['seed','fit_count']:
        if type(src[k]) is not int or src[k]<=0:raise ValueError('Invalid calibration '+k)
    for v in [a['bundle_sha256']]+[src[k] for k in SOURCE_FIELDS if k.endswith('_sha256')]:
        if not isinstance(v,str) or re.fullmatch('[0-9a-f]{64}',v) is None:raise ValueError('Invalid calibration artifact digest')
    path=Path(bundle)/'bundle.json'
    if digest(path)!=a['bundle_sha256']:raise ValueError('Temperature belongs to a different model bundle')
    b=json.loads(path.read_text())
    if src['arm'] not in ['set','independent'] or src['arm']!=b['arm']:raise ValueError('Calibration architecture mismatch')
    if src['checkpoint_receipt_sha256']!=b['source']['checkpoint_receipt_sha256'] or src['seed']!=b['source']['seed']:raise ValueError('Calibration checkpoint/seed mismatch')
    return a
class TemperatureCalibration:
    def __init__(self,path,bundle):
        self._artifact=validate_artifact(json.loads(Path(path).read_text()),bundle)
        self.artifact_sha256=digest(path)
    def validate_questions(self,questions):
        validate_questions(questions)
        if any(len(q['candidates'])!=4 for q in questions):raise ValueError('This temperature was fitted for four candidate slots only')
    def apply(self,q,logits):
        self.validate_questions([q]);raw=decision(q,logits);t=self._artifact['temperature'];maximum=max(logits);scaled=[(z-maximum)/t for z in logits]
        if any(not math.isfinite(z) for z in scaled):raise ValueError('Temperature scaling produced nonfinite logits')
        calibrated=decision(q,scaled)
        if calibrated['selected_id']!=raw['selected_id']:raise ValueError('Positive temperature changed selected slot')
        return {'raw':raw,'calibrated':calibrated,'calibration':{'artifact_sha256':self.artifact_sha256,'bundle_sha256':self._artifact['bundle_sha256'],'method':METHOD,'temperature':t,'candidate_count':4,'probability_semantics':'conditional on supplied slots; duplicate literal-text probabilities summed after scaling; no per-decision correctness or risk guarantee'}}
