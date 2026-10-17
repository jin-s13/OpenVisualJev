import copy,json,math
import pytest
from open_visualjev.calibration import TemperatureCalibration,digest,METHOD
from open_visualjev.contract import decision

def fixture(tmp_path,t=2.):
 b=tmp_path/'bundle';b.mkdir();(b/'bundle.json').write_text(json.dumps({'arm':'set','source':{'checkpoint_receipt_sha256':'a'*64,'seed':7}}))
 a={'schema_version':1,'method':METHOD,'candidate_count':4,'temperature':t,'bundle_sha256':digest(b/'bundle.json'),'source':{'arm':'set','seed':7,'checkpoint_receipt_sha256':'a'*64,'registration_sha256':'b'*64,'calibration_manifest_sha256':'c'*64,'temperature_artifact_sha256':'d'*64,'audit_sha256':'e'*64,'fit_count':852}}
 p=tmp_path/'calibration.json';p.write_text(json.dumps(a));q={'id':'q','question':'Choose','candidates':[{'id':str(i),'text':s} for i,s in enumerate(['same','same','','other'])]};return p,b,a,q

def test_scale_slots_before_adding_duplicate_texts(tmp_path):
 p,b,a,q=fixture(tmp_path);z=[math.log(9),math.log(4),0.,math.log(2)];r=TemperatureCalibration(p,b).apply(q,z);normalizer=6+math.sqrt(2)
 assert r['raw']==decision(q,z)
 assert r['calibrated']['selected_id']=='0'
 assert r['calibrated']['selected_text_probability']==pytest.approx(5/normalizer)
 assert [c['probability'] for c in r['calibrated']['candidates']]==pytest.approx([3/normalizer,2/normalizer,1/normalizer,math.sqrt(2)/normalizer])
 assert r['calibrated']['selected_text_probability']!=pytest.approx(math.sqrt(13)/(math.sqrt(13)+1+math.sqrt(2)))

@pytest.mark.parametrize('t',[0.,-1.,True,float('nan'),float('inf')])
def test_bad_temperature_refused(tmp_path,t):
 p,b,a,q=fixture(tmp_path,t)
 with pytest.raises(ValueError,match='finite and positive'):TemperatureCalibration(p,b)

def test_wrong_bundle_refused(tmp_path):
 p,b,a,q=fixture(tmp_path);(b/'bundle.json').write_text((b/'bundle.json').read_text()+' ')
 with pytest.raises(ValueError,match='different model'):TemperatureCalibration(p,b)

def test_wrong_checkpoint_refused(tmp_path):
 p,b,a,q=fixture(tmp_path);a['source']['checkpoint_receipt_sha256']='f'*64;p.write_text(json.dumps(a))
 with pytest.raises(ValueError,match='checkpoint/seed'):TemperatureCalibration(p,b)

@pytest.mark.parametrize('field,value',[('candidate_count',10),('method','text_sum_before_temperature'),('schema_version',True)])
def test_wrong_contract_refused(tmp_path,field,value):
 p,b,a,q=fixture(tmp_path);a[field]=value;p.write_text(json.dumps(a))
 with pytest.raises(ValueError):TemperatureCalibration(p,b)

def test_unknown_field_refused(tmp_path):
 p,b,a,q=fixture(tmp_path);a['fallback_temperature']=1;p.write_text(json.dumps(a))
 with pytest.raises(ValueError,match='exact artifact'):TemperatureCalibration(p,b)

def test_different_candidate_count_refused(tmp_path):
 p,b,a,q=fixture(tmp_path);q['candidates'].pop()
 with pytest.raises(ValueError,match='four candidate'):TemperatureCalibration(p,b).apply(q,[1.,2.,3.])

def test_tie_and_blank_text_kept(tmp_path):
 p,b,a,q=fixture(tmp_path);r=TemperatureCalibration(p,b).apply(q,[0.,0.,0.,0.]);assert r['raw']['selected_id']==r['calibrated']['selected_id']=='0';assert r['calibrated']['selected_text_probability']==.5

def test_nonfinite_scaling_refused(tmp_path):
 p,b,a,q=fixture(tmp_path,.001)
 with pytest.raises(ValueError,match='nonfinite'):TemperatureCalibration(p,b).apply(q,[1e308,-1e308,0.,0.])
