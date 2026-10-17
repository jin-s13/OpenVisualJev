"""Strict ordinary image-Choice batch runner; no labels, implicit resume, or calibration."""
import argparse,hashlib,json,re
from pathlib import Path
from open_visualjev.contract import validate_questions,decision
from evaluate_choice_predictions import load_jsonl

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def validate_requests(path,images):
 root=Path(images).resolve()
 if not root.is_dir():raise ValueError('Image root is missing')
 rows,rh=load_jsonl(path);seen=set();cache={};image_paths=[]
 for row in rows:
  if not isinstance(row,dict) or set(row)!={'image','image_sha256','questions'}:raise ValueError('Each request requires exactly image/image_sha256/questions; no labels accepted')
  name=row['image']
  if not isinstance(name,str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Image path must stay relative to image root')
  p=(root/name).resolve()
  if not p.is_relative_to(root) or not p.is_file():raise ValueError('Missing image or image path escapes root: '+name)
  expected=row['image_sha256']
  if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('Expected lowercase SHA256 for image '+name)
  if p not in cache:cache[p]=digest(p)
  if cache[p]!=expected:raise ValueError('Image content changed: '+name)
  validate_questions(row['questions'])
  for q in row['questions']:
   if q['id'] in seen:raise ValueError('Duplicate question ID across requests: '+q['id'])
   seen.add(q['id'])
  image_paths.append(p)
 return rows,image_paths,dict(status='inputs_validated',requests=len(rows),questions=len(seen),images=len(cache),requests_sha256=rh,scope='Input schema, global question identity and image-byte hashes checked. No decoding, model execution, weight validation or metric evaluation.')
def write_predictions(rows,image_paths,path,predictor):
 if len(rows)!=len(image_paths):raise ValueError('Request/image alignment mismatch')
 count=0
 with Path(path).open('x') as f:
  for row,p in zip(rows,image_paths):
   if digest(p)!=row['image_sha256']:raise ValueError('Image changed after preflight: '+row['image'])
   results=predictor.predict(p,row['questions'])
   if not isinstance(results,list) or len(results)!=len(row['questions']):raise ValueError('Predictor returned incomplete/extra question inventory')
   for q,result in zip(row['questions'],results):
    if not isinstance(result,dict) or not isinstance(result.get('candidates'),list):raise ValueError('Predictor returned malformed decision')
    expected=decision(q,[c['logit'] for c in result['candidates']])
    if result!=expected:raise ValueError('Predictor decision fields or ordering differ from the public contract: '+q['id'])
    f.write(json.dumps(result,allow_nan=False)+'\n');f.flush();count+=1
 return count

def main():
 p=argparse.ArgumentParser();p.add_argument('--requests',required=True);p.add_argument('--images',required=True);p.add_argument('--output',required=True);p.add_argument('--base');p.add_argument('--bundle');p.add_argument('--validate-only',action='store_true');a=p.parse_args();out=Path(a.output)
 if out.exists():raise ValueError('Output directory exists; no implicit overwrite/resume')
 if not a.validate_only and (not a.base or not a.bundle):raise ValueError('Inference requires explicit --base and --bundle')
 rows,paths,preflight=validate_requests(a.requests,a.images)
 if a.validate_only:
  if a.base or a.bundle:raise ValueError('Input-only validation does not check base/bundle; omit these arguments')
  out.mkdir(parents=True,exist_ok=False)
  with (out/'preflight.json').open('x') as f:json.dump(preflight,f,indent=2)
  print(json.dumps(preflight));return
 from open_visualjev.predictor import VisualChoicePredictor
 bundle_hash=digest(Path(a.bundle)/'bundle.json')
 predictor=VisualChoicePredictor(a.base,a.bundle)
 if digest(Path(a.bundle)/'bundle.json')!=bundle_hash:raise ValueError('Bundle manifest changed during model loading')
 out.mkdir(parents=True,exist_ok=False)
 with (out/'preflight.json').open('x') as f:json.dump(preflight,f,indent=2)
 count=write_predictions(rows,paths,out/'predictions.jsonl',predictor)
 if count!=preflight['questions']:raise ValueError('Prediction count differs from validated inventory')
 if digest(Path(a.bundle)/'bundle.json')!=bundle_hash:raise ValueError('Bundle manifest changed during inference')
 complete=dict(status='inference_complete',questions=count,requests_sha256=preflight['requests_sha256'],predictions_sha256=digest(out/'predictions.jsonl'),bundle_sha256=bundle_hash,runner_sha256=digest(Path(__file__)),scope='Ordinary packaged raw-probability pointer inference. No scoring, temperature fitting, tree/cache path, or speed claim.')
 with (out/'complete.json').open('x') as f:json.dump(complete,f,indent=2)
 print(json.dumps(complete))
if __name__=='__main__':main()
