"""Portable supervised Set/Independent training on explicit Choice manifests.

This entry point trains on user-supplied candidate sets; see docs/training.md.
"""
import argparse
import json
import math
from pathlib import Path
import random

from run_choice_requests import validate_requests, digest
from evaluate_choice_predictions import load_jsonl, exact


def training_cases(requests, references, images):
    rows, paths, identity = validate_requests(requests, images)
    labels, label_hash = load_jsonl(references)
    by_id = {}
    for label in labels:
        exact(label, ['question_id', 'candidates', 'correct_id'], 'training reference')
        qid = label['question_id']
        if not isinstance(qid, str) or not qid or qid in by_id:
            raise ValueError('Missing or duplicate training reference question ID')
        by_id[qid] = label
    cases = []
    for row, path in zip(rows, paths):
        for q in row['questions']:
            if q['id'] not in by_id:
                raise ValueError('Missing training reference: ' + q['id'])
            label = by_id[q['id']]
            if label['candidates'] != q['candidates']:
                raise ValueError('Training candidate identity/order mismatch: ' + q['id'])
            names = [c['text'] for c in q['candidates']]
            if any(not name.strip() for name in names) or len(set(names)) != len(names):
                raise ValueError('Training requires nonblank, unique candidate texts')
            ids = [c['id'] for c in q['candidates']]
            if label['correct_id'] not in ids:
                raise ValueError('Training gold ID is absent: ' + q['id'])
            cases.append(dict(image=path, image_sha256=row['image_sha256'],
                              question=q['question'], candidates=names,
                              gold=ids.index(label['correct_id'])))
    if len(cases) != len(by_id):
        raise ValueError('Extra reference questions without training requests')
    return cases, dict(identity, references_sha256=label_hash)


def read_config(path):
    c = json.loads(Path(path).read_text())
    exact(c, ['arm', 'epochs', 'seed', 'learning_rate', 'weight_decay',
              'accumulation', 'max_grad_norm', 'warmup_ratio'], 'training config')
    if c['arm'] not in ['set', 'independent']:
        raise ValueError('arm must be set or independent')
    for k in ['epochs', 'seed', 'accumulation']:
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError(k + ' must be a positive integer')
    for k in ['learning_rate', 'weight_decay', 'max_grad_norm', 'warmup_ratio']:
        if type(c[k]) not in (float, int) or not math.isfinite(c[k]):
            raise ValueError(k + ' must be finite')
    if c['learning_rate'] <= 0 or c['weight_decay'] < 0 or c['max_grad_norm'] <= 0 or not 0 <= c['warmup_ratio'] < 1:
        raise ValueError('Invalid optimizer or warmup range')
    return c


def save_bundle(model, base, directory, source):
    import torch
    directory.mkdir()
    model.backbone.save_pretrained(directory / 'adapter')
    # A user supplies --base explicitly; avoid carrying a private machine path.
    adapter_path = directory / 'adapter/adapter_config.json'
    adapter = json.loads(adapter_path.read_text())
    adapter['base_model_name_or_path'] = 'Qwen/Qwen3-VL-4B-Instruct'
    adapter_path.write_text(json.dumps(adapter, indent=2))
    torch.save(model.readout.state_dict(), directory / 'readout.pt')
    base_files = {p.name: digest(p) for p in sorted(base.iterdir())
                  if p.is_file() and p.suffix in ['.json', '.txt', '.safetensors']}
    if not {'config.json', 'tokenizer_config.json', 'preprocessor_config.json'}.issubset(base_files) or not any(n.endswith('.safetensors') for n in base_files):
        raise ValueError('Base model is missing required config/tokenizer/weights files')
    files = {str(p.relative_to(directory)): digest(p) for p in directory.rglob('*') if p.is_file()}
    config = dict(schema_version=1, arm=source['arm'], layers=2 if source['arm']=='set' else 0,
                  max_pixels=262144, crop_bottom_pixels=0, base_files=base_files,
                  files=files, source=source)
    (directory / 'bundle.json').write_text(json.dumps(config, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['requests', 'references', 'images', 'config', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--base')
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError('Output already exists; use a new directory')
    config = read_config(a.config)
    cases, identity = training_cases(a.requests, a.references, a.images)
    if a.validate_only:
        out.mkdir(parents=True)
        (out/'preflight.json').write_text(json.dumps(identity, indent=2))
        print('Training manifests validated:', len(cases)); return
    if not a.base or not Path(a.base).is_dir():
        raise ValueError('Training requires an existing local --base model directory')
    import torch
    from transformers import get_cosine_schedule_with_warmup
    from open_visualjev.backbone import QwenSetPointer
    if not torch.cuda.is_available():
        raise RuntimeError('VLM training requires a CUDA device; set CUDA_VISIBLE_DEVICES explicitly')
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    model = QwenSetPointer(a.base, layers=2 if config['arm']=='set' else 0)
    trainable = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    total = config['epochs'] * math.ceil(len(cases)/config['accumulation'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total*config['warmup_ratio']), total)
    out.mkdir(parents=True)
    (out/'run.json').write_text(json.dumps(dict(config=config, inputs=identity,
        scope='User-supplied candidate sets; CE training; final-epoch export. Explicitly select checkpoints on separate development data for research comparisons.'), indent=2))
    step = 0
    with (out/'training.jsonl').open('x') as log:
        for epoch in range(config['epochs']):
            order = list(range(len(cases)))
            random.Random(config['seed'] + epoch).shuffle(order)
            model.train()
            for start in range(0, len(order), config['accumulation']):
                batch = order[start:start + config['accumulation']]
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
                for index in batch:
                    c = cases[index]
                    if digest(c['image']) != c['image_sha256']:
                        raise ValueError('Training image changed after validation: ' + str(c['image']))
                    scores = model(c['image'], c['question'], c['candidates'])
                    loss = torch.nn.functional.cross_entropy(scores, torch.tensor([c['gold']], device=scores.device))
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    (loss / len(batch)).backward()
                    loss_sum += float(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(trainable, config['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); step += 1
                log.write(json.dumps(dict(epoch=epoch+1, step=step, loss=loss_sum/len(batch), gradient_norm=float(norm)))+'\n'); log.flush()
                print('TRAIN', epoch+1, step, total, loss_sum/len(batch), flush=True)
    source = dict(arm=config['arm'], seed=config['seed'], epoch=config['epochs'],
                  checkpoint_receipt_sha256=digest(out/'training.jsonl'),
                  requests_sha256=identity['requests_sha256'], references_sha256=identity['references_sha256'])
    save_bundle(model, Path(a.base), out/'bundle', source)
    (out/'complete.json').write_text(json.dumps(dict(updates=step, bundle_sha256=digest(out/'bundle/bundle.json')), indent=2))


if __name__ == '__main__':
    main()
