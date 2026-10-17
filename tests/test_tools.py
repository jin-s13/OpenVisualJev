import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from train_choices import training_cases, read_config
from run_lm_choice_requests import prompt_and_order, align_logits


def question(k=4):
    return dict(id='q', question='Which color?', candidates=[dict(id=str(i), text=f'color {i}') for i in range(k)])


@pytest.mark.parametrize('k', [2, 4, 10, 20])
def test_dynamic_lm(k):
    prompt, order = prompt_and_order(question(k), 'native')
    assert order == list(range(k))
    assert f'{"ABCDEFGHIJKLMNOPQRST"[k-1]}. color {k-1}' in prompt
    assert align_logits(list(range(k)), order[::-1]) == list(range(k))[::-1]
    with pytest.raises(ValueError):
        prompt_and_order(question(k), 'canonical')


def test_training_identity(tmp_path):
    image = tmp_path / 'sample.jpg'
    image.write_bytes(b'test content: preflight verifies bytes, not decoding')
    q = question()
    req = tmp_path / 'requests.jsonl'
    refs = tmp_path / 'references.jsonl'
    req.write_text(json.dumps(dict(image=image.name, image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), questions=[q]))+'\n')
    label = dict(question_id='q', candidates=q['candidates'], correct_id='2')
    refs.write_text(json.dumps(label)+'\n')
    cases, _ = training_cases(req, refs, tmp_path)
    assert len(cases) == 1 and cases[0]['gold'] == 2
    label['candidates'] = list(reversed(label['candidates']))
    refs.write_text(json.dumps(label)+'\n')
    with pytest.raises(ValueError, match='identity/order'):
        training_cases(req, refs, tmp_path)
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='content changed'):
        training_cases(req, refs, tmp_path)


def test_config_ranges(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'configs/set.json'
    assert read_config(source)['arm'] == 'set'
    config = json.loads(source.read_text())
    config['learning_rate'] = float('nan')
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='finite'):
        read_config(path)
