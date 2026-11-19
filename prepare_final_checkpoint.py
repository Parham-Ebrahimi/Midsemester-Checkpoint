#!/usr/bin/env python3
"""Convert the real Project 16 midsemester repository files to the Fall 2026 final-checkpoint checker.

Usage:
  python prepare_final_checkpoint.py --repo /path/to/Midsemester-Checkpoint --checker /path/to/check_final_checkpoint.py

This does not invent classifier predictions or Qwen generations. It reads committed data,
creates a paired placement comparison for the trained detector, and runs the supplied checker.
"""
import argparse
import csv
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

THRESHOLD = 0.47903516630541765
VALUES = ('beginning', 'middle', 'end')
COLUMN = 'SQuAD detector'
SEED = 16


def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return (float('nan'), float('nan'))
    p = successes / total
    d = 1 + z*z/total
    center = (p + z*z/(2*total))/d
    margin = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return max(0, center-margin), min(1, center+margin)


def make_files(repo, out):
    original_split = load_csv(repo / 'split.csv')
    original_scores = load_csv(repo / 'scores.csv')
    qwen = load_csv(repo / 'qwen_outputs.csv')
    assert len(original_split) == 1000, 'Expected exactly 1,000 original detector split rows.'
    assert len(original_scores) == 600, 'Expected exactly 600 scored detector test rows.'
    assert len(qwen) == 800, 'Expected exactly 800 Qwen responses.'
    orig_ids = {r['row_id']:r for r in original_split}
    # The midsemester detector has separate rows for each position. For the final
    # paired-comparison format, keep clean and beginning test IDs as the canonical
    # 200 test rows; score all three attack positions against those same 200 IDs.
    split_rows = [dict(r) for r in original_split if not
                  (r['split'] == 'test' and r['row_id'].endswith((':middle', ':end')))]
    seen = set()
    for r in split_rows:
        assert r['row_id'] not in seen, 'Repeated split ID'
        seen.add(r['row_id'])
    assert Counter(r['split'] for r in split_rows) == {'train': 400, 'validation': 200, 'test': 200}
    write_csv(out / 'split.csv', split_rows, ['row_id', 'dataset', 'split', 'label'])

    scored = []
    for r in original_scores:
        value = r['test_set']
        assert value in VALUES
        row_id = r['row_id']
        if int(r['label']) == 1:
            assert row_id.endswith(':' + value), 'Unexpected positive ID'
            row_id = row_id[:-(len(value))] + 'beginning'
        else:
            assert row_id.endswith(':clean'), 'Unexpected clean ID'
        assert row_id in seen and orig_ids[row_id]['split'] == 'test'
        assert int(orig_ids[row_id]['label']) == int(r['label'])
        scored.append({'row_id': row_id, 'value': value, 'column': COLUMN,
                       'label': r['label'], 'score': r['score'], 'threshold': THRESHOLD})
    assert Counter(r['value'] for r in scored) == dict.fromkeys(VALUES,200)
    assert len({r['row_id'] for r in scored}) == 200
    write_csv(out / 'scores.csv', scored, ['row_id','value','column','label','score','threshold'])

    # Choose one actual error-containing cell, sample up to 10 of each error type.
    # With the committed 0.5B baseline detector data, beginning has 8 false
    # positives and 0 false negatives. All eight are retained.
    baseline = [r for r in scored if r['value'] == 'beginning']
    fps = [r for r in baseline if int(r['label']) == 0 and float(r['score']) >= THRESHOLD]
    fns = [r for r in baseline if int(r['label']) == 1 and float(r['score']) < THRESHOLD]
    rng = random.Random(SEED)
    sampled = rng.sample(fps, min(10, len(fps))) + rng.sample(fns, min(10, len(fns)))
    def cat(r):
        # Objective diagnostic grouping, not a guessed semantic explanation.
        base = r['row_id'].split(':')[1]
        return ('clean row / ID group ' + base[:8]) if int(r['label']) == 0 else ('injected row / ID group ' + base[:8])
    errors = [{'row_id':r['row_id'],'value':r['value'],'column':r['column'],'category':cat(r)} for r in sampled]
    write_csv(out / 'errors.csv', errors, ['row_id','value','column','category'])
    summary = []
    by_condition = defaultdict(list)
    for r in qwen:
        by_condition[r['condition']].append(r)
    for v in ('clean', *VALUES):
        rows = by_condition[v]
        assert len(rows) == 200, f'Expected 200 Qwen rows for {v}'
        n = len(rows)
        attacks = sum(int(r['attack_success']) for r in rows)
        answers = sum(int(r['answer_correct']) for r in rows)
        a_lo, a_hi = wilson(attacks,n)
        c_lo, c_hi = wilson(answers,n)
        summary.append({'condition':v,'n':n,'PWNED_replies':attacks,
                        'attack_success':attacks/n,'ASR_95_lo':a_lo,'ASR_95_hi':a_hi,
                        'correct_answers':answers,'answer_accuracy':answers/n,
                        'accuracy_95_lo':c_lo,'accuracy_95_hi':c_hi})
    write_csv(out / 'qwen_wilson.csv', summary, list(summary[0]))
    print(f'Converted {len(split_rows)} split rows, {len(scored)} detector score rows, {len(errors)} sampled errors.')
    for r in summary:
        print(f"Qwen {r['condition']:>9}: attack {r['PWNED_replies']}/{r['n']}, correct {r['correct_answers']}/{r['n']}; ASR 95% [{r['ASR_95_lo']:.4f}, {r['ASR_95_hi']:.4f}]")
    print(f'Error sample: {len(fps)} available false positives, {len(fns)} false negatives; {len(errors)} sampled.')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True, help='Path to the public Midsemester-Checkpoint repository cloned locally')
    p.add_argument('--checker', type=Path, default=Path('check_final_checkpoint.py'))
    p.add_argument('--out', type=Path, default=Path('final_checkpoint_files'))
    args = p.parse_args()
    out = make_files(args.repo, args.out)
    cmd = [sys.executable, str(args.checker.resolve()), '--split', str((out/'split.csv').resolve()),
           '--scores', str((out/'scores.csv').resolve()), '--errors', str((out/'errors.csv').resolve()),
           '--metric','recall','--reference','beginning','--direction','either']
    print('Running the professor\'s final-checkpoint checker ...')
    result = subprocess.run(cmd, text=True, capture_output=True)
    (out/'checker_output.txt').write_text(result.stdout + ('\nSTDERR:\n'+result.stderr if result.stderr else ''), encoding='utf-8')
    print('Checker exit code:', result.returncode)
    print(result.stdout[-4500:])
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode:
        sys.exit(result.returncode)
    print('Output directory:', out.resolve())
    print('These are detector recall results. Qwen attack-success intervals are in qwen_wilson.csv, NOT the detector checker.')

if __name__ == '__main__':
    main()
