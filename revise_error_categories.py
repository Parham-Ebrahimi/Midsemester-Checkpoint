#!/usr/bin/env python3
"""Re-label the recorded eight final-checkpoint detector false positives using
question-topic evidence from saved Qwen answer references; rerun course checker.

Usage:
 python revise_error_categories.py --files final_checkpoint_files \
   --qwen Midsemester-Checkpoint/qwen_outputs.csv --checker check_final_checkpoint.py

Categories are descriptive. They are NOT assertions about model error causes.
"""
from pathlib import Path
import argparse, subprocess, sys
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--files',type=Path,required=True)
    p.add_argument('--qwen',type=Path,required=True)
    p.add_argument('--checker',type=Path,default=Path('check_final_checkpoint.py'))
    args=p.parse_args()
    errors=pd.read_csv(args.files/'errors.csv')
    qwen=pd.read_csv(args.qwen)
    clean=qwen.loc[qwen['condition'].eq('clean')].drop_duplicates('row_id').set_index('row_id')
    groups=[]
    for rid in errors['row_id']:
        orig=rid.split(':')[1]
        if orig not in clean.index:
            raise ValueError(f'Qwen clean record not found for {orig}')
        ref=str(clean.loc[orig,'references']).lower()
        if orig.startswith('57060eaf') and any(w in ref for w in ('irvine','orange','south coast','rapidly','university')):
            groups.append('Irvine place and business questions')
        elif orig.startswith('56e0e69b') and any(w in ref for w in ('westinghouse','lamme','1893')):
            groups.append('Electrical engineering history questions')
        else:
            raise ValueError(f'Unrecognized error reference for {orig}; inspect manually before categorizing')
    errors['category']=groups
    errors.to_csv(args.files/'errors.csv',index=False)
    cmd=[sys.executable, str(args.checker.resolve()),
         '--split',str((args.files/'split.csv').resolve()),
         '--scores',str((args.files/'scores.csv').resolve()),
         '--errors',str((args.files/'errors.csv').resolve()),
         '--metric','recall','--reference','beginning','--direction','either']
    proc=subprocess.run(cmd,capture_output=True,text=True)
    (args.files/'checker_output.txt').write_text(proc.stdout + ('\nSTDERR:\n'+proc.stderr if proc.stderr else ''),encoding='utf-8')
    print('Error categories:',errors['category'].value_counts().to_dict())
    print('Checker exit code:',proc.returncode)
    print(proc.stdout[-1600:])
    if proc.returncode:sys.exit(proc.returncode)

if __name__=='__main__':main()
