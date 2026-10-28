#!/usr/bin/env python3
"""One command to train the baseline, export CSV files, and run the course checker."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='outputs')
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    base=Path(__file__).resolve().parent
    subprocess.run([sys.executable,str(base/'project16_checkpoint.py'),'--out',str(out)],check=True)
    audit=json.loads((out/'audit.json').read_text())
    cmd=[sys.executable,str(base/'check_checkpoint.py'), '--split',str(out/'split.csv'),
         '--scores',str(out/'scores.csv'),'--threshold',str(audit['threshold'])]
    checker=subprocess.run(cmd,check=True,capture_output=True,text=True)
    (out/'checker_output.txt').write_text(checker.stdout,encoding='utf-8')
    print(checker.stdout)
    print('Baseline and checker finished. Commit the code, split.csv and scores.csv to your repository.')

if __name__=='__main__':main()
