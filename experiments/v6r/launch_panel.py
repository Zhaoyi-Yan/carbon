"""Freeze one standalone strategy and play both seats on every map assigned to its round."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('round',type=int);p.add_argument('output',type=Path)
    p.add_argument('--jobs',type=int,default=6)
    a=p.parse_args();panels=json.loads((HERE/'panels.json').read_text())
    if not 1<=a.round<=len(panels['fresh_rounds']):p.error('round is outside preassigned panels')
    if a.jobs<1 or a.output.exists():p.error('supply a positive job count and a new output file')
    seeds=panels['fresh_rounds'][a.round-1];name=a.candidate.stem
    frozen=dict(candidate=name,sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
                opponent_sha256=panels['baseline_sha256'],seeds=seeds,criterion=panels['accept'])
    path=HERE/f'qualification-{a.round}.json'
    if path.exists() and json.loads(path.read_text())!=frozen:p.error('a different strategy was already registered on these maps')
    if not path.exists():
        with path.open('x') as f:f.write(json.dumps(frozen,indent=2)+'\n')
    command=[sys.executable,str(ROOT/'scripts/selfplay.py'),'--candidate',str(a.candidate),'--baseline','yan-v5','--seeds',
             *map(str,seeds),'--jobs',str(a.jobs),'--record-actions','--output',str(a.output)]
    return subprocess.run(command).returncode

if __name__=='__main__':raise SystemExit(main())
