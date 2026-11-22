"""Freeze the exact file and compare it to verified v6 on each reserved map and both seats."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import os

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/v6r'))
from report import summarize
from freshness import check


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('round',type=int);p.add_argument('--confirm',action='store_true')
    p.add_argument('--jobs',type=int,default=8);a=p.parse_args()
    design=json.loads((HERE/'panels.json').read_text());batch='confirmation' if a.confirm else 'qualifying'
    source=a.candidate.resolve();sha=hashlib.sha256(source.read_bytes()).hexdigest()
    prefix=ROOT/'analysis/v7'/f'{a.round:02d}-{batch}'
    q=ROOT/'analysis/v7'/f'{a.round:02d}-qualifying-summary.json'
    if a.confirm and (not q.exists() or not (prev:=json.loads(q.read_text()))['passed'] or prev['sha256']!=sha):
        p.error('confirmation requires the same file as a successful qualifying batch')
    frozen=dict(candidate=source.stem,sha256=sha,opponent_sha256=design['frozen_v6'],seeds=design[batch][a.round-1],criterion=design['rule'])
    manifest=prefix.with_suffix('.json');games=prefix.with_suffix('.jsonl')
    if manifest.exists():
        if json.loads(manifest.read_text())!=frozen:raise ValueError('candidate already frozen')
    else:manifest.write_text(json.dumps(frozen,indent=2)+'\n')
    if games.exists():raise ValueError('output already exists')
    cmd=[sys.executable,str(ROOT/'scripts/selfplay.py'),'--candidate',str(source),'--baseline','yan-v6','--seeds',*map(str,frozen['seeds']),
         '--jobs',str(a.jobs),'--record-actions','--output',str(games)]
    proc=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CARBON_DEBUG='1'))
    rows=[json.loads(l) for l in games.read_text().splitlines()]
    result=summarize(frozen,rows)
    if not result['complete']:raise ValueError('incomplete pairings')
    result['freshness']=check(frozen,rows)
    training=set(json.loads((prefix.parent/'public-layouts.json').read_text()))
    overlaps={r['map_sha256'] for r in rows}&training
    if overlaps:raise ValueError('an evaluation layout occurs in deposit training, holdout, or the source replay')
    result['freshness'].update(receipt_and_target_layouts=len(training),receipt_and_target_overlap=len(overlaps))
    previous=set()
    for path in prefix.parent.glob('*.jsonl'):
        if path==games:continue
        for line in path.open():
            g=json.loads(line)
            if 'layout_sha256' in g:previous.add(g['layout_sha256'])
    layouts={r['layout_sha256'] for r in rows}
    if layouts & previous:raise ValueError('overlap with previous v7 panel')
    if a.confirm:result['passed']=not result['invalid'] and result['wins']>=5 and min(result['wins_by_seat'])>=2 and result['average']>0
    prefix.with_name(prefix.name+'-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['wins'],result['losses'],result['wins_by_seat'],result['average'],result['passed'],flush=True)
    return proc.returncode


if __name__=='__main__':raise SystemExit(main())
