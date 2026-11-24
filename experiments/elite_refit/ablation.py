"""Isolate the small rollout net and the primary style of the frozen supplemental refit."""
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from evaluate import (ROOT, V7, Strategy, audit, configure_engine, limit_math_threads,
                      new_game, run_game, source_for, write)
from network import INITIAL, read_blob, replace_blob
from report import summarize

ARCHIVE = ROOT/'analysis/elite-refit'
CONTROL = 'top6-plus'
ARMS = (CONTROL, 'old-rollout', 'generic-primary')


def sha(content):
    return hashlib.sha256(content).hexdigest()


def prepare(parent, out):
    base = source_for('yan-v7', ROOT)[2]
    refit = parent.read_bytes()
    previous = json.loads((ARCHIVE/'protocol.json').read_text())['sha256']
    assert sha(base) == V7 and sha(refit) == previous[CONTROL]
    text = refit.decode()
    original = read_blob(base.decode(), '_V6R_BYTES')
    lookup = [*range(23), 21, 21, 22, 22, *INITIAL]
    rollback = replace_blob(text, '_V6R_BYTES', {**original, 'emb':original['emb'][lookup]})
    restored = read_blob(rollback, '_V6R_BYTES')
    assert set(restored)==set(original)
    for k, v in original.items():
        assert np.array_equal(restored[k], v[lookup] if k=='emb' else v)
    anchor = '    active_expert=27+int(order_teachers[0])'
    assert text.count(anchor)==1 and '27+int(order_teachers[1])' in text
    generic = text.replace(anchor, '    active_expert=20')
    strategies = {}
    for name,source in [('yan-v7',base),(CONTROL,refit),('old-rollout',rollback.encode()),
                        ('generic-primary',generic.encode())]:
        path = out/(name+'.py')
        compile(source,str(path),'exec')
        assert len(source)<=10_000_000
        path.write_bytes(source)
        strategies[name] = Strategy(name,path,sha(source))
    return strategies


def qualifies(summary):
    return (summary['valid']==8 and min(summary['wins_by_seat'])>=3 and
            summary['average_margin']>0)


def schedule(arms, files, maps):
    jobs = []
    for i,seed in enumerate(maps):
        for arm in arms[i%len(arms):]+arms[:i%len(arms)]:
            seats = [files[arm],files['yan-v7']]
            for side in (i%2,1-i%2):
                jobs.append((arm,side,(seats[::-1] if side else seats,seed,300,True)))
    return jobs


def run(panel, jobs, pool, out, protocol):
    by_arm = {a:[] for a, _, _ in jobs}
    with gzip.open(out/(panel+'.jsonl.gz'),'xt') as log:
        for (arm,side,spec),row in zip(jobs,pool.map(run_game,[j[2] for j in jobs]),strict=True):
            log.write(json.dumps(row)+'\n')
            record = audit(row,protocol,candidate=arm)
            assert (record['seed'],record['seat'])==(spec[1],side)
            assert record['layout']==protocol['layouts'][str(spec[1])]
            by_arm[arm].append(record)
    return summarize_arms(by_arm)


def summarize_arms(by_arm):
    results = {}
    for arm,rows in by_arm.items():
        eligible = [r for r in rows if r['turns']==299 and r['status']=='completed']
        results[arm]=dict(summary=summarize(eligible),games=rows)
    return results


def main(out, parent):
    configure_engine();limit_math_threads()
    out.mkdir(parents=True,exist_ok=False)
    files=prepare(parent,out)
    seeds=[int(sha(f'elite-refit/ablation/01/{i}'.encode())[:8],16)%10**9 for i in range(8)]
    protocol=dict(screening=seeds[:4],confirmation=seeds[4:],tag='v10-yan-v7-selfplay-prior',
                  sha256={k:v.sha256 for k,v in files.items()},
                  gate='Four new maps per stage, 8 complete games, at least three wins per seat and positive mean margin in both stages. Control is descriptive only.')
    write(out/'protocol.json',protocol)
    previous=sorted((ROOT/'analysis').rglob('freshness.json'))
    previous+=[ROOT/'analysis/v8'/p for p in ('payback-layouts.json','option-layouts.json')]
    subprocess.run([sys.executable,str(ROOT/'experiments/v8/freshness.py'),str(out),'--previous',*map(str,previous)],check=True)
    protocol['layouts']=json.loads((out/'freshness.json').read_text())['maps']
    public=json.loads((ARCHIVE/'split.json').read_text())+json.loads((ARCHIVE/'forward-test.json').read_text())['maps']
    known={r['layout'] for r in public}
    for seed in seeds:
        state,_=new_game(seed,300);board=state[0].observation
        assert sha(json.dumps([board.full_carbon,[sorted(p[1].values()) for p in board.players]],separators=(',',':')).encode()) not in known
    write(out/'protocol.json',protocol)
    with ProcessPoolExecutor(3) as pool:
        results={'screening':run('screening',schedule(ARMS,files,seeds[:4]),pool,out,protocol)}
        advanced=tuple(a for a in ARMS[1:] if qualifies(results['screening'][a]['summary']))
        if advanced:
            results['confirmation']=run('confirmation',schedule(advanced,files,seeds[4:]),pool,out,protocol)
    write(out/'results.json',results)


def retry_timeouts(out):
    """Preserve original forfeits and replay only board/seat pairs without a completed game."""
    configure_engine()
    limit_math_threads()
    protocol = json.loads((out/'protocol.json').read_text())
    original = json.loads((out/'results.json').read_text())
    files = {}
    for name, expected in protocol['sha256'].items():
        path = out/(name+'.py')
        assert sha(path.read_bytes())==expected
        files[name] = Strategy(name, path, expected)
    jobs = []
    for arm, group in original['screening'].items():
        for game in group['games']:
            if game['status']!='forfeit':
                continue
            faults = [p['failure'] for p in game['players'] if p['failure']]
            if not faults or any(f['kind']!='timeout' for f in faults):
                continue
            seats = [files[arm], files['yan-v7']]
            side = game['seat']
            jobs.append((arm, side, (seats[::-1] if side else seats, game['seed'], 300, True)))
    with ProcessPoolExecutor(2) as pool:
        recovered = run('retry',jobs,pool,out,protocol)
        replacement = {(a,r['seed'],r['seat']):r for a,g in recovered.items() for r in g['games']}
        merged = {a:[replacement.get((a,r['seed'],r['seat']),r) for r in g['games']]
                  for a,g in original['screening'].items()}
        results = {'screening':summarize_arms(merged)}
        write(out/'results-after-retry.json',results)
        # The original refit remains an eligible submission candidate in a new
        # confirmation panel if its fresh screening result now passes the same gate.
        advanced = tuple(a for a in ARMS if qualifies(results['screening'][a]['summary'])
                         and a not in original.get('confirmation',{}))
        confirmed = original.get('confirmation',{})
        if advanced:
            confirmed.update(run('confirmation-after-retry',schedule(advanced,files,protocol['confirmation']),
                                 pool,out,protocol))
        if confirmed:
            results['confirmation'] = confirmed
    write(out/'results-after-retry.json',results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('parent',type=Path,nargs='?')
    parser.add_argument('--retry-timeouts',action='store_true')
    args=parser.parse_args()
    if args.retry_timeouts:
        retry_timeouts(args.out.resolve())
    elif args.parent is None:
        parser.error('provide the frozen top6-plus source file')
    else:
        main(args.out.resolve(),args.parent.resolve())
