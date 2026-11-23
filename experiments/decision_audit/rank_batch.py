"""Freeze a ranking experiment, fit on archived v6 games and test on independent maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import os

from run import ROOT, audit_match, configure_engine, V7_SHA256
from rank import trace_source, build, evaluate, fit

sys.path.insert(0,str(ROOT/'experiments/v8'))
from arena import Strategy, run_game
from benchmark import limit_math_threads
from scripts.selfplay import source_for
from report import audit


def write(path, data):
    path.write_text(json.dumps(data,indent=2)+'\n')


def source_matches():
    archive=ROOT/'analysis'
    for split,sources,own,opponent in (
        ('fit',[archive/'v7/13-qualifying.jsonl',archive/'v7/13-confirmation.jsonl'],
         'selfplay-prior','yan-v6'),
        ('holdout',[archive/'v8/trip-games.jsonl.gz'],'v7-control','yan-v7'),
    ):
        by_map={}
        for path in sources:
            opener=gzip.open if path.suffix=='.gz' else open
            with opener(path,'rt') as records:
                for row in map(json.loads,records):
                    if row['status']!='completed' or row['turns']!=299:
                        continue
                    for seat,p in enumerate(row['players']):
                        if p['name']==own and p['sha256']==V7_SHA256:
                            by_map.setdefault((row['seed'],row['layout_sha256']),{})[seat]=row
        if len(by_map)!=(12 if split=='fit' else 4):
            raise ValueError(f'Unexpected archived map count for {split}')
        _,digest,_=source_for(opponent,ROOT)
        for i,((seed,layout),rows) in enumerate(sorted(by_map.items())):
            seat = i%2 if i%2 in rows else next(iter(rows))
            row=rows[seat]
            assert row['players'][1-seat]['sha256']==digest
            yield split,row,seat,ROOT/(opponent+'.py')


def stats(games):
    games=[g for g in games if 'margin' in g]
    margins=[g['margin'] for g in games]
    if not margins:
        return {'valid':0}
    left=[g['players'][g['seat']] for g in games]
    right=[g['players'][1-g['seat']] for g in games]
    return dict(valid=len(margins),wins=sum(x>0 for x in margins),draws=sum(x==0 for x in margins),
                seat_wins=[sum(g['seat']==s and g['margin']>0 for g in games) for s in (0,1)],
                average=round(mean(margins),2),median=round(median(margins),2),
                worst=min(margins),time=[dict(avg_ms=round(1000*mean(p['total_act_seconds']/p['calls'] for p in group),1),
                max_seconds=round(max(p['max_act_seconds'] for p in group),4),
                slow_steps=sum(p['slow_steps'] for p in group)) for group in (left,right)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--resume-labels',action='store_true',help='reuse complete, validated label archive')
    args=p.parse_args()
    out=args.out.resolve()
    if args.resume_labels:
        if not (out/'labels.jsonl').is_file() or (out/'games.jsonl.gz').exists():
            raise ValueError('Expected labels with no arena games')
    else:
        out.mkdir(parents=True,exist_ok=False)
    limit_math_threads()
    configure_engine()
    os.environ['CARBON_DEBUG']='1'
    roles=('yan-v7','yan-v4')
    files={}
    for role in roles:
        _,digest,content=source_for(role,ROOT)
        filename=out/(role+'.py')
        filename.write_bytes(content)
        files[role]=Strategy(role,filename,digest)
    base=files['yan-v7'].path.read_text()
    trace=out/'traced.py'
    source=trace_source(base)
    compile(source,str(trace),'exec')
    trace.write_text(source)
    archived=list(source_matches())
    turns=(215,260)
    seeds=[int(hashlib.sha256(f'carbon-rank-cycle-1/final/{i}'.encode()).hexdigest()[:8],16)%10**9
           for i in range(4)]
    protocol=dict(screening=seeds,confirmation=[],training_roots=turns,
                  archived=[dict(split=split, seed=row['seed'],layout=row['layout_sha256'],seat=seat,
                                 opponent=opponent.stem) for split,row,seat,opponent in archived],
                  sha256={a:f.sha256 for a,f in files.items()},
                  evaluation='Four frozen fresh paired maps per matchup. v4 withheld from all rank learning. Same seats and maps for v4 control.')
    write(out/'protocol.json',protocol)
    archive=ROOT/'analysis/v8'
    previous=out/'trip-freshness.json'
    write(previous,json.loads((archive/'trip-results.json').read_text())['freshness'])
    prev=[previous,archive/'payback-layouts.json',archive/'option-layouts.json',
          archive/'team/freshness.json',archive/'opening-and-grove/freshness.json']
    subprocess.run([sys.executable,str(ROOT/'experiments/v8/freshness.py'),str(out),'--previous',*map(str,prev)],check=True)
    labels=out/'labels.jsonl'
    if not args.resume_labels:
        with labels.open('x') as records, ProcessPoolExecutor(args.jobs) as pool:
            specs=[(row,seat,opp,trace,turns,False) for _,row,seat,opp in archived]
            for (split,_,_,_),game in zip(archived,pool.map(audit_match,specs)):
                records.write(json.dumps(dict(split=split,**game))+'\n')
    games={'fit':[],'holdout':[]}
    with labels.open() as records:
        for saved,(split,row,seat,opp) in zip(map(json.loads,records),archived,strict=True):
            assert (saved['split'],saved['seed'],saved['layout'],saved['own'],saved['opponent']) == (
                split,row['seed'],row['layout_sha256'],seat,opp.stem)
            assert tuple(r['turn'] for r in saved['roots'])==turns
            games[split].append(saved)
    if any(b['steps']!=299 for group in games.values() for g in group for r in g['roots'] for b in r['results']):
        raise ValueError('Branch did not reach terminal')
    fitted=fit(games['fit'])
    write(out/'model.json',fitted)
    write(out/'closed-loop.json',evaluate(games['holdout'],fitted['weights']))
    target=out/'yan-v8-rank.py'
    source=build(base,fitted['weights'])
    compile(source,str(target),'exec')
    target.write_text(source)
    files['rank']=Strategy('rank',target,hashlib.sha256(target.read_bytes()).hexdigest())
    protocol['sha256']['rank']=files['rank'].sha256
    write(out/'protocol.json',protocol)

    matchups=(('rank','yan-v7'),('rank','yan-v4'),('yan-v7-control','yan-v4'))
    files['yan-v7-control']=Strategy('yan-v7-control',files['yan-v7'].path,files['yan-v7'].sha256)
    protocol['sha256']['yan-v7-control']=files['yan-v7'].sha256
    write(out/'protocol.json',protocol)
    tasks=[]
    for i,seed in enumerate(seeds):
        for match in matchups[i%3:]+matchups[:i%3]:
            seats=[files[a] for a in match]
            for swap in (i%2,1-i%2):
                tasks.append(('/'.join(match),(seats[::-1] if swap else seats,seed,300,True)))
    results={f'{a}/{b}':[] for a,b in matchups}
    with gzip.open(out/'games.jsonl.gz','wt') as records, ProcessPoolExecutor(args.jobs) as pool:
        for (match,_),row in zip(tasks,pool.map(run_game,[spec for _,spec in tasks])):
            records.write(json.dumps(dict(matchup=match,**row))+'\n')
            verified=audit(row,protocol,candidate=match.split('/')[0])
            results[match].append(verified)
    expected=json.loads((out/'freshness.json').read_text())['maps']
    for gameset in results.values():
        assert sorted((g['seed'],g['seat']) for g in gameset)==sorted((s,i) for s in seeds for i in (0,1))
        assert all(g['layout']==expected[str(g['seed'])] for g in gameset)
    report={a:dict(summary=stats(games),games=games) for a,games in results.items()}
    write(out/'results.json',report)
    print(json.dumps({k:v['summary'] for k,v in report.items()}))


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='0':
        os.environ['PYTHONHASHSEED']='0'
        os.execv(sys.executable,[sys.executable,*sys.argv])
    main()
