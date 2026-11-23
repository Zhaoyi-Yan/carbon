"""Batch the four isolated team policies and their combination against frozen v7."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from batch import ROOT, Strategy, candidate_scores, configure_engine, limit_math_threads, run_game, source_for
from report import audit, summarize
from teamplay import ARMS, build
from statistics import mean

RESULTS=ROOT/'analysis/v8'


def seeds(label):
    return [int(hashlib.sha256(f'yan-team-20261123/{label}/{i}'.encode()).hexdigest()[:8],16)%10**9 for i in range(4)]


def run_panel(out,label,arms,players,protocol,jobs):
    tasks=[]
    for i,seed in enumerate(protocol[label]):
        for arm in arms[i%len(arms):]+arms[:i%len(arms)]:
            for seat in (i%2,1-i%2):
                pair=[players[arm],players['yan-v7']]
                tasks.append((pair[::-1] if seat else pair,seed,300,True))
    result={arm:[] for arm in arms}
    with gzip.open(out/(label+'.jsonl.gz'),'xt') as f, ProcessPoolExecutor(jobs) as pool:
        for fut in as_completed([pool.submit(run_game,task) for task in tasks]):
            row=fut.result()
            f.write(json.dumps(row,separators=(',',':'))+'\n')
            name=next(p['name'] for p in row['players'] if p['name']!='yan-v7')
            result[name].append(row)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    out=args.out.resolve()
    out.mkdir(exist_ok=False)
    limit_math_threads()
    configure_engine()
    os.environ['CARBON_DEBUG']='1'
    _,digest,source=source_for('yan-v7',ROOT)
    players={'yan-v7':Strategy('yan-v7',ROOT/'yan-v7.py',digest)}
    for arm in ARMS:
        path=out/('yan-v8-'+arm+'.py')
        code=build(source.decode(),arm)
        compile(code,str(path),'exec')
        path.write_text(code)
        assert path.stat().st_size<10_000_000
        players[arm]=Strategy(arm,path,hashlib.sha256(path.read_bytes()).hexdigest())
    protocol=dict(screening=seeds('screen'),confirmation=seeds('confirm'),sha256={a:p.sha256 for a,p in players.items()},
                  rule='All five arms on same four mirrored maps. Every arm with 3/4 wins per seat and positive mean advances to four separate maps under the same rule.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    prior=out/'previous.json'
    prior.write_text(json.dumps(json.loads((RESULTS/'trip-results.json').read_text())['freshness']))
    subprocess.run([sys.executable,str(Path(__file__).with_name('freshness.py')),str(out),'--previous',str(prior),str(RESULTS/'option-layouts.json'),str(RESULTS/'payback-layouts.json')],check=True)
    stages={'screening':run_panel(out,'screening',ARMS,players,protocol,args.jobs)}
    passed=tuple(a for a in ARMS if candidate_scores(stages['screening'][a]))
    if passed:
        stages['confirmation']=run_panel(out,'confirmation',passed,players,protocol,args.jobs)
    layouts=json.loads((out/'freshness.json').read_text())['maps']
    detail={}
    for label,arms in stages.items():
        detail[label]={}
        for arm,raw in arms.items():
            rows=[audit(r,protocol,(75,125)) for r in raw]
            assert sorted((r['seed'],r['seat']) for r in rows)==sorted((n,s) for n in protocol[label] for s in (0,1))
            assert all(r['layout']==layouts[str(r['seed'])] for r in rows)
            summary=summarize(rows)
            valid=[r for r in rows if 'opening' in r and 125 in r['opening']]
            for turn in (75,125):
                pairs=[r['opening'][turn] for r in valid]
                if pairs:
                    summary[f'cash_gap_{turn}']=round(mean(a['cash']-b['cash'] for a,b in pairs),2)
                    summary[f'bank_gap_{turn}']=round(mean(sum(a['by_port'].values())-sum(b['by_port'].values()) for a,b in pairs),2)
            detail[label][arm]=dict(summary=summary,passed=candidate_scores(raw),games=rows)
            print(label,arm,summary['wins_by_seat'],summary['average_margin'],summary.get('cash_gap_75'),summary.get('bank_gap_125'),flush=True)
    (out/'results.json').write_text(json.dumps(detail,indent=2)+'\n')


if __name__=='__main__':
    main()
