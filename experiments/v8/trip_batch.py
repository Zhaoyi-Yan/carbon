"""Run two predeclared map panels plus historical v7 mirror controls, then audit once."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from batch import ROOT, Strategy, candidate_scores, configure_engine, limit_math_threads, run_game, source_for
from report import audit, summarize
from trip_value import build

HERE=Path(__file__).resolve().parent
RESULTS=ROOT/'analysis/v8'


def fresh_seeds(panel):
    return [int.from_bytes(hashlib.sha256(f'yan-qq-future-trips/{panel}/{i}/20261123'.encode()).digest()[:4],'big')%1_000_000_000 for i in range(4)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    limit_math_threads()
    configure_engine()
    os.environ['CARBON_DEBUG']='1'
    _,digest,raw=source_for('yan-v7',ROOT)
    base=out/'yan-v7.py'
    base.write_bytes(raw)
    candidate=out/'yan-v8-cargo-trips.py'
    priors=json.loads((RESULTS/'trip-prior.json').read_text())['prior']
    candidate.write_text(build(raw.decode(),priors))
    data=candidate.read_bytes()
    compile(data,str(candidate),'exec')
    assert len(data)<10_000_000
    challenger=Strategy('cargo-trips',candidate,hashlib.sha256(data).hexdigest())
    frozen=Strategy('yan-v7',base,digest)
    control=Strategy('v7-control',base,digest)
    protocol=dict(baseline_tag='v10-yan-v7-selfplay-prior',sha256={p.name:p.sha256 for p in (challenger,frozen,control)},
                  screening=fresh_seeds('first'),confirmation=fresh_seeds('second'),
                  controls=json.loads((RESULTS/'option-protocol.json').read_text())['screening'],
                  rule='Both panels always run. In each require 3/4 wins per seat, positive mean, complete normal search-enabled games. Controls are not counted.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    subprocess.run([sys.executable,str(HERE/'freshness.py'),str(out),'--previous',str(RESULTS/'payback-layouts.json'),str(RESULTS/'option-layouts.json')],check=True)
    fresh=json.loads((out/'freshness.json').read_text())
    tasks=[]
    for i,old in enumerate(protocol['controls']):
        for key in ('screening','confirmation'):
            seed=protocol[key][i]
            for s in (i%2,1-i%2):
                pair=[challenger,frozen] if s==0 else [frozen,challenger]
                tasks.append((key,(pair,seed,300,True)))
        tasks.append(('controls',([control,frozen],old,300,True)))
    rows={key:[] for key in ('screening','confirmation','controls')}
    with gzip.open(out/'games.jsonl.gz','xt',encoding='utf8') as log, ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_game,task):panel for panel,task in tasks}
        for future in as_completed(futures):
            panel=futures[future]
            row=future.result()
            row['panel']=panel
            log.write(json.dumps(row,separators=(',',':'))+'\n')
            log.flush()
            rows[panel].append(row)
    report={'protocol':protocol,'freshness':fresh,'panels':{}}
    for panel,raw_rows in rows.items():
        verified=[audit(r,protocol) for r in raw_rows]
        if panel!='controls':
            assert {r['layout'] for r in verified}=={fresh['maps'][str(n)] for n in protocol[panel]}
            assert sorted((r['seed'],r['seat']) for r in verified)==sorted((n,s) for n in protocol[panel] for s in (0,1))
        else:
            assert sorted(r['seed'] for r in verified)==sorted(protocol[panel])
        wins=candidate_scores(raw_rows) if panel!='controls' else None
        report['panels'][panel]=dict(games=sorted(verified,key=lambda r:(r['seed'],r['seat'])),summary=summarize(verified),passed=wins)
    report['passed']=all(report['panels'][p]['passed'] for p in ('screening','confirmation'))
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    for panel,v in report['panels'].items():
        s=v['summary']
        print(panel,s['wins_by_seat'],s['average_margin'],s['all_turns']['delta'],s['margins'])
    print('Passed:',report['passed'])


if __name__=='__main__':
    main()
