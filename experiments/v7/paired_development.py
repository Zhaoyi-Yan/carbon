"""Add one arm to an existing equal-budget development tournament with the same seeds and seats."""
import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from development_duel import ROOT,game,single_thread,configure_engine,model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('reference',type=Path);p.add_argument('reference_results',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();single_thread();configure_engine();prior=json.loads(a.reference_results.read_text())
    ref=hashlib.sha256(a.reference.read_bytes()).hexdigest();frozen=hashlib.sha256((ROOT/'yan-v6.py').read_bytes()).hexdigest()
    idx=prior['sha256'].index(ref);assert prior['sha256'][-1]==frozen
    m=model(a.candidate,'paired_candidate',True);e=model(ROOT/'yan-v6.py','paired_estuary',True)
    out=dict(sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),reference=ref,opponent=frozen,
             design='Fixed four candidate actions, depth two, one opponent reply plus three tail steps on four previously used development maps. Each seat paired with its identical reference arm.',games=[])
    for base in [r for r in prior['games'] if r['arm']==idx]:
        r=game(base['seed'],base['seat'],m,e);assert r['layout']==base['layout']
        r['reference_margin']=base['margin'];r['delta']=round(r['margin']-base['margin'],2);out['games'].append(r)
        print(r['seed'],r['seat'],r['margin'],r['delta'],flush=True)
        a.output.write_text(json.dumps(out,indent=2)+'\n')
    out['summary']=dict(wins=sum(r['margin']>0 for r in out['games']),mean=round(statistics.mean(r['margin'] for r in out['games']),2),
                        paired_mean=round(statistics.mean(r['delta'] for r in out['games']),2),
                        gains=sum(r['margin']>0>=r['reference_margin'] for r in out['games']),lost=sum(r['reference_margin']>0>=r['margin'] for r in out['games']))
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(out['summary'],flush=True)


if __name__=='__main__':main()
