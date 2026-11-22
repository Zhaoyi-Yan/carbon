"""Eight seat-paired full games against the fixed v6, using the archived Lifeline budget."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from development_duel import ROOT,game,single_thread,configure_engine,model

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(source,output,limit=None):
    single_thread();configure_engine()
    base=ROOT/'experiments/v7/checkpoints/lifeline.py';opponent=ROOT/'yan-v6.py';prior=json.loads((ROOT/'analysis/v7/lifeline-development.json').read_text())
    assert prior['sha256']==sha(base) and prior['opponent']==sha(opponent)
    tag=sha(source)
    report=dict(candidate=tag,reference=sha(base),opponent=sha(opponent),design=prior['design'],games=[])
    if output.exists():
        saved=json.loads(output.read_text());assert all(saved[k]==report[k] for k in ('candidate','reference','opponent'))
        report=saved
    mine=model(source,'master_full_lifeline_paired',True);rival=model(opponent,'master_full_estuary_paired',True)
    done={(r['seed'],r['seat']) for r in report['games']}
    for row in prior['games']:
        if (row['seed'],row['seat']) in done:continue
        out=game(row['seed'],row['seat'],mine,rival);assert out['layout']==row['layout']
        out['lifeline_margin']=row['margin'];out['paired_delta']=round(out['margin']-row['margin'],2);report['games'].append(out)
        print('paired',out['seed'],out['seat'],out['margin'],out['paired_delta'],flush=True)
        write(output,report)
        if limit and len(report['games'])>=limit:break
    if len(report['games'])==8:
        r=report['games'];report['summary']=dict(wins=sum(g['margin']>0 for g in r),reference_wins=sum(g['lifeline_margin']>0 for g in r),
                  mean_margin=round(statistics.mean(g['margin'] for g in r),2),mean_paired_delta=round(statistics.mean(g['paired_delta'] for g in r),2),
                  new_wins=sum(g['margin']>0>=g['lifeline_margin'] for g in r),lost_wins=sum(g['lifeline_margin']>0>=g['margin'] for g in r),
                  by_seat=[sum(g['seat']==s and g['margin']>0 for g in r) for s in (0,1)])
        write(output,report);print(report['summary'],flush=True)
def write(path,report):
    tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--limit',type=int);a=p.parse_args();main(a.source,a.output,a.limit)
