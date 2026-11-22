"""Equal-budget two-seat tournament for the dual-domain network with per-turn gate instrumentation."""
import argparse
from collections import Counter
import json
import statistics
from pathlib import Path
from master_full_development import sha,write,ROOT,game,single_thread,configure_engine,model

def main(candidate,out):
    single_thread();configure_engine()
    base=ROOT/'experiments/v7/checkpoints/lifeline.py';op=ROOT/'yan-v6.py';ref=json.loads((ROOT/'analysis/v7/lifeline-development.json').read_text())
    assert ref['sha256']==sha(base) and ref['opponent']==sha(op)
    result=dict(candidate=sha(candidate),reference=sha(base),opponent=sha(op),design=ref['design'],games=[])
    if out.exists():
        j=json.loads(out.read_text());assert all(j[k]==result[k] for k in ('candidate','reference','opponent'));result=j
    done={(r['seed'],r['seat']) for r in result['games']}
    me=model(candidate,'multi_master_candidate',True);other=model(op,'multi_master_v6',True);original=me._v5_root;telemetry={}
    def probe(obs,*args):
        policy=original(obs,*args);state=me._ST['v5_history'];telemetry[obs['step']]=int(state['mm_id']);return policy
    me._v5_root=probe
    for r in ref['games']:
        if (r['seed'],r['seat']) in done:continue
        telemetry.clear();z=game(r['seed'],r['seat'],me,other);assert r['layout']==z['layout']
        z['lifeline_margin']=r['margin'];z['paired_delta']=round(z['margin']-r['margin'],2)
        ticks=[k for k,v in telemetry.items() if v>=27];z['gate_steps']=ticks
        z['style_counts']=dict(Counter(telemetry.values()));z['root_turns']=len(telemetry);assert len(telemetry)==z['end']
        result['games'].append(z);write(out,result);print('paired',z['seed'],z['seat'],z['margin'],z['paired_delta'],len(ticks),flush=True)
    if len(result['games'])==8:
        a=result['games'];result['summary']=dict(wins=sum(r['margin']>0 for r in a),reference_wins=sum(r['lifeline_margin']>0 for r in a),
            average_margin=round(statistics.mean(r['margin'] for r in a),2),paired_mean=round(statistics.mean(r['paired_delta'] for r in a),2),
            newly_won=sum(r['margin']>0>=r['lifeline_margin'] for r in a),lost=sum(r['lifeline_margin']>0>=r['margin'] for r in a),
            gate_turns=sum(len(r['gate_steps']) for r in a),observed_turns=sum(r['root_turns'] for r in a))
        write(out,result);print(result['summary'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.candidate,a.out)
