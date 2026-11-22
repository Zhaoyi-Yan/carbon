"""Exercise the embedded Master proposal through a normal search, timed on legal observations."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import master
from master_data import ROOT
from localhead_data import load
from benchmark import validate_actions

def main(a):
    src=master.build(a.parent.read_text())
    assert a.candidate.read_text()==src
    rows=json.loads((a.cache/'index.json').read_text())
    r=next(r for r in rows if r['fold']=='test' and r['teacher']=='Master7sE')
    with gzip.open(r['file']) as f:g=json.load(f)
    m=load(a.candidate);s=r['side'];real=m._search;prep=m._ms_prepare;original=m._ms_recruits
    times=[];active=[];changes=[];true_calls=[]
    def prepare(state,obs,x,heads,mine,h):
        start=time.perf_counter();prep(state,obs,x,heads,mine,h)
        times.append((time.perf_counter()-start)*1000);active.append(state['ms_hiring'] is not None)
    def probe(obs,base,logits):
        alt=original(obs,base,logits)
        if alt is not None:
            own=obs['players'][s];validate_actions(alt,own)
            assert all(alt.get(u)==base.get(u) for u in own[2]);changes.append(obs['step'])
        true_calls.append(obs['step'])
        return alt
    m._ms_prepare=prepare;m._ms_recruits=probe
    def dry(o,c,x,act,*args):
        probe(o,act,m._ST['v5_history']['ms_hiring']);return act
    for t,frame in enumerate(g['steps'][:40]):
        o={k:v for k,v in frame[0]['observation'].items() if k!='full_carbon'}
        o.update(frame[s]['observation']);o.pop('full_carbon',None);o['player']=s
        m._search=dry
        if t==20:
            m._search=real;m._S_K=4;m._S_D2=2;m._S_COST[:]=[.0004,.0003,.0004,.0003]
            act=m.agent(o,g['configuration'])
        else:act=m._act(o,g['configuration'],None)
        validate_actions(act,o['players'][s])
        m._ST['tr'].set_actions(o['players'],g['steps'][t+1][s]['action'] or {})
    assert m._S_STATS['searched']>0 and sum(active)>0
    report=dict(candidate_sha=hashlib.sha256(src.encode()).hexdigest(),parent_sha=hashlib.sha256(a.parent.read_bytes()).hexdigest(),
                public_match=r['match'],real_search_count=int(m._S_STATS['searched']),master_active=sum(active),total_frames=len(active),
                distinct_base_alternatives=len(changes),root_generator_calls=len(true_calls),milliseconds_active_p50_p95=np.percentile(np.array(times)[active],[50,95]).round(3).tolist(),
                single_file_bytes=len(src.encode()))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('parent',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path)
    main(p.parse_args())
