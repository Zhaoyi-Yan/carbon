"""Stress-test the three frozen second-half policies on four preselected losing seats."""
import os
import sys
keys = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS')
if __name__ == '__main__' and (any(os.environ.get(k)!='1' for k in keys) or os.environ.get('PYTHONHASHSEED')!='0'):
    os.environ.update(dict.fromkeys(keys,'1'),PYTHONHASHSEED='0')
    os.execv(sys.executable,[sys.executable,*sys.argv])
import hashlib
import json
from pathlib import Path
from phase_switch import ARMS, HASHES, HERE, OUT, ROOT, checkpoint, configure_engine, output, prepare, run_arm, single_thread


def scheduled_rows(spec):
    original=json.loads((OUT/'phase-switch.json').read_text())['schedule']
    used={r[3] for r in original}
    pool=[]
    for name in spec['archived_sources']:
        for line in (ROOT/'analysis'/name).open():
            r=json.loads(line)
            cash=r['actions_and_post_cash'][149][1]
            for side in (0,1):
                if cash[side]>=cash[1-side] or r['layout_sha256'] in used:continue
                h=hashlib.sha256(('v7-phase-150-behind/'+r['layout_sha256']).encode()).hexdigest()
                pool.append(((h,side,name,r['players'][side]['name'],r['seed'],r['layout_sha256']),side,name,r))
    out=[]
    for side,c in zip((0,1,0,1),spec['cases']):
        _,_,name,r=next(a for a in sorted(pool,key=lambda a:a[0]) if a[1]==side and a[3]['layout_sha256'] not in used)
        cash=r['actions_and_post_cash'][149][1]
        assert [side,name,r['seed'],r['layout_sha256'],round(cash[side]-cash[1-side],2)]==[c['side'],c['source'],c['seed'],c['layout'],c['margin']]
        assert [p['name'] for p in r['players']]==c['historical_names']
        used.add(c['layout']);out.append((c,r))
    assert len(out)==4
    return out


def main():
    rt=single_thread();configure_engine();os.environ['CARBON_DEBUG']='1'
    schedule=OUT/'phase-deficit-schedule.json';spec=json.loads(schedule.read_text())
    cases=scheduled_rows(spec)
    modules=[checkpoint(n, 'deficit_'+n) for n in ARMS]+[checkpoint('estuary','deficit_opponent')]
    import torch
    from threadpoolctl import threadpool_info
    assert torch.get_num_threads()==torch.get_num_interop_threads()==1
    rt['loaded_pools']=[{k:p[k] for k in ('internal_api','prefix','num_threads')} for p in threadpool_info()]
    assert all(p['num_threads']==1 for p in rt['loaded_pools'])
    d=dict(runtime=rt,sha256=HASHES,schedule_sha256=hashlib.sha256(schedule.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_runner_sha256=hashlib.sha256((HERE/'phase_switch.py').read_bytes()).hexdigest(),
        budget=dict(search_each_turn=True,width=4,depth=2,opponent_replies=1,tail=3,wall_clock=False),
        opponent='estuary',cases=[])
    actions={}
    def save():
        output(OUT/'phase-deficit.json', d)
        output(OUT/'phase-deficit-actions.json.gz',actions,True)
    save()
    for c,r in cases:
        side=c['side'];salt=int(hashlib.sha256(('phase-common/'+c['layout']).encode()).hexdigest()[:8],16)%100000
        cfg,visible,snap,mid,mem=prepare(r,side,modules,salt)
        cash=[p[0] for p in snap['seats'][0]['observation']['players']]
        case=dict(c,split='deficit',initial=cash,initial_margin=round(cash[side]-cash[1-side],2),salt=salt,
                  initial_board_sha256=mid,initial_id_counters=snap['id_counters'],historical_scores=r['scores'],arms={})
        assert case['initial_margin']==c['margin']
        d['cases'].append(case);key=f"{r['seed']}/{side}";actions[key]={}
        for i,name in enumerate(ARMS):
            a,trace=run_arm(modules[i],modules[-1],side,(mem[i],mem[-1]),snap,cfg,visible,salt)
            case['arms'][name]=a;actions[key][name]=trace;save()
            print(key,name,a['score_margin'],a['completed_searches_by_seat'],flush=True)


if __name__=='__main__':main()
