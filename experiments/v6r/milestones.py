"""Read opening milestones from recorded actions and the actual referee."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark import configure_engine,limit_math_threads,new_game


def opening(record):
    from zerosum_env.envs.carbon import carbon
    state,env=new_game(record['seed'],record['episode_steps'])
    marks=[{'first_tree':None,'first_collect_death':None,'first_plant_death':None,'first_five':None,'snapshots':[]} for _ in range(2)]
    for t,(actions,cash) in enumerate(record['actions_and_post_cash'][:75],1):
        pre=state[0].observation.players
        for s,a in zip(state,actions):s.action=a
        carbon.compute_next_board(state,env);pl=state[0].observation.players
        for i,p in enumerate(pl):
            x=marks[i];w=p[2];tree=p[3]
            if x['first_tree'] is None and tree:x['first_tree']=t
            if x['first_five'] is None and len(w)>=5:x['first_five']=t
            for uid in pre[i][2].keys()-w.keys():
                name='first_collect_death' if pre[i][2][uid][2]=='COLLECTOR' else 'first_plant_death'
                if x[name] is None:x[name]=t
            if t in (5,10,15,25,40,60,75):
                x['snapshots'].append([t,p[0],sum(k=='COLLECTOR' for _,_,k in w.values()),sum(k=='PLANTER' for _,_,k in w.values()),len(tree)])
            if abs(cash[i]-p[0])>1.e-6:raise ValueError('incorrect replay')
    return marks


if __name__=='__main__':
    par=argparse.ArgumentParser();par.add_argument('files',type=Path,nargs='+');par.add_argument('--output',type=Path,required=True)
    a=par.parse_args();configure_engine();limit_math_threads();rows=[]
    for f in a.files:
        for line in f.open():
            r=json.loads(line)
            if r['status']!='completed':continue
            s=int(r['players'][0]['name']=='yan-v5');m=opening(r)
            rows.append(dict(seed=r['seed'],seat=s,delta=r['scores'][s]-r['scores'][1-s],candidate=m[s],v5=m[1-s]))
    a.output.write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:
        print(r['seed'],r['seat'],round(r['delta']),[(v['first_tree'],v['first_five'],v['first_plant_death'],v['snapshots'][3][1:]) for v in (r['candidate'],r['v5'])])
