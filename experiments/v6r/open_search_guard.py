"""Replay an observational early-root filter on the previously scored 48 decisions, without running search again."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from localhead_data import load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('measurements',type=Path)
    p.add_argument('data',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    m=load(a.candidate);report=json.loads(a.measurements.read_text());cache={};results=[]
    for r in report['samples']:
        game=r['game']
        if game not in cache:
            with gzip.open(a.data/'replays'/(game+'.json.gz')) as f:cache[game]=json.load(f)
        replay=cache[game];step=r['step'];me=r['seat'];op=1-me
        frame=replay['steps'][step];nxt=replay['steps'][step+1];obs=frame[0]['observation'];players=obs['players']
        config=replay['configuration'];own,rival=players[me],players[op]
        defended=[]
        for u in r['units']:
            if u['kind']=='base' or u['sampling_p']<.90 or u['root']==u['search']:continue
            pos,cargo,role=own[2][u['uid']]
            dest=m._ROLLOUT_LANDING[pos][u['root']]
            # A root landing is exposed if any legal opposing collector can beat or draw at that cell.
            threat=False
            for q,enemy_cargo,kind in rival[2].values():
                if kind!='COLLECTOR' or m.FEAT_DIST[q,dest]>1:continue
                theirs=enemy_cargo*(1-config.get('moveCost',0.)) if q!=dest else enemy_cargo
                ours=cargo*(1-config.get('moveCost',0.)) if pos!=dest else cargo
                threat|=role=='PLANTER' or theirs<=ours
            threat|=dest in rival[1].values()
            if not threat:defended.append(u['uid'])
        choose_root=bool(defended)
        legal=[u for u in r['units'] if u['eligible']]
        scores={k:sum(u[k]==u['teacher'] for u in legal) for k in ('root','search','played')}
        teacher=nxt[me]['action'] or {};enemy=nxt[op]['action'] or {}
        state={'players':players,'trees':obs['trees'],'step':step,'player':me,'carbon':obs['full_carbon']}
        def follow(action):
            acts=[action,enemy] if me==0 else [enemy,action]
            b=m._turn(state,config,acts)
            return np.array([[pl[0],pl[0]+sum(w[1] for w in pl[2].values()),len(pl[2]),len(pl[3])] for pl in b['players']])
        factual=follow(teacher)
        assert max(abs(factual[i,0]-nxt[0]['observation']['players'][i][0]) for i in (0,1))<.015
        assert all(factual[i,2]==len(nxt[0]['observation']['players'][i][2]) for i in (0,1))
        diff=follow(r['searched_joint'])-follow(r['root_joint'])
        cash_cargo=diff[me]-diff[op]
        results.append(dict(game=game,step=step,phase=r['phase'],protected=defended,n=len(legal),**scores,
                            guarded=scores['root'] if choose_root else scores['search'],
                            guarded_joint_match=(r['root_joint'] if choose_root else r['searched_joint'])==r['teacher_joint'],
                            delta_search_minus_root_cash_carried_workers_trees=cash_cargo.round(4).tolist()))
    summary={}
    for phase in ('all','pre','post'):
        rows=[r for r in results if phase=='all' or r['phase']==phase]
        summary[phase]={k:sum(r[k] for r in rows) for k in ('n','root','search','played','guarded','guarded_joint_match')}
        summary[phase]['guarded_joints']=sum(bool(r['protected']) for r in rows)
        summary[phase]['sum_material_search_minus_root']=np.sum([r['delta_search_minus_root_cash_carried_workers_trees'] for r in rows],axis=0).round(3).tolist()
    with a.output.open('x') as f:json.dump(dict(rule='If search changes any eligible worker action with root proposal P>=.90 and landing unreachable by a winning/equal enemy collector, use the complete original root joint proposal. Enemy base cells are exposed. Assessed on the same diagnostic set, without re-search.',summary=summary,samples=results),f,indent=2)
    print(summary,flush=True)


if __name__=='__main__':main()
