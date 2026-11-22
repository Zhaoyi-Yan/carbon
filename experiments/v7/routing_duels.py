"""Choose between two complete policies for 25 turns, then continue automatically on the same old map."""
import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path
from branch_replay import HERE,ROOT,OUT,step
from benchmark import configure_engine,new_game,observe,validate_actions
from close_replay import archive_state,restore
from crosscurrent_rescue import load_dual
from phase_switch import preserved_decision,single_thread
from preempt_tree_probe import model,budget


def schedule():
    choices=[]
    for fold,rounds in [('dev',(1,2,3,4)),('hold',(5,6,7,8))]:
        for r in rounds:
            source=OUT/f'{r:02d}-qualifying.jsonl';pool=[json.loads(l) for l in source.open()]
            used=set()
            for j in (0,1):
                t=(150,190)[(r+j)%2]
                # Alternate leading and trailing according to the current cash only.
                def eligible(row):
                    if row['status']!='completed' or row['turns']<299 or row['layout_sha256'] in used:return False
                    own=row['actions_and_post_cash'][t-1][1];return (own[j]>own[1-j])==bool((r+j)%2)
                eligible_rows=[x for x in pool if eligible(x)]
                if not eligible_rows:eligible_rows=[x for x in pool if x['status']=='completed' and x['turns']==299 and x['layout_sha256'] not in used]
                row=min(eligible_rows,key=lambda x:hashlib.sha256(f"policy-block-25/{x['layout_sha256']}/{x['players'][j]['name']}".encode()).digest())
                used.add(row['layout_sha256']);choices.append((fold,j,t,source,row))
    assert len({c[-1]['layout_sha256'] for c in choices})==len(choices)
    return choices


def record(m,obs,actual,proposed):
    m._ST['tr'].set_actions(obs['players'],actual)
    if 'liquidity_action' in m._ST:m._ST['liquidity_action']=dict(actual)
    p=m._ST.get('v7_next_depot')
    if p and proposed.get(p[1])!=actual.get(p[1]):m._ST.pop('v7_next_depot',None)
    if 'corridor_track' in m._ST:m._ST['corridor_track']=[x for x in m._ST['corridor_track'] if proposed.get(x[1])==actual.get(x[1])]


def experiment(m,e,row,side,start):
    st,env=new_game(row['seed'],300);cfg=dict(env.configuration);visible={k:v for k,v in cfg.items() if k!='randomSeed'}
    salt=int(row['layout_sha256'][-6:],16)%99991
    m._BRANCH_SEED=salt;m._CC_WARM=True;budget(e,False)
    for n in (m,e):n._ST={};n._FC_HISTORY={}
    mods=[m,e] if side==0 else [e,m]
    for t,(actions,balances) in enumerate(row['actions_and_post_cash'][:start]):
        for seat,n in enumerate(mods):
            o=observe(st,seat,t);a=preserved_decision(n,o,visible,salt);record(n,o,actions[seat],a)
        step(st,env,actions);assert [p[0] for p in st[0].observation.players]==balances
    mem=copy.deepcopy([(n._ST,n._FC_HISTORY) for n in mods]);snap=archive_state(st)
    ps=st[0].observation.players;cash=ps[side][0]-ps[1-side][0];flow=m._v7_receipts(ps,start,298);diff=flow[side]-flow[1-side]
    features=dict(cash=round(cash,2),projected=round(float(cash+.45*diff[0]+.55*diff[1]),2),
                  trees=len(ps[side][3])-len(ps[1-side][3]),workers=len(ps[side][2])-len(ps[1-side][2]),
                  normal=m._v7_choose_team(observe(st,side,start),m._ST))
    m._CC_WARM=False;budget(e,True);original=m._v7_choose_team;arms={}
    try:
        for arm in ('pursuit','reserve'):
            st,env=restore(snap,cfg)
            for n,(a,b) in zip(mods,copy.deepcopy(mem)):n._ST,n._FC_HISTORY=a,b
            m._v7_choose_team=lambda obs,state:arm if obs['step']<start+25 else original(obs,state)
            marks={};m._CC_BUDGET.clear()
            for t in range(start,299):
                actions=[]
                for seat,n in enumerate(mods):
                    o=observe(st,seat,t);a=preserved_decision(n,o,visible,salt);validate_actions(a,o['players'][seat]);actions.append(a)
                step(st,env,actions);ps=st[0].observation.players
                done=t==298 or any(s.status!='ACTIVE' for s in st)
                if (t+1-start) in (10,25,50) or done:marks[t+1]=round(ps[side][0]-ps[1-side][0],2)
                if done:break
            assert len(m._CC_BUDGET)==t+1-start
            scores=[s.reward for s in st];arms[arm]=dict(end=t+1,margin=round(scores[side]-scores[1-side],2),marks=marks)
    finally:m._v7_choose_team=original
    return features,arms


def pick(rule,feat):
    if rule=='incumbent':return feat['normal']
    if rule=='pursuit' or rule=='reserve':return rule
    scores={'cash':feat['cash'],'projected':feat['projected'],
            'productive_assets':feat['projected']+75*feat['trees']+50*feat['workers']}
    return 'reserve' if scores[rule]>0 else 'pursuit'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    single_thread();configure_engine();m=load_dual(HERE/'checkpoints/crosscurrent.py');e=model(ROOT/'yan-v6.py','duel_estuary',True)
    choices=schedule();design=[dict(fold=f,seat=s,turn=t,source=p.name,seed=r['seed'],layout=r['layout_sha256']) for f,s,t,p,r in choices]
    rules=('incumbent','cash','projected','productive_assets','pursuit','reserve')
    d=dict(design=design,rules=rules,method='One branch chooses a policy for 25 turns, then both branches use exactly the same public score-based automatic router until the official terminal. Frozen v6 reacts independently after the historical prefix. Both policies use fixed width 4/depth 2/one reply/three-ply tail. Each old map appears once. Selection maximizes dev terminal wins, then mean cash. No current qualification maps used.',cases=[])
    a.output.write_text(json.dumps(d,indent=2)+'\n')
    def assess(fold):
        rows=[r for r in d['cases'] if r['fold']==fold];out={}
        for rule in rules:
            values=[r['arms'][pick(rule,r['features'])]['margin'] for r in rows]
            out[rule]=dict(wins=sum(v>0 for v in values),mean=round(statistics.mean(values),2))
        return out
    for (fold,side,t,path,r),meta in zip(choices,design):
        f,x=experiment(m,e,r,side,t);d['cases'].append(dict(meta,features=f,arms=x))
        print(fold,side,t,r['seed'],f['normal'],x['pursuit']['margin'],x['reserve']['margin'],flush=True)
        if len(d['cases'])==8:
            d['dev']=assess('dev');d['chosen']=max(rules,key=lambda rule:(d['dev'][rule]['wins'],d['dev'][rule]['mean']))
        if len(d['cases'])==16:d['holdout']=assess('hold')
        a.output.write_text(json.dumps(d,indent=2)+'\n')


if __name__=='__main__':main()
