"""Distill short counterfactual returns from six finished paired local panels."""
import argparse
import hashlib
import inspect
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
import numpy as np

import advantage
from branch_replay import step as official_step
from benchmark import configure_engine,new_game,observe,validate_actions
from receipt_data import LAND,ACT
from starvation_replay import landings

ROOT=Path(__file__).resolve().parents[2]
FIELDS=('cash','receipt','cargo','near_cargo','fleet','trees','lost','score','terminal')
def mod(path):
    src=path.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)')
    begin=src.index('    if _S_COST[0] > 0.0:',src.index('def _search('));end=src.index('    stages =',begin)
    src=src[:begin]+'    most, D2, M = 6, min(D2,2), 1\n'+src[end:]
    src=src.replace('for tick in range(1,15):','for tick in range(1,4):')
    anchor='    if _S_REC is not None:\n        _S_REC.append('
    if src.count(anchor)!=1:raise ValueError('Root scoring hook changed')
    src=src.replace(anchor,"    _ADV_CAPTURE(obs,est,cands,alive,mean,best)\n"+anchor)
    m=types.ModuleType('paired_action_oracle');exec(compile(src,str(path),'exec'),m.__dict__)
    m._torch.set_num_threads(1);m._time=types.SimpleNamespace(perf_counter=lambda:0.);m._S_MMAX=1
    advantage._ADV_L=np.asarray(LAND);advantage.FEAT_DIST=m.FEAT_DIST;advantage.WIDX=m.WIDX;advantage.LAST_TURN=m.LAST_TURN
    return m

def manifest():
    rows=[]
    for j in range(1,7):
        path=ROOT/f'analysis/v7/{j:02d}-qualifying.jsonl';games=[json.loads(line) for line in path.open()]
        groups=sorted({r['layout_sha256'] for r in games},key=lambda q:hashlib.sha256(('joint-credit-v1'+q).encode()).hexdigest())
        assert len(games)==16 and len(groups)==8
        for r in games:
            assert r['status']=='completed' and len(r['actions_and_post_cash'])==299
            s=next(i for i,p in enumerate(r['players']) if p['name']!='yan-v6')
            rows.append((dict(panel=j,seed=r['seed'],layout=r['layout_sha256'],side=s,version=r['players'][s]['name'],
                         candidate_won=r['winner']==s,split='test' if r['layout_sha256'] in groups[:2] else 'train',
                         name=f"{j}-{r['seed']}-{s}"),r))
    assert len({v['layout'] for v,r in rows})==48
    return rows

def footprint(board,side,m):
    vals=[]
    for j,p in enumerate(board['players']):
        near=0.;cash=p[0];cargo=sum(v for q,v,k in p[2].values());homes=list(p[1].values())
        for q,v,k in p[2].values():
            if v:
                d=min(m.FEAT_DIST[q,h] for h in homes);near+=v*max(0.,1.-.095*d)
        tree_eq=sum(min(12.,50.-age)*min(10.,board['_st'].trees_meta.get(u,(None,0.))[1]) for u,(q,age) in p[3].items())
        vals.append(np.array([cash,cargo,near,len(p[2]),tree_eq]))
    return vals[side]-vals[1-side]

def counterfactual(m,start,t,me,cmd,states,tape,cfg,feedback):
    board=start;receipts=np.zeros(2);lost=np.zeros(2);markers=[];mm=[{u:u for u in p[2]} for p in start['players']]
    for j in range(10):
        truth=states[t+j]['players'];after=states[t+j+1]['players'];teams=board['players']
        raw=tape[t+j][0];orders=[]
        for s,pl in enumerate(teams):
            mapped={}
            for u,a in raw[s].items():
                if u in pl[1]:mapped[u]=a
                elif u in mm[s]:mapped[mm[s][u]]=a
            if feedback and j:
                for u,alt in mm[s].items():
                    if u not in after[s][2]:continue
                    p=pl[2][alt][0];q0=truth[s][2][u][0]
                    if p==q0:continue
                    goal=after[s][2][u][0]
                    choices=LAND[p];go=np.array([m.FEAT_DIST[q,goal] for q in choices])
                    former=m.WIDX.get(mapped.get(alt),0)
                    move=former if go[former]==go.min() else int(go.argmin())
                    if move:mapped[alt]=m.WACT[move]
                    else:mapped.pop(alt,None)
            orders.append(mapped)
        if j==0:orders[me]=cmd
        homes={q:s for s,p in enumerate(teams) for q in p[1].values()}
        for q,group in landings(teams,orders,cfg).items():
            bank=homes.get(q)
            if bank is not None:receipts[bank]+=sum(w[2] for w in group)
            if len(group)<2 and bank is None:continue
            cm=[w for w in group if w[3]=='COLLECTOR'];low=min((w[2] for w in cm),default=None)
            light=[w for w in cm if w[2]==low];winner=light[0][0] if len(light)==1 else None
            for s,u,v,k in group:
                if (bank is not None and bank!=s) or (bank is None and len(group)>1 and winner!=s):lost[s]+=v
        board=m._turn(board,cfg,orders)
        for s,pl in enumerate(board['players']):
            new={u:v for u,v in mm[s].items() if u in after[s][2] and v in pl[2]}
            births={(w[0],w[2]):u for u,w in pl[2].items() if u not in teams[s][2]}
            for u,w in after[s][2].items():
                if u not in truth[s][2] and (w[0],w[2]) in births:new[u]=births[w[0],w[2]]
            mm[s]=new
        exhausted=[not p[2] and not p[3] and (not p[1] or p[0]<min(cfg['recCollectorCost'],cfg['recPlanterCost'])) for p in board['players']]
        if j in (4,9) or any(exhausted):
            cash,cargo,near,fleet,trees=footprint(board,me,m)
            bank=receipts[me]-receipts[1-me];sunk=lost[me]-lost[1-me]
            terminal=int(exhausted[1-me])-int(exhausted[me])
            worth=cash+.55*near+26.*fleet+.55*trees+1200.*terminal
            point=[cash,bank,cargo,near,fleet,trees,sunk,worth,terminal]
            markers.append(point)
            if any(exhausted):
                while len(markers)<2:markers.append(point)
                break
    return np.asarray(markers).round(4).tolist()

def prepare(m,meta,r):
    s=meta['side'];seed=r['seed'];state,env=new_game(seed,300);cfg=dict(env.configuration)
    o=state[0].observation;initial=dict(players=o.players,carbon=o.full_carbon,trees=o.trees,step=0,player=s)
    true=[initial];obs=[]
    for t,(act,cash) in enumerate(r['actions_and_post_cash']):
        obs.append(observe(state,s,t));official_step(state,env,act);true.append(m._turn(true[-1],cfg,act))
        assert [p[0] for p in state[0].observation.players]==cash and [p[0] for p in true[-1]['players']]==cash
        assert all(set(x[2])==set(y[2]) and set(x[3])==set(y[3]) for x,y in zip(state[0].observation.players,true[-1]['players']))
    return cfg,true,obs

def collect(m,meta,r):
    cfg,true,obs=prepare(m,meta,r);s=meta['side']
    jitter=int(meta['layout'][:4],16)%9-4
    times={t+jitter for t in (18,48,85,122,156,190,225,267)}
    real=m._search;records=[];m._ST.clear();m._FC_HISTORY.clear()
    m._ST['tr']=m.BeliefTracker(m._Bel,s);m._ST['rng']=np.random.RandomState(m._SEED);m._ST['prev']=None
    # We evaluate the completed search. The separate final collision correction is a different experiment.
    endfix=m._reply_choice;m._reply_choice=lambda o,c,e,a,*args:a
    for t in range(max(times)+1):
        o=obs[t];pl=o['players']
        if t in times:
            box={}
            def capture(o,est,cands,alive,mean,best):
                ids=list(dict.fromkeys([0]+list(alive)));acts=[cands[k] for k in ids]
                f=advantage._adv_joint(o,est,m._ST['v5_history']['opponent'][0],acts)
                box.update(acts=acts,features=f.tolist(),eligible=[ids.index(i) for i in alive],winner=ids.index(best),
                           oldscore=[float(mean[k]) for k in ids])
            m._ADV_CAPTURE=capture;m._search=real;m._act(o,cfg,0.)
            if box:
                ys=[];ys2=[];hamming=[];logged=r['actions_and_post_cash'][t][0][s]
                for a in box['acts']:
                    validate_actions(a,pl[s])
                    hamming.append(sum(a.get(k)!=logged.get(k) for k in list(pl[s][1])+list(pl[s][2])))
                    ys.append(counterfactual(m,true[t],t,s,a,true,r['actions_and_post_cash'],cfg,False))
                    ys2.append(counterfactual(m,true[t],t,s,a,true,r['actions_and_post_cash'],cfg,True))
                records.append(dict(t=t,**box,literal=ys,waypoint=ys2,teacher_distance=hamming))
        elif t+1 in times:
            m._search=lambda o,c,e,a,*args:a;m._act(o,cfg,None)
        else:
            m._fc_track(o);tr=m._ST['tr'];est,known=tr.step(pl,t,o['carbon']);m._sense_tree_carbon(tr.bel,o,cfg,m._ST['prev'],est,known)
        m._ST['tr'].set_actions(pl,r['actions_and_post_cash'][t][0][s]);m._ST['prev']=pl
    m._search=real;m._reply_choice=endfix
    return dict(**meta,records=records)
def main(a):
    configure_engine();a.cache.mkdir(exist_ok=True,parents=True);m=mod(a.parent)
    for j,(meta,r) in enumerate(manifest()):
        out=a.cache/(meta['name']+'.json')
        if out.exists():continue
        record=collect(m,meta,r);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(record,separators=(',',':')));tmp.replace(out)
        print(j+1,meta['version'],meta['seed'],len(record['records']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('cache',type=Path);main(p.parse_args())
