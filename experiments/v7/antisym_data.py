"""Audit archived legal one-step leaves, holding bank/asset blends and the routed critic IDs fixed."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import types
import numpy as np
from distill_data import load,load_manifest,public
from advantage_data import ROOT,configure_engine,prepare,manifest,footprint
from receipt_data import LAND
from benchmark import validate_actions

PARENT=ROOT/'experiments/v7/checkpoints/fairlead.py'
def margin(pl,side):return pl[side][0]-pl[1-side][0]

def values(m,boards,prev,side,priors,search_end=False):
    op=1-side;h=m._ST['v5_history'];net=m._net()
    assert m._RNET is net
    views=np.stack([m._v7_visible_views(b,prev) for b in boards])[:,[side,op]]
    scalars=np.stack([[m._scalars(b['players'],b['step'],s) for s in (side,op)] for b in boards])
    ids=m._v5_ids([side,op]*len(boards))
    big=net(views.reshape(-1,m.NPL,15,15),ids,scalars.reshape(-1,5))[2].reshape(-1,2)
    opp=m._V6R_TINY(np.ascontiguousarray(views[:,1]),ids[1::2],np.ascontiguousarray(scalars[:,1]))[2]
    one=big[:,0];both=.5*(one-big[:,1]);tiny=.5*(one-opp);anchor=tiny+(opp[0]-big[0,1])*.5
    out=[];econ=[]
    for i,b in enumerate(boards):
        h=m.LAST_TURN+1-b['step'];bank=m._banking_value(b['players'],side,b['step'])
        rate=.72*max(0.,22-h)/22
        legacy=m._fc_leaf(b,side,m.LAST_TURN,m._v5_value(b,side))
        near=m._v7_nc_leaf(b,side)
        cooked=lambda raw:(.74*(.71*((1.-rate)*raw+rate*bank)+.29*legacy)+.26*near) if h>0 else (1.-rate)*raw+rate*bank
        out.append([one[i],big[i,1],opp[i],both[i],tiny[i],anchor[i],cooked(one[i]),cooked(both[i]),cooked(tiny[i]),cooked(anchor[i]),priors[i]])
        if search_end and (b.get('_terminal_sign') is not None or b['step']>m.LAST_TURN):
            terminal=m._terminal(margin(b['players'],side)) if b.get('_terminal_sign') is None else 3.*b['_terminal_sign']*(1-2*side)
            out[-1][6:10]=[terminal]*4
        econ.append([float(legacy),float(near)])
    return np.round(out,8).tolist(),econ

def update(m,tracker,o,cfg,previous):
    m._fc_track(o);pl=o['players'];t=o['step']
    est,known=tracker.step(pl,t,o['carbon'])
    return m._sense_tree_carbon(tracker.bel,o,cfg,previous,est,known)

def stockroot(m,o,cfg,prior,tracker,doroot):
    s=o['player'];pl=o['players'];t=o['step'];est,known=update(m,tracker,o,cfg,prior)
    if not doroot and t>0:
        # The observer knows the rival worker positions every single turn even though
        # the expensive router logits are sampled intermittently for this diagnosis.
        m._r6_opponent_view(m._ST['v5_history'],pl,t,1-s,est,prior)
        return None
    x=m._quant(m.encode(pl,t,s,est,known,prior));lw,lb=m._v5_root(o,est,x,prior)
    if not doroot:return None
    root=dict(players=pl,step=t,player=s,trees=o.get('trees',{}),carbon=est,
              _v7_seen=m._ST['v5_history']['forecast_known'])
    return root,lw,lb

def archive(m,rows,fold,out):
    for row in rows:
        if row['source']!='public':continue
        path=out/(row['id']+'.json')
        if path.exists() and json.loads(path.read_text()).get('visibility_revision')==2:continue
        with gzip.open(row['payload']) as f:g=json.load(f)
        frames=g['steps'];record=[]
        for s in (0,1):
            m._ST.clear();m._FC_HISTORY.clear();tracker=m.BeliefTracker(m._Bel,s);prev=None
            for t,o,after,orders,cfg in public(row['payload'],s):
                sample=t>=10 and (t-int(row['group'][:2],16)%16)%16==0
                pos=stockroot(m,o,cfg,prev,tracker,sample);pl=o['players']
                if pos:
                    root,lw,lb=pos;b=m._turn(root,cfg,orders);scores,_=values(m,[b],pl,s,[0.])
                    horizon=min(len(frames)-1,t+1+35);end=frames[-1][0]['observation']['players']
                    label=frames[horizon][0]['observation']['players']
                    record.append(dict(t=t+1,s=s,p=scores[0],terminal=b.get('_terminal_sign') is not None or t+1>=len(frames)-1,cash=float(margin(b['players'],s)),
                        future=float(margin(label,s)),final=float(margin(end,s)),seats=m._v5_ids([s,1-s]).tolist()))
                tracker.set_actions(pl,orders[s]);prev=pl
        path.write_text(json.dumps(dict(layout=row['group'],fold=fold[row['group']],id=row['id'],rows=record,visibility_revision=2),separators=(',',':'))+'\n')
        print('archive',row['id'][:6],len(record),flush=True)

def expected(m,state,t,side,chosen,states,tape,cfg,feedback):
    board=state;mm=[{u:u for u in team[2]} for team in state['players']];marks=[]
    for j in range(35):
        pl=board['players'];real=states[t+j]['players'];after=states[t+j+1]['players']
        acts=tape[t+j][0];orders=[]
        for k in (0,1):
            cmd={u if u in pl[k][1] else mm[k][u]:v for u,v in acts[k].items() if u in pl[k][1] or u in mm[k]}
            if j and feedback:
                for u,v in mm[k].items():
                    w=after[k][2].get(u)
                    if w is None:continue
                    q=pl[k][2][v][0]
                    if q==real[k][2][u][0]:continue
                    land=LAND[q];d=m.FEAT_DIST[list(land),w[0]]
                    act=m.WIDX.get(cmd.get(v),0)
                    if d[act]!=d.min():
                        a=int(d.argmin())
                        if a:cmd[v]=m.WACT[a]
                        else:cmd.pop(v,None)
            orders.append(cmd)
        if not j:orders[side]=chosen
        board=m._turn(board,cfg,orders)
        for k,team in enumerate(board['players']):
            mm[k]={u:v for u,v in mm[k].items() if u in after[k][2] and v in team[2]}
            newborn={(w[0],w[2]):uid for uid,w in team[2].items() if uid not in pl[k][2]}
            for u,w in after[k][2].items():
                if u not in real[k][2] and (w[0],w[2]) in newborn:mm[k][u]=newborn[w[0],w[2]]
        dead=board.get('_terminal_sign')
        if j in (9,34) or dead is not None:
            cash,cargo,near,fleet,trees=footprint(board,side,m)
            score=cash+.55*near+26*fleet+.55*trees+1200*(dead or 0)*(1-2*side)
            marks.append([float(cash),float(score)])
            if dead is not None:
                while len(marks)<2:marks.append(marks[-1])
                break
    return marks

def branch(m,out):
    rows=manifest();selected=set()
    for k in range(1,7):
        training=sorted({a['layout'] for a,r in rows if a['split']=='train' and a['panel']==k})
        selected.add(training[0])
    # Use the existing full simulation's own candidate pool, before the first pruning pass.
    import inspect
    source=inspect.getsource(m._search)
    beg=source.index('    if _S_COST[0] > 0.0:');end=source.index('    stages =',beg)
    source=source[:beg]+'    most,D2,M=6,2,1\n'+source[end:]
    at=source.index('    root_modes={i:mode_keys')
    source=source[:at]+'    return cands\n'
    scope=m.__dict__.copy();exec(source,scope);gen=scope['_search']
    for meta,g in rows:
        if meta['split']=='train' and meta['layout'] not in selected:continue
        dest=out/(meta['name']+'.json')
        if dest.exists() and json.loads(dest.read_text()).get('pool_revision')==3:continue
        s=meta['side'];cfg,states,obs=prepare(m,meta,g)
        jitter=int(meta['layout'][:4],16)%9-4
        ts={q+jitter for q in (31,83,139,194,254)}
        m._ST.clear();m._FC_HISTORY.clear();tr=m.BeliefTracker(m._Bel,s);previous=None;rowsout=[]
        for t in range(max(ts)+1):
            o=obs[t];pl=o['players'];pos=stockroot(m,o,cfg,previous,tr,t in ts)
            orders=g['actions_and_post_cash'][t][0]
            if pos:
                root,lw,lb=pos
                # The logged first joint proposal anchors the pool; all further proposals are the frozen search generators.
                acts=gen(o,cfg,root['carbon'],orders[s],lw,lb,previous,m._time.perf_counter())
                if isinstance(acts,list) and len(acts)>1:
                    for action in acts:validate_actions(action,pl[s])
                    costs=m._r6_action_costs(pl,s,lw,lb,acts,t)
                    boards=[m._turn(root,cfg,[a,orders[1]] if s==0 else [orders[0],a]) for a in acts]
                    scores,e=values(m,boards,pl,s,costs,True)
                    targets=[[expected(m,states[t],t,s,a,states,g['actions_and_post_cash'],cfg,fb) for a in acts] for fb in (False,True)]
                    rowsout.append(dict(t=t,actions=acts,p=scores,e=e,y=targets,terminal=[b.get('_terminal_sign') for b in boards]))
            tr.set_actions(pl,orders[s]);previous=pl
        dest.write_text(json.dumps(dict(**meta,rows=rowsout,pool_revision=3),separators=(',',':'))+'\n')
        print('branch',meta['name'],len(rowsout),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('what',choices=['archive','branch']);a=p.parse_args()
    configure_engine();m=load(PARENT);out=a.data/('antisym-'+a.what);out.mkdir(exist_ok=True)
    if a.what=='archive':
        rows,fold,_=load_manifest(a.data/'distill-public');archive(m,rows,fold,out)
    else:branch(m,out)
