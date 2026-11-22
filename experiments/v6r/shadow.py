"""Use a compact frozen v5 network to propose collector moves and predict searched replies."""
import argparse
import base64
import inspect
import io
from pathlib import Path
import re
import numpy as np


def _shadow_policies(x,other,players,side,style,enemy_style):
    views=np.stack([x,other]).reshape(2,NPL,N)
    rotations=(_V5_ROTATIONS[0],_V5_ROTATIONS[2]);batch=[]
    for cells,inverse,actions,channels in rotations:
        batch.append(views[:,channels][:,:,inverse].reshape(2,NPL,SIZE,SIZE))
    ids=np.array([style,enemy_style]*2)
    hidden=_R6_OLD_NET(np.concatenate(batch).astype(np.float32),ids,hidden=True)
    policy=_R6_OLD_POLICY.p;out=[]
    for index,ident in enumerate((style,enemy_style)):
        w=np.zeros((N,5),'f');b=np.zeros((N,3),'f');p=players[side if index==0 else 1-side]
        for i,(cells,inverse,actions,channels) in enumerate(rotations):
            h=hidden[2*i+index,cells]
            w+=.5*((h@policy['hw.w'].reshape(5,-1).T+policy['hw.b'])[:,actions])
            b+=.5*(h@policy['hb.w'].reshape(3,-1).T+policy['hb.b'])
            for pos,load,kind in p[2].values():
                role=int(kind=='PLANTER')
                w[pos]+=.135*(_V5['coef'][ident,role]@h[pos]+_V5['bias'][ident,role])[actions]
            for pos in p[1].values():
                b[pos]+=.135*(_V5['coef'][ident,2,:3]@h[pos]+_V5['bias'][ident,2,:3])
        out.append((w,b))
    return out


def _shadow_update(obs,x,opponent_x,state,mine,econ,styles,style,rival):
    side=obs['player'];op=1-side;players=obs['players'];foe=players[op];step=obs['step']
    if step==1:
        first=[k for _,_,k in foe[2].values()]
        state['shadow_open']=1.5 if first==['COLLECTOR','COLLECTOR'] else -1.5
    past=[]
    for uid,(pos,delta) in state.get('shadow_pending',{}).items():
        if uid in foe[2] and foe[2][uid][0] in _ROLLOUT_LANDING[pos]:
            a=_ROLLOUT_LANDING[pos].index(foe[2][uid][0]);past.append(delta[a])
    edge=state.get('shadow_evidence',0.)*.94+(float(np.mean(past)) if past else 0.)
    state['shadow_evidence']=np.clip(edge,-4.,4.)
    alpha=1./(1.+np.exp(-state.get('shadow_open',-1.)-2.0*edge))
    _,context=_v5_context(econ,styles,players,step,LAST_TURN,op)
    # Estimate which of v5's original teachers its own adaptive router favors.
    phase=min(3,step*4//max(1,LAST_TURN+1))
    bias=((_V5['prototypes'][phase]-styles[side])/_V5['scale'])**2
    prob=np.exp(np.clip(-.4*(bias.sum(1)-bias.sum(1).min()),-30.,0));prob/=prob.sum()
    merit=_V5['route']@context-.80*_V5['route_center']+.20*(_V5['matchup']@prob)-.024*np.log1p(np.arange(20))
    foe_id=int(merit.argmax()) if step>=12 else 0
    old,shadow=_shadow_policies(x,opponent_x,players,side,style,foe_id)
    hybrid=mine[0].copy()
    for pos,cargo,k in players[side][2].values():
        if k=='COLLECTOR':hybrid[pos]=old[0][pos]
    state['alternatives'].insert(1,(hybrid,old[1]))
    current=state['opponent']
    state['shadow_pending']={}
    for uid,(pos,cargo,k) in foe[2].items():
        p=_probs(shadow[0][pos],.85);q=_probs(current[0][pos],.85)
        state['shadow_pending'][uid]=(pos,np.clip(np.log(np.maximum(p,1.e-7))-np.log(np.maximum(q,1.e-7)),-3.,3.))
    state['opponent']=tuple((1-alpha)*a+alpha*b for a,b in zip(current,shadow))
    state['shadow_alpha']=alpha;state['shadow_style']=foe_id


def _shadow_reply(root,config,mine,enemy_logits,seed,t0):
    history=_ST['v5_history'];op=1-root['player'];step=root['step'];players=root['players']
    if history.get('shadow_alpha',0.)<.50 or _time.perf_counter()-t0>.28 or not players[op][2]:return None
    w,b=enemy_logits;options=[];seen=set()
    for k,(temp,ofs) in enumerate(((1.,7),(.60,13),(1.5,31),(1.8,59),(1.1,83))):
        a=_sample(players,op,w/temp,b/temp,np.random.RandomState(seed+ofs),step)
        key=tuple(sorted(a.items()))
        if key not in seen:seen.add(key);options.append(a)
    if len(options)<2:return options[0]
    boards=[]
    for choice in options:
        acts=[mine,choice] if op==1 else [choice,mine]
        boards.append(_turn(root,config,acts))
    X=np.stack([_v5_pair(g['players'],g['step'],g['carbon'],players)[op] for g in boards])
    S=np.stack([_scalars(g['players'],g['step'],op) for g in boards])
    _,_,value=_R6_OLD_NET(X,history['shadow_style'],S)
    estimates=np.asarray([_v5_cash_only(g,op) for g in boards])
    score=.70*value+.30*estimates
    best=int(score.argmax())
    return options[best] if score[best]>score[0]+.007 else options[0]


def build(code,source):
    raw={}
    for k in ('_WEIGHTS','_VALUE'):
        encoded=re.search('^'+k+r' = "([^"]*)"$',source,re.M)[1]
        with np.load(io.BytesIO(base64.b64decode(encoded)),allow_pickle=False) as f:raw.update(dict(f))
    buf=io.BytesIO();np.savez_compressed(buf,**{k:v.astype('f2') if v.dtype.kind=='f' else v for k,v in raw.items()})
    payload=base64.b64encode(buf.getvalue()).decode()
    init="""if _SEARCH_OK:
    _SHADOW_DATA='ARCHIVE'
    _R6_OLD_POLICY=_Policy(_SHADOW_DATA)
    with np.load(io.BytesIO(base64.b64decode(_SHADOW_DATA)),allow_pickle=False) as _ow:
        _R6_OLD_NET=_build(_R6_OLD_POLICY,{k:_ow[k].astype('f') for k in _ow if k.startswith(('v1.','v2.','v3.'))})
""".replace('ARCHIVE',payload)
    def change(old,new):
        nonlocal code
        if code.count(old)!=1:raise ValueError(old)
        code=code.replace(old,new)
    change('\n_net()\n','\n_net()\n'+init)
    anchor='def _v5_root(obs,'
    change(anchor,inspect.getsource(_shadow_policies)+'\n'+inspect.getsource(_shadow_update)+'\n'+inspect.getsource(_shadow_reply)+'\n'+anchor)
    anchor="    state['pending'] = {}"
    change(anchor,'    _shadow_update(obs,x,opponent_x,state,mine,econ,styles,style,rival)\n'+anchor)
    anchor='    opp = {}\n'
    change(anchor,anchor+"""    first_reply=_shadow_reply(root,configuration,act0,(lwo,lbo),seed0,t0)
    if first_reply is not None:opp[0]=first_reply
""")
    compile(code,'<shadow>','exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('frozen_v5',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(build(a.source.read_text(),a.frozen_v5.read_text()))
