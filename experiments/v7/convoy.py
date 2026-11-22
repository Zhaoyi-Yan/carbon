"""Offer a root search branch only when a likely three-turn crossing is valuable."""
import argparse,inspect,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FIT=Path(__file__).with_name('convoy-model.json')


def _convoy_context(players,side,pos,load):
    own,foe=players[side],players[1-side]
    def nearest(cells):
        return _CONVOY_D[cells].min(0) if cells else np.full(225,16,np.int8)
    homes=nearest(list(own[1].values()));enemyhome=nearest(list(foe[1].values()))
    weak=[p for p,v,k in foe[2].values() if k=='COLLECTOR' and v<=load]
    prey=[p for p,v,k in foe[2].values() if k=='COLLECTOR' and v>=max(45.,load+25)]
    return homes,enemyhome,nearest(weak),nearest(prey),min(2.5,load/90.)


def _convoy_inputs(pos,prev,ctx):
    home,hostile,risk,prey,load=ctx
    dest=_CONVOY_L[pos];moves=np.arange(5)[None,:]
    wait=np.broadcast_to(moves==0,dest.shape);same=moves==prev[:,None]
    prev_wait=prev[:,None]==0
    reverse=moves==np.array([0,3,4,1,2])[prev,None]
    hr=home[pos,None]-home[dest];enemy=hostile[pos,None]-hostile[dest]
    rich=load>=.65
    danger=(risk[dest]<2);contact=(risk[dest]==0)
    preyward=np.clip(prey[pos,None]-prey[dest],-1,1)*(prey[dest]<=3)
    return np.stack((wait,wait*load,same&~prev_wait,same*load,reverse&~prev_wait,wait*prev_wait,
                     hr,hr*load,hr*rich,hr*(home[pos,None]<=3),home[dest]==0,(home[dest]==0)*load,
                     enemy,enemy*load,hostile[dest]==0,danger,contact,danger*load,
                     danger*(risk[pos,None]<=2),preyward,preyward/(1.+load)),axis=2).astype(np.float32)


def _convoy_next(pos,prev,ctx):
    z=_convoy_inputs(pos,prev,ctx)@_CONVOY_W
    w=np.exp(z-z.max(axis=1,keepdims=True))
    return w/w.sum(axis=1,keepdims=True)


def _convoy_paths(pos,root,ctx,depots=()):
    # maps[first direction, time-1, cell] conditional on the first move, then marginalized.
    conditional=np.zeros((5,3,225),np.float32)
    first=np.arange(5);now=_CONVOY_L[pos];direction=first.copy();mass=np.ones(5)
    cash_cond=np.zeros_like(conditional) if depots else None
    unbanked=mass.copy();depot_mask=np.zeros(225,bool)
    if depots:depot_mask[list(depots)]=True
    for h in range(3):
        if depots:
            # A trip's cargo is paid to the owner of the first depot reached.
            # Once paid, this original cargo cannot be intercepted on a later square.
            unbanked[depot_mask[now]]=0.
            cash_cond[:,h,:]=np.bincount(first*225+now,weights=unbanked,minlength=5*225).reshape(5,225)
        conditional[:,h,:]=np.bincount(first*225+now,weights=mass,minlength=5*225).reshape(5,225)
        if h<2:
            prob=_convoy_next(now,direction,ctx)
            first=np.repeat(first,5);mass=(mass[:,None]*prob).ravel()
            if depots:unbanked=(unbanked[:,None]*prob).ravel()
            now=_CONVOY_L[now].ravel();direction=np.tile(np.arange(5),len(prob))
    paths=conditional,np.einsum('a,atp->tp',root,conditional)
    return paths+(cash_cond,np.einsum('a,atp->tp',root,cash_cond)) if depots else paths


def _convoy_plan(obs,base,work,foework,locked):
    me=obs['player'];players=obs['players'];own,enemy=players[me],players[1-me]
    if obs['step']<18 or LAST_TURN+1-obs['step']<3:return None
    mine=[(u,p,v) for u,(p,v,k) in own[2].items() if k=='COLLECTOR']
    rivals=[(u,p,v) for u,(p,v,k) in enemy[2].items() if k=='COLLECTOR']
    if not any(_CONVOY_D[p,q]<=6 and ((v>=65. and w<=v-10) or (w>=65. and v<=w-25))
               for _,p,v in mine for _,q,w in rivals):return None
    dest={u:_CONVOY_L[p,WIDX.get(base.get(u),0)] for u,(p,v,k) in own[2].items()}
    recruits={p for u,p in own[1].items() if base.get(u)}
    foe_depots=set(enemy[1].values());all_depots=foe_depots|set(own[1].values())
    own_paths={};foe_paths={}
    def predict(u,p,v,side,logit,cache):
        if u not in cache:
            z=logit[p].astype(np.float64);z-=z.max()
            prob=np.exp(z);prob/=prob.sum()
            cache[u]=(prob,)+_convoy_paths(p,prob,_convoy_context(players,side,p,v),all_depots)
        return cache[u]
    best_score,best_act=0.,None
    for uid,pos,load in mine:
        if uid in locked:continue
        pairs=[(k,q,v) for k,q,v in rivals if _CONVOY_D[pos,q]<=6 and
               ((load>=65. and v<=load-10) or (v>=65. and load<=v-25))]
        if not pairs:continue
        prior,paths,_,unpaid,_=predict(uid,pos,load,me,work,own_paths)
        start=WIDX.get(base.get(uid),0);homes=list(own[1].values())
        if not homes:continue
        dh=_CONVOY_D[homes].min(0)[_CONVOY_L[pos]]
        avoid={x for k,x in dest.items() if k!=uid}|recruits|foe_depots
        for rid,rp,rv in pairs:
            _,_,forecast,_,valuable=predict(rid,rp,rv,1-me,foework,foe_paths)
            if load>=65. and rv<=load-10:
                # A carrier's alternative must remove a credible crossing. Extra retreat by
                # itself has no score benefit; a long detour increases the threshold.
                risk=np.einsum('atp,tp->at',unpaid,forecast)
                if np.max(risk[start])<.17:continue
                utility=load*(risk[start]-risk)@np.array([1.,.64,.39])
                utility-=13.*np.maximum(0.,dh-dh[start])
                min_score=14.
            else:
                # Occupying a depot gives the enemy the cargo. Score only approaches away from it.
                contact=np.einsum('atp,tp->at',paths,valuable)
                if contact.max()<.16:continue
                utility=(rv-load)*(contact-contact[start])@np.array([1.,.68,.43])
                min_score=17.
            utility-=5.+4.*np.maximum(0.,np.log((prior[start]+.005)/(prior+.005)))
            for a,q in enumerate(_CONVOY_L[pos]):
                if a==start or q in avoid or utility[a]<=min_score:continue
                # Do not send a would-be interceptor into an immediate credible losing collision.
                if q not in all_depots and any(v<=load and _CONVOY_D[j,q]<=1 and
                       predict(k,j,v,1-me,foework,foe_paths)[2][0,q]>.16 for k,j,v in rivals):continue
                if utility[a]>best_score:
                    best_score=float(utility[a]);best_act=dict(base)
                    if a:best_act[uid]=WACT[a]
                    else:best_act.pop(uid,None)
    return best_act


def build(code):
    model=json.loads(FIT.read_text());anchor='def _act(obs, configuration=None, t0=None):'
    where="    if most >= 3:\n        for aw, ab in _ST['v5_history']['alternatives']"
    if '_CONVOY_W=' in code or code.count(anchor)!=1 or code.count(where)!=1:raise ValueError('Expected a compatible frozen root search')
    init=("_CONVOY_D=np.asarray(FEAT_DIST,dtype=np.int16)\n_CONVOY_L=np.asarray([[move(p,a) for a in range(5)] for p in range(225)])\n"
          +"_CONVOY_W=np.asarray("+repr(model['weights'])+",dtype=np.float32)\n")
    source='\n\n'.join(inspect.getsource(f) for f in (_convoy_context,_convoy_inputs,_convoy_next,_convoy_paths,_convoy_plan))
    code=code.replace(anchor,init+'\n'+source+'\n\n'+anchor)
    extra="""    if most>=3 and len(cands)<most:
        convoy=_convoy_plan(obs,act0,lw0,_ST['v5_history']['opponent'][0],fixed or {})
        if convoy is not None:
            tag=tuple(sorted(convoy.items()))
            if tag not in keys:
                cands.append(convoy)
                keys.add(tag)
"""
    return code.replace(where,extra+where)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();text=build(a.parent.read_text());compile(text,'<convoy>','exec')
    with a.output.open('x') as f:f.write(text)
