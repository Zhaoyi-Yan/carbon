"""Assign inexpensive collectors to different patches by expected harvest and delivery time."""
import argparse
import inspect
from pathlib import Path


def _r6_patch_proposal(obs,stock,act,locked,logits):
    me=obs['player'];step=obs['step'];pl=obs['players'];own,foe=pl[me],pl[1-me]
    if step<3 or LAST_TURN-step<12:return None
    workers=[(uid,p,v) for uid,(p,v,role) in own[2].items() if role=='COLLECTOR']
    hunters=[(p,v) for p,v,r in foe[2].values() if r=='COLLECTOR']
    # Aim this extra proposal at economic workers while retaining the raw policy
    # for contact, defense, and any imminent strike on an opposing shipment.
    available=[(uid,p,v) for uid,p,v in workers if uid not in locked and v<75 and
               all(_V5_DIST[p][q]>2 or c<=v+15 for q,c in hunters)]
    if not available:return None
    grid=_V5_GEOMETRY;land=_V5_REACH;homes=np.asarray(_v5_home(tuple(own[1].values())))
    resource=np.asarray(stock);blocked=np.zeros(N,dtype=bool);tree_count=np.zeros(N)
    for team in pl:
        blocked[list(team[1].values())]=True
        for p,age in team[3].values():
            blocked[p]=True;tree_count[_V5_ADJ[p]]+=min(1.,(51-age)/8.)
    growth=np.where(tree_count>0,1.-.0375*tree_count,1.03)
    enemy_home=np.asarray(_v5_home(tuple(foe[1].values())))
    assignments=[];base_ends={u:land[p,WIDX.get(act.get(u),0)] for u,(p,_,_) in own[2].items()}
    hires={q for uid,q in own[1].items() if uid in act}
    for uid,p,cargo in available:
        distance=grid[p]
        incoming=np.minimum(100.,resource*np.power(growth,np.minimum(distance,8)))
        rival_dist=np.minimum.reduce([grid[q] for q,v in hunters]) if hunters else np.full(N,15)
        safety=np.clip(.53+.105*(rival_dist-np.minimum(distance,5)),.27,1.)
        safety*=np.clip(.80+.07*enemy_home,.75,1.)
        score=np.zeros(N)
        for dwell in (2,4,6):
            time=distance+dwell+homes
            take=incoming*(1-.70**dwell)
            # Quick repeated deposits have more value when another recruit is unfunded.
            value=(take+min(cargo,65.))/(1.+distance+dwell+(.65 if own[0]>90 else 1.)*homes)
            score=np.maximum(score,value*safety*(time<=LAST_TURN-step+1))
        score[blocked|(distance>7)]=0.
        for target in np.argsort(-score)[:7]:
            if score[target]<4.:continue
            moves=[a for a,q in enumerate(land[p]) if (a==0 and target==p) or (a and grid[q,target]<distance[target])]
            safe=[]
            for a in moves:
                q=land[p,a]
                if q in hires or any(v<=cargo and _V5_DIST[q][ep]<=1 for ep,v in hunters):continue
                if any(ou!=uid and dest==q for ou,dest in base_ends.items()):continue
                safe.append(a)
            if not safe:continue
            a=max(safe,key=lambda move:logits[p,move])
            mode=WIDX.get(act.get(uid),0);old=land[p,mode]
            # Measured advantage over the best patch in the direction already selected.
            if a==mode:prior=score[target]
            else:
                compatible=grid[old]<distance if mode else (np.arange(N)==p)
                prior=score[compatible].max(initial=0.)
            assignments.append((score[target],score[target]-prior,uid,target,a,land[p,a]))
    if not assignments:return None
    picked=set();patches=set();destinations=set();out=act.copy()
    for merit,adv,uid,target,a,q in sorted(assignments,reverse=True):
        if uid in picked or target in patches:continue
        if q in destinations:continue
        if adv<.65 and a!=WIDX.get(act.get(uid),0):continue
        picked.add(uid);patches.add(target);destinations.add(q)
        if a:out[uid]=WACT[a]
        else:out.pop(uid,None)
    return out if out!=act else None


def build(code):
    a='def _search(obs,';b='    for T, cap in _S_GEN:'
    if code.count(a)!=1 or code.count(b)!=1:raise ValueError('incompatible planner')
    code=code.replace(a,inspect.getsource(_r6_patch_proposal)+'\n\n'+a)
    extra="""    if most>=4 and len(cands)<most:
        patch=_r6_patch_proposal(obs,est,act0,fixed or {},lw0)
        if patch is not None:
            marker=tuple(sorted(patch.items()))
            if marker not in keys:cands.append(patch);keys.add(marker)
"""
    code=code.replace(b,extra+b);compile(code,'<foraging>','exec');return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(build(a.source.read_text()))
