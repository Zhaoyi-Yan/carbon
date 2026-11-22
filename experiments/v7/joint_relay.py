"""Grow a genuine preferred-destination conflict along plausible second-choice yielding chains."""
from collections import Counter
import numpy as np
from joint_assign import ACT,L


def propose(obs,base,logits,locked,assign,min_gain=.25):
    units=obs['players'][obs['player']];workers=units[2];homes=set(units[1].values())
    idx={u:ACT.index(base.get(u)) for u in workers}
    end={u:int(L[q,idx[u]]) for u,(q,c,k) in workers.items()}
    n=Counter(end.values())
    fixed={u for u,(q,c,k) in workers.items() if u in locked or n[end[u]]>1 or end[u] in homes and c>0}
    blocked={end[u] for u in fixed}|{q for b,q in units[1].items() if b in base}
    free={u for u in workers if u not in fixed};owner={end[u]:u for u in free}
    if len(free)<2:return base,None,0
    best={};second={};rows={}
    for u in sorted(free):
        q=workers[u][0];v=logits[q]
        row={int(z):(float(v[a]),a) for a,z in enumerate(L[q]) if z not in blocked and np.isfinite(v[a])}
        if not row:return base,None,0
        seq=sorted(row,key=lambda z:(-row[z][0],row[z][1]));best[u]=seq[0];rows[u]=row
        second[u]=[z for z in seq[:2] if row[seq[0]][0]-row[z][0]<=2.]
    seeded=set()
    for a in free:
        for b in free:
            if a!=b and (best[a]==best[b] or best[a]==end[b]):seeded.update((a,b))
    if not seeded:return base,None,0
    edges={u:set() for u in free}
    for a in free:
        for b in free:
            if a!=b and (best[a]==best[b] or end[b] in second[a]):
                edges[a].add(b);edges[b].add(a)
    seen=set();offers=[];conflicts=0
    for root in sorted(seeded):
        if root in seen:continue
        comp=[root];seen.add(root)
        for u in comp:
            for v in sorted(edges[u]-seen):seen.add(v);comp.append(v)
        comp.sort();member=set(comp);outside={end[v] for v in workers if v not in member}
        legal=[]
        for i,u in enumerate(comp):
            rr={z:(score+1.e-6*(a==idx[u]),a) for z,(score,a) in rows[u].items() if z not in outside}
            rr[225+i]=(-10000.,-1);legal.append(rr)
        conflicts+=1;matched=assign(legal)
        if any(z is None or z>=225 or z not in row for z,row in zip(matched,legal)):continue
        changed={u:z for u,z in zip(comp,matched) if z!=end[u]};seen2=set()
        for first in sorted(changed):
            if first in seen2:continue
            linked=[first];seen2.add(first)
            for u in linked:
                for v in sorted(changed.keys()-seen2):
                    if changed[v]==end[u] or changed[u]==end[v]:linked.append(v);seen2.add(v)
            if len(linked)<2 or not seeded.intersection(linked):continue
            gain=sum(rows[u][changed[u]][0]-float(logits[workers[u][0],idx[u]]) for u in linked)
            if gain<min_gain:continue
            new=dict(base);edits=[]
            for u in sorted(linked):
                a=rows[u][changed[u]][1];edits.append([u,base.get(u),ACT[a],workers[u][2]])
                if a:new[u]=ACT[a]
                else:new.pop(u,None)
            offers.append((gain,new,dict(gain=round(gain,4),component=comp,edits=edits,
                 type='shared-first' if len({best[u] for u in linked})<len(linked) else 'yield-to-occupied-destination')))
    if not offers:return base,None,conflicts
    _,new,meta=max(offers,key=lambda row:row[0]);return new,meta,conflicts
