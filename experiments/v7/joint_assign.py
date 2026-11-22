"""Propose one conflict-connected reassignment using the frozen policy's Hungarian solver."""
from collections import Counter
import numpy as np
from mine_bank import L, ACT


def propose(obs,base,logits,locked,assign_moves,min_gain=.75):
    side=obs['player'];own=obs['players'][side];workers=own[2]
    if len(workers)<2:return base,None,0
    old={u:ACT.index(base.get(u)) for u in workers}
    ends={u:int(L[v[0],old[u]]) for u,v in workers.items()}
    counts=Counter(ends.values());homes=set(own[1].values())
    protect=set(locked)
    for u,(q,c,kind) in workers.items():
        # Every deliberate shared destination, including simultaneous depot banking, is reserved.
        if counts[ends[u]]>1 or ends[u] in homes and c>0:protect.add(u)
    fixed_tiles={ends[u] for u in protect if u in workers}|{p for b,p in own[1].items() if b in base}
    free=[u for u in workers if u not in protect]
    if len(free)<2:return base,None,0
    preferred={}
    for u in free:
        q=workers[u][0];order=np.argsort(-logits[q],kind='stable')
        preferred[u]=next((int(L[q,a]) for a in order if int(L[q,a]) not in fixed_tiles),ends[u])
    adj={u:set() for u in free}
    for i,u in enumerate(free):
        for v in free[:i]:
            if preferred[u]==preferred[v] or preferred[u]==ends[v] or preferred[v]==ends[u]:
                adj[u].add(v);adj[v].add(u)
    seen=set();best=None;conflicts=0
    for u0 in free:
        if u0 in seen or not adj[u0]:continue
        comp=[];stack=[u0];seen.add(u0)
        while stack:
            u=stack.pop();comp.append(u)
            for v in sorted(adj[u]):
                if v not in seen:stack.append(v);seen.add(v)
        comp.sort();members=set(comp);blocked=fixed_tiles|{q for uid,q in ends.items() if uid not in members};rows=[]
        conflicts+=1
        for i,u in enumerate(comp):
            q=workers[u][0]
            row={int(z):(float(logits[q,a])+1.e-6*(a==old[u]),a)
                 for a,z in enumerate(L[q]) if int(z) not in blocked and np.isfinite(logits[q,a])}
            # Each row gets its own overflow destination. Reject any matching that uses one.
            row[225+i]=(-10000.,-1);rows.append(row)
        targets=assign_moves(rows)
        if any(z is None or z>=225 or z not in row for z,row in zip(targets,rows)):continue
        edits=[(u,rows[i][z][1]) for i,(u,z) in enumerate(zip(comp,targets)) if rows[i][z][1]!=old[u]]
        if len(edits)<2:continue
        changed={u for u,_ in edits}
        if not any(ends[v]==int(L[workers[u][0],a]) for u,a in edits for v in changed if u!=v):continue
        gain=sum(float(logits[workers[u][0],a])-float(logits[workers[u][0],old[u]]) for u,a in edits)
        if gain<min_gain or best is not None and gain<=best[0]:continue
        best=(gain,comp,edits,len({preferred[u] for u in comp})<len(comp))
    if best is None:return base,None,conflicts
    gain,comp,edits,shared=best;actions=dict(base)
    for u,a in edits:
        if a:actions[u]=ACT[a]
        else:actions.pop(u,None)
    meta=dict(gain=round(gain,4),component=comp,edits=[[u,base.get(u),ACT[a],workers[u][2]] for u,a in edits],
              type='shared-first' if shared else 'yield-to-occupied-destination')
    return actions,meta,conflicts
