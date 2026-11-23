"""Build v7 challengers with independently selectable team decisions."""
import argparse
import hashlib
import inspect
from pathlib import Path
from economy import BASELINE

ARMS=('opening','convoy','intercept','closing','combined')


def _team_proposal(obs,cfg,stock,base,logits,locks):
    side=obs['player']
    us,foe=obs['players'][side],obs['players'][1-side]
    ports=tuple(us[1].values())
    enemy_ports=tuple(foe[1].values())
    if not ports or not enemy_ports:
        return None
    turn=obs['step']
    remaining=cfg['episodeSteps']-1-turn
    closing='closing' in _TEAM and turn>=190
    miners={u:(p,v) for u,(p,v,k) in us[2].items() if k=='COLLECTOR'}
    rivals=[(p,v) for p,v,k in foe[2].values() if k=='COLLECTOR']
    homes=_fc_home(tuple(sorted(ports)))
    hires={p for u,p in us[1].items() if base.get(u)}
    woods={p for pl in (us,foe) for p,age in pl[3].values() if age<cfg['treeLifespan']}
    ends={u:move(p,WIDX.get(base.get(u),0)) for u,(p,_,_) in us[2].items()}
    proposals=[]

    def safe(q,load):
        return q not in enemy_ports and all(_V5_DIST[q][z]>1 or v>load for z,v in rivals)

    def offer(edits,value):
        edits={u:a for u,a in edits.items() if a!=WIDX.get(base.get(u),0)}
        if not edits or any(u in locks for u in edits):
            return
        reserved=hires | {q for u,q in ends.items() if u not in edits}
        out=dict(base)
        for u,a in edits.items():
            p=us[2][u][0]
            q=move(p,a)
            if q in reserved:
                return
            reserved.add(q)
            if a:
                out[u]=WACT[a]
            else:
                out.pop(u,None)
            value-=2*max(0.,float(logits[p,WIDX.get(base.get(u),0)]-logits[p,a]))
        if value>0:
            proposals.append((value,out))

    if 'opening' in _TEAM and turn<125 and len(miners)>=2:
        # Give each depot a local collector. Other collectors can range further.
        assigned={}
        pairs=sorted((_V5_DIST[p][b],-v,u,b) for b in ports for u,(p,v) in miners.items())
        for _,_,u,b in pairs:
            if u not in assigned and b not in assigned.values():
                assigned[u]=b
        targets=set()
        edits={}
        for u,b in assigned.items():
            p,v=miners[u]
            if u in locks:
                continue
            if v>=max(32,min(85,us[0]*.35)):
                goal=b
            else:
                sites=[q for q in range(225) if _V5_DIST[b][q]<=3 and _V5_DIST[p][q]<=4
                       and q not in woods and q not in ports and q not in targets and stock[q]>8 and safe(q,v)]
                if not sites:
                    continue
                goal=max(sites,key=lambda q:stock[q]/(2+_V5_DIST[p][q]+_V5_DIST[b][q]))
                targets.add(goal)
            steps=[a for a,q in enumerate((p,*NB[p])) if
                   (safe(q,v) or q in ports) and (q==goal or _V5_DIST[q][goal]<_V5_DIST[p][goal])]
            if steps:
                edits[u]=max(steps,key=lambda a:logits[p,a])
        offer(edits,30.)

    if 'convoy' in _TEAM or closing:
        for u,(p,v) in miners.items():
            if v<55 or homes[p]>min(remaining,6) or u in locks:
                continue
            for a,q in enumerate(NB[p],1):
                if homes[q]>=homes[p] or not (safe(q,v) or q in ports):
                    continue
                if closing:
                    offer({u:a},v/(1+homes[p])*.65)
                if 'convoy' not in _TEAM:
                    continue
                for h,(z,light) in miners.items():
                    if u==h or h in locks or light>35 or _V5_DIST[z][q]>2:
                        continue
                    blocking=ends[h]==q
                    threats=[e for e,c in rivals if light<c<=v and _V5_DIST[e][q]<=3]
                    for k,dst in enumerate((z,*NB[z])):
                        if dst==q or not safe(dst,light):
                            continue
                        shielding=any(_V5_DIST[dst][e]<=1 and _V5_DIST[ends[h]][e]>1 for e in threats)
                        if blocking or shielding:
                            offer({u:a,h:k},v*(.48 if blocking else .32)/(1+.22*homes[p])-.16*light)

    if 'intercept' in _TEAM or closing:
        for z,prize in rivals:
            shortest=min(_V5_DIST[z][b] for b in enemy_ports)
            if prize<65 or not 1<shortest<=min(7,remaining):
                continue
            # Count all shortest paths to both reachable depots; discount a readily available detour.
            paths={z:1.}
            for tick in range(1,min(shortest,4)):
                nxt={}
                for cell,weight in paths.items():
                    forward=[n for n in NB[cell] if min(_V5_DIST[n][b] for b in enemy_ports)==shortest-tick]
                    for n in forward:
                        nxt[n]=nxt.get(n,0.)+weight/len(forward)
                paths=nxt
                for q,chance in paths.items():
                    if chance<.25:
                        continue
                    alt=min((_V5_DIST[z][b] for b in enemy_ports if _V5_DIST[z][q]+_V5_DIST[q][b]>_V5_DIST[z][b]),default=99)
                    denied=prize*chance*(.25 if alt<=shortest+1 else .65)
                    for u,(p,v) in miners.items():
                        travel=_V5_DIST[p][q]
                        if travel>tick or v>35 or v>=prize:
                            continue
                        steps=[a for a,n in enumerate((p,*NB[p])) if (n==q or _V5_DIST[n][q]<travel) and safe(n,v)]
                        if steps:
                            offer({u:max(steps,key=lambda a:logits[p,a])},denied/(1+.3*tick)-.45*v-4*tick)
    if closing:
        # Value a capture against both players' remaining receipts. Guard only a threatened tree.
        thieves=[p for p,_,k in foe[2].values() if k=='PLANTER']
        for owner,pl in enumerate(obs['players']):
            for tid,(q,age) in pl[3].items():
                life=min(remaining,cfg['treeLifespan']-age)
                if life<2:
                    continue
                income=obs['trees'].get(tid,(None,0))[1]
                for u,(p,load,role) in us[2].items():
                    d=_V5_DIST[p][q]
                    if d>min(3,life-1):
                        continue
                    if owner==side:
                        if role!='COLLECTOR' or load>30 or not any(_V5_DIST[z][q]<=d+2 for z in thieves):
                            continue
                        value=income*min(10,life-d)*.65-.5*load
                    else:
                        if role!='PLANTER' or us[0]<cfg['seizeCost']:
                            continue
                        value=1.6*income*min(15,life-d)-cfg['seizeCost']
                    steps=[a for a,n in enumerate((p,*NB[p])) if (n==q or _V5_DIST[n][q]<d) and
                           safe(n,load if role=='COLLECTOR' else float('inf'))]
                    if steps:
                        offer({u:max(steps,key=lambda a:logits[p,a])},value/(1+.35*d))
    return max(proposals,key=lambda pair:pair[0])[1] if proposals else None


def build(source,arm):
    if hashlib.sha256(source.encode()).hexdigest()!=BASELINE or arm not in ARMS:
        raise ValueError('Expected frozen v7 and a known arm')
    modes=ARMS[:-1] if arm=='combined' else (arm,)
    anchor='def _act(obs, configuration=None, t0=None):'
    point='    if most>=3:\n        tree_action=_r6_first_tree'
    assert source.count(point)==source.count(anchor)==1
    extra="""    if most>=4 and len(cands)<most:
        option=_team_proposal(obs,configuration,est,act0,lw0,fixed or {})
        if option is not None:
            marker=tuple(sorted(option.items()))
            if marker not in keys:
                cands.append(option)
                keys.add(marker)
"""
    return source.replace(anchor,f'_TEAM={modes!r}\n'+inspect.getsource(_team_proposal)+'\n\n'+anchor).replace(point,extra+point)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm',choices=ARMS)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    args.output.write_text(build((root/'yan-v7.py').read_text(),args.arm))
