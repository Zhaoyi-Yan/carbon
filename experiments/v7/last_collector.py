"""Keep the last source of working capital alive when a planter cannot afford any tree."""
import inspect


def _v7_solvency(obs, config, action, worker_logits):
    us,them=obs['players'][obs['player']],obs['players'][1-obs['player']]
    hire=config.get('recCollectorCost',30);seize=config.get('seizeCost',20)
    if obs['step']<=1 or us[0]>=hire+max(seize,config.get('recPlanterCost',30)):return action
    if _v7_tree_float(us,obs.get('trees') or {},6)>=hire:return action
    carriers=[(u,p,c) for u,(p,c,k) in us[2].items() if k=='COLLECTOR']
    if len(carriers)>1:return action
    homes=set(us[1].values());foe_homes=set(them[1].values())
    out=dict(action)
    # A planter without the price of a first tree cannot restore income by moving.
    reserve=us[0]-sum(config.get('recPlanterCost',30) if out[b]=='RECPLANTER' else hire for b in us[1] if b in out)
    if reserve+_v7_tree_float(us,obs.get('trees') or {},4)<min(seize,config.get('plantCost',20)):
        for bid in us[1]:
            if out.get(bid)=='RECPLANTER':out.pop(bid);reserve+=config.get('recPlanterCost',30)
    if len(carriers)!=1 or reserve>=hire or any(out.get(b)=='RECCOLLECTOR' for b in us[1]):return out
    uid,pos,load=carriers[0];orig=WIDX.get(out.get(uid),0);nodes=_V7_L[pos]
    enemies=[q for q,c,k in them[2].values() if k=='COLLECTOR' and c<=load]
    blockers={p for b,p in us[1].items() if b in out}
    for other,(p,_,_) in us[2].items():
        if other!=uid:blockers.add(int(_V7_L[p,WIDX.get(out.get(other),0)]))
    def safe(q):
        if q in blockers or q in foe_homes:return False
        if q in homes and reserve+load>=hire:return True
        return all(_V7_D[z,q]>1 for z in enemies)
    if safe(nodes[orig]):return out
    choices=[a for a,q in enumerate(nodes) if safe(q)]
    if not choices:return out
    ranked=_probs(worker_logits[pos],.87)
    new=max(choices,key=lambda a:float(ranked[a])+.16*(nodes[a] in homes and load>0))
    if new:out[uid]=WACT[new]
    else:out.pop(uid,None)
    return out


def build(code):
    marker='def _act(obs, configuration=None, t0=None):'
    check="        st['liquidity_action']=dict(act)"
    payoff='        valuations.append(p[0]+c+tree+horizon*workforce-float_cost)'
    assert all(code.count(x)==1 for x in (marker,check,payoff))
    code=code.replace(marker,inspect.getsource(_v7_solvency)+'\n\n'+marker)
    code=code.replace(check,'        act=_v7_solvency(obs,configuration,act,lw)\n'+check)
    return code.replace(payoff,'''        # A surviving planter without planting money earns nothing by itself.
        if not collectors and not p[3] and p[0]<20.:
            float_cost+=horizon*600.
        valuations.append(p[0]+c+tree+horizon*workforce-float_cost)''')
