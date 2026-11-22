"""Take the first-turn mixed opening as selected on the development maps."""
import inspect


def _v7_early_income(obs, config, base_logits, action):
    s=obs['player'];pl=obs['players'][s]
    if obs['step']!=0 or len(pl[1])!=2 or pl[2]:return action
    if pl[0]<config.get('recCollectorCost',30)+config.get('recPlanterCost',30):return action
    bases=list(pl[1].items())
    (_,p),(_,q)=bases
    low=base_logits[p,2]+base_logits[q,1]
    high=base_logits[p,1]+base_logits[q,2]
    planter=0 if low>=high else 1
    return {uid:BACT[2 if i==planter else 1] for i,(uid,_) in enumerate(bases)}


def build(code):
    marker='def _act(obs, configuration=None, t0=None):'
    spot='    act = _r6_opening(obs, configuration, lb, act)'
    assert code.count(marker)==code.count(spot)==1
    code=code.replace(marker,inspect.getsource(_v7_early_income)+'\n\n'+marker)
    return code.replace(spot,spot+'\n    if configuration is not None:act=_v7_early_income(obs,configuration,lb,act)')
