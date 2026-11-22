"""Compose elite proposals, a balanced first cohort, and a replay-priced economic critic."""
import argparse
import json
from pathlib import Path

from export import build, original


def _r6_opening(obs, config, lb, action):
    step, side = obs['step'], obs['player']
    if step > 2 or config is None:
        return action
    own = obs['players'][side]
    if len(own[1]) != 2 or own[2] or own[0] < config.get('recCollectorCost',30)+config.get('recPlanterCost',30):
        return action
    if step == 0 and _R6_OPENING in ('observe','react'):
        return {}
    bases = list(own[1].items())
    (_, a), (_, b) = bases
    # Compare the joint role assignments after observing the rival's recruits.
    diff = lb[a,2] + lb[b,1] - lb[a,1] - lb[b,2]
    planter = 0 if diff >= 0 else 1
    if _R6_OPENING == 'react':
        enemy = list(obs['players'][1-side][2].values())
        if len(enemy)==2 and all(role=='COLLECTOR' for _,_,role in enemy):
            planter = int(b>a)
        elif len(enemy)==2 and side==0:
            target = next((pos for pos,_,role in enemy if role=='PLANTER'),None)
            if target is not None:
                planter = int(b//15 == target//15)
    return {bid: BACT[2 if k == planter else 1] for k,(bid,_) in enumerate(bases)}


def compose(code,opening,assets):
    import inspect
    def change(old,new):
        nonlocal code
        if code.count(old)!=1: raise ValueError(old)
        code=code.replace(old,new)
    if opening!='native':
        helper='_R6_OPENING = '+repr(opening)+'\n'+inspect.getsource(_r6_opening)+'\n'
        change('def _act(obs, configuration=None, t0=None):',helper+'\ndef _act(obs, configuration=None, t0=None):')
        change("    st['tr'].set_actions(players, act)","    act = _r6_opening(obs, configuration, lb, act)\n    st['tr'].set_actions(players, act)")
    if assets:
        folder=Path(__file__).parent
        prices=json.loads((folder/'assets-model.json').read_text())
        fitted=(folder/'assets.py').read_text()
        for name in ('cargo','tree'):
            fitted+='\n_R6_'+name.upper()+' = np.array('+repr(prices[name]['coef'])+', dtype=float)\n'
        change('def _search(obs,',fitted+'\n\ndef _search(obs,')
        anchor='                    value += _V5_VALUE_MIX * (_v5_value(future, me) - value)'
        change(anchor,anchor+'\n                    value += '+str(assets)+' * (_r6_asset_score(future, me) - value)')
    # If initialization or one pass hit transient CPU contention, avoid permanently
    # rejecting every candidate based on a stale cost. Measure a minimal real batch.
    anchor='        if late((2 if both else 1) * n, n if both else 0):'
    change(anchor,anchor[:-1]+' and (t > 1 or _time.perf_counter()-t0 > .27 or len(live)>4):')
    compile(code,'<v6-candidate>','exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--source',type=Path)
    p.add_argument('--weights',type=Path)
    p.add_argument('--mode',choices=['duo','artemis','apollo'],default='duo')
    p.add_argument('--opening',choices=['native','observe','immediate','react'],default='observe')
    p.add_argument('--assets',type=float,default=.26)
    a=p.parse_args()
    code=a.source.read_text() if a.source else original()
    if a.weights: code=build(a.weights,a.mode,code)
    a.output.write_text(compose(code,a.opening,a.assets))
