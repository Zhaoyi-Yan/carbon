"""Price access to the next hire and avoid buying a planter that cannot finance a tree."""
import argparse
import inspect
from pathlib import Path


def _v7_tree_float(pl, meta, horizon):
    """Credit only the short income stream before the tree expires."""
    return sum(max(0.,min(horizon,50-age))*max(0.,meta.get(uid,(None,0.))[1])
               for uid,(_,age) in pl[3].items())


def _v7_keep_float(obs, config, action, drought):
    side=obs['player'];us,them=obs['players'][side],obs['players'][1-side]
    fee=config.get('recPlanterCost',30);seize=config.get('seizeCost',20)
    if obs['step']<25 or drought<10 or len(us[2])<2 or us[0]>=2*fee+seize:return action
    bids=[bid for bid in us[1] if action.get(bid)=='RECPLANTER']
    if not bids:return action
    release=sum(config.get('recCollectorCost',30) if action[b]=='RECCOLLECTOR' else fee for b in us[1] if b in action)
    if us[0]-release>=seize:return action
    own_receipts=_v7_receipts(obs['players'],obs['step'],LAST_TURN)[side,0]
    inflow=_v7_tree_float(us,obs.get('trees') or {},5)+.30*float(own_receipts)
    if us[0]-release+inflow>=seize:return action
    if len(us[3])>2 or len(them[2])<len(us[2])+2:return action
    out=dict(action)
    for bid in bids:
        out.pop(bid)
        release-=fee
        if us[0]-release+inflow>=seize:break
    return out


def build(code):
    marker='def _act(obs, configuration=None, t0=None):'
    assert code.count(marker)==1
    func='\n\n'.join(inspect.getsource(f) for f in (_v7_tree_float,_v7_keep_float))
    code=code.replace(marker,func+'\n\n'+marker)
    marker="        st['liquidity_action']=dict(act)"
    assert code.count(marker)==1
    code=code.replace(marker,"        act=_v7_keep_float(obs,configuration,act,drought)\n"+marker)
    marker='        valuations.append(p[0]+c+tree+horizon*workforce)'
    assert code.count(marker)==1
    return code.replace(marker,'''        # With a depleted roster, an unavailable next hire can cost several future turns.
        deficit=max(0.,min(1.,(len(players[1-me][2])-len(p[2]))/4.))
        available=p[0]+.45*float(five)+_v7_tree_float(p,meta,3)
        float_cost=1.75*max(0.,65.-available)*deficit*horizon
        valuations.append(p[0]+c+tree+horizon*workforce-float_cost)''')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    with a.output.open('x') as f:f.write(build(a.parent.read_text()))
