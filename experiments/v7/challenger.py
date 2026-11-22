"""Assemble the single-file v7 policy from the frozen v6 and the fitted deposit forecaster."""
import argparse
import hashlib
import inspect
from pathlib import Path

import lanes
import receipts
import shipping_styles

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE='e5c328aa5d18ced597791e7b839088f1a37d18e4e8dc83dd3d54b13d66e877b0'


def _v7_flows(board):
    if '_v7_flow' not in board:
        board['_v7_flow']=_v7_receipts(board['players'],board['step'],LAST_TURN)
    return board['_v7_flow']


def _v7_modes(obs,flow):
    me=obs['player'];own,rival=obs['players'][me],obs['players'][1-me]
    ours,theirs=flow[me],flow[1-me]
    left=LAST_TURN+1-obs['step']
    if obs['step']<70 or left<2:return ()
    if max(ours[1],theirs[1])<45.:return ('tree',) if left>25 else ()
    cargo_edge=.45*(theirs[0]-ours[0])+.55*(theirs[1]-ours[1])
    margin=own[0]-rival[0]
    urgency=1. if left<40 else .65
    bank=(ours[1]+.28*max(0.,-margin)*urgency)*(1. if len(own[2])>2 else .60)
    hunt=theirs[1]+.7*cargo_edge+.6*max(0.,min(200.,margin)-max(0.,margin-cargo_edge))
    forest=55.+max(0.,len(rival[3])-len(own[3]))*18. if left>22 else -1.
    priorities={'bank':bank,'intercept':hunt,'tree':forest}
    priorities[_V7_PREFERRED]*=_V7_BIAS
    return sorted(priorities,key=priorities.get,reverse=True)


def _v7_proposals(obs,cfg,stock,base,locked,flow,limit):
    options=[];seen={tuple(sorted(base.items()))}
    for mode in _v7_modes(obs,flow):
        act=_v7_plan(obs,cfg,stock,base,mode)
        if any(act.get(uid)!=base.get(uid) for uid in locked):continue
        key=tuple(sorted(act.items()))
        if key not in seen:
            seen.add(key);options.append((mode,act))
            if len(options)>=limit:break
    return options


def build(source, preferred='bank', bias=1.24, asset=.56, weights=None, persistent=True):
    if hashlib.sha256(source).hexdigest()!=BASE:raise ValueError('Expected frozen v6')
    code=receipts.build(source.decode())
    if weights:code=shipping_styles.apply(code,weights)
    def change(old,new):
        nonlocal code
        if code.count(old)!=1:raise ValueError(old)
        code=code.replace(old,new)
    marker='def _act(obs, configuration=None, t0=None):'
    core=Path(lanes.__file__).read_text().split('_V7_ACTIONS =',1)[1].split('\n\nplan =',1)[0]
    functions='\n\n'.join(inspect.getsource(f) for f in (_v7_flows,_v7_modes,_v7_proposals))
    change(marker,"_V7_PREFERRED = "+repr(preferred)+"\n_V7_BIAS = "+repr(bias)+"\n_V7_ASSET = "+repr(asset)+"\n_V7_ACTIONS ="+core+'\n\n'+functions+'\n\n'+marker)
    marker="        tree=_sapling_total(board,meta,me,woods,sap_context,LAST_TURN)"
    change(marker,"""        # Existing cargo already has a value. Replace that part of the estimate
        # with the two expected bank receipts, avoiding double counting it.
        five,ten=_v7_flows(board)[me]
        mix=min(.90,_V7_ASSET+.28*max(0.,1.-(LAST_TURN+1-t)/19.))
        c=(1.-mix)*c + mix*(.28*five+.72*ten)
"""+marker)
    marker='    if most>=3:\n        tree_action='
    change(marker,"""    root_modes={}
    if most>=3:
        limit=1 if most<=6 else 2
        alternatives=_v7_proposals(obs,configuration,est,act0,fixed or {},_v7_flows(root),limit)
        for mode,action in alternatives[:limit]:
            key=tuple(sorted(action.items()))
            if key not in keys:
                root_modes[len(cands)]=mode;cands.append(action);keys.add(key)
"""+marker)
    change('if tree_key not in keys:', 'if len(cands)<most and tree_key not in keys:')
    # Early safety locks can filter candidates and change their numerical indices.
    marker='    if protected:\n        cands=[a for a in cands'
    change(marker,"    mode_keys={tuple(sorted(cands[i].items())):mode for i,mode in root_modes.items()}\n"+marker)
    marker="    # the opponent's view at the root, for its first move"
    change(marker,"    root_modes={i:mode_keys[tuple(sorted(a.items()))] for i,a in enumerate(cands) if tuple(sorted(a.items())) in mode_keys}\n"+marker)
    if persistent:
        marker="            ao = _sample(o['players'], op, lwp, lbp, r2, o['step'])"
        change(marker,"""            mode=root_modes.get(b[0])
            if mode and t<=2 and _time.perf_counter()-t0<.48:
                sim=o.get('_st')
                local={'players':o['players'],'step':o['step'],'trees':sim.trees_meta if sim is not None else o.get('trees',{}),'player':me}
                am=_v7_plan(local,configuration,o['carbon'],am,mode)
"""+marker)
    code=code.replace('yan-v6 research challenger.','yan-v7 research challenger. Forecasts near-term bank receipts for both sides.')
    compile(code,'<yan-v7>','exec')
    if len(code.encode())>10_000_000:raise ValueError('competition size limit exceeded')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--preferred',choices=('tree','bank','intercept'),default='bank');p.add_argument('--bias',type=float,default=1.24)
    p.add_argument('--asset',type=float,default=.56);p.add_argument('--weights',type=Path);p.add_argument('--no-persistence',action='store_true')
    a=p.parse_args()
    with a.output.open('x') as f:f.write(build((ROOT/'yan-v6.py').read_bytes(),a.preferred,a.bias,a.asset,a.weights,not a.no_persistence))
