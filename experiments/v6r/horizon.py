"""Verify the two surviving joint moves with a longer cash-flow rollout."""
import argparse
import inspect
from pathlib import Path


def _r6_tail(boards, candidates, me, config, t0, seed, allowance):
    states = {b[0]:[b[0], b[1], b[2], b[3]] for b in boards if b[0] in candidates and b[1]==0}
    if len(states)!=2 or len({b[2]['step'] for b in states.values()})!=1:
        return None
    pair=[states[c] for c in candidates]
    completed=0
    per_step=.016
    for tick in range(1,15):
        start=_time.perf_counter()
        if start-t0+max(.022,1.45*per_step) > allowance or pair[0][2]['step']>LAST_TURN:
            break
        views=[]
        for _,_,board,prev in pair:
            if 'v5_views' not in board:
                board['v5_views']=_v5_pair(board['players'],board['step'],board['carbon'],prev)
            views.extend(board['v5_views'])
        # Both continuations see the same random choices. This preserves differences
        # caused by the original root move several turns down the line.
        w,b,_=_V6R_TINY(np.asarray(views),0)
        for j,state in enumerate(pair):
            board=state[2];pl=board['players'];t=board['step']
            moves=[_sample(pl,s,w[2*j+s],b[2*j+s],np.random.RandomState(seed+373*t+17*s),t) for s in (0,1)]
            state[3],state[2]=pl,_turn(board,config,moves)
        completed=tick
        per_step=.55*per_step+.45*(_time.perf_counter()-start)
    if completed<3:
        return None
    values=[]
    for _,_,board,_ in pair:
        if board['step']>LAST_TURN:
            cash=[p[0] for p in board['players']]
            values.append(_terminal(cash[me]-cash[1-me]))
        else:
            values.append(_r6_asset_score(board,me))
    return values


def _r6_second_style(x, players, side, expert):
    # Two rotations per alternative keep the root inside the same turn budget.
    batch=[];selected=(_V5_ROTATIONS[0],_V5_ROTATIONS[2])
    for cells,inverse,actions,channels in selected:
        batch.append(x.reshape(NPL,N)[channels][:,inverse].reshape(NPL,SIZE,SIZE))
    lw,lb,_=_net()(np.asarray(batch,dtype=np.float32),expert)
    w=np.zeros_like(lw[0]);b=np.zeros_like(lb[0])
    for i,(cells,_,actions,_) in enumerate(selected):
        w+=.5*lw[i,cells][:,actions];b+=.5*lb[i,cells]
    return w,b


def build(code):
    def change(old,new):
        nonlocal code
        if code.count(old)!=1: raise ValueError(old)
        code=code.replace(old,new)
    for dependency in ('_R6_TREE =','_V6R_TINY =','expert ='):
        if dependency not in code:raise ValueError(dependency)
    change('def _search(obs,',inspect.getsource(_r6_tail)+'\n\n'+inspect.getsource(_r6_second_style)+'\n\ndef _search(obs,')
    change("    state['opponent'] = _v5_mix", """    state['alternatives'].insert(0, _r6_second_style(x, players, me, 43-expert))
    state['opponent'] = _v5_mix""")
    change("    if most >= 4:\n        for aw, ab in _ST['v5_history']['alternatives'][:min(2,(most-1)//3)]:","    if most >= 3:\n        for aw, ab in _ST['v5_history']['alternatives'][:min(2,max(1,(most-1)//3))]:")
    change('_S_ABORT = 0.78','_S_ABORT = 0.66')
    anchor='    best = max(alive, key=lambda c: mean[c])'
    change(anchor,"""    if depth >= 2 and len(alive) >= 2 and _time.perf_counter()-t0 < .69:
        two=sorted(alive,key=lambda c:-mean[c])[:2]
        tail=_r6_tail(boards,two,me,configuration,t0,seed0,.83)
        if tail is not None:
            # Common offset is immaterial. Only compare branches both rolled
            # to exactly the same future step.
            shared=.5*(sum(mean[c] for c in two)-sum(tail))
            for c, v in zip(two,tail):
                mean[c]=.52*mean[c]+.48*(v+shared)
            alive=two
"""+anchor)
    compile(code,'<horizon>','exec')
    return code

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(build(a.source.read_text()))
