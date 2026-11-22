"""Approximate what an opponent knows using its publicly observed worker positions."""
import argparse
import inspect
from pathlib import Path


def _r6_opponent_view(state, players, step, side, estimated, previous):
    if step == 0 or 'shadow_visibility' not in state:
        state['shadow_visibility'] = BeliefTracker(_Bel,side)
    tracker = state['shadow_visibility']
    if previous is not None:
        # Surviving workers reveal their last move. A lost worker's destination stays unknown.
        movement = {}
        for uid,(pos,_,_) in previous[side][2].items():
            now=players[side][2].get(uid)
            if now is not None and now[0] in _ROLLOUT_LANDING[pos]:
                idx=_ROLLOUT_LANDING[pos].index(now[0])
                if idx:movement[uid]=WACT[idx]
        tracker.set_actions(previous,movement)
    _,rival_known=tracker.step(players,step,estimated)
    # BeliefTracker can treat a guessed sensor value as exact forever. Preserve
    # the current carbon estimate and use the second tracker only for visibility.
    return _quant(encode(players,step,side,estimated,rival_known,previous))


def build(code):
    head='def _v5_root(obs,'
    spot='    opponent_x = _quant(encode(players,step,op,est,[1.]*N,previous))'
    if code.count(head)!=1 or code.count(spot)!=1:
        raise ValueError('expected the rival view at the root')
    code=code.replace(head,inspect.getsource(_r6_opponent_view)+'\n\n'+head)
    code=code.replace(spot,'    opponent_x = _r6_opponent_view(state,players,step,op,est,previous)')
    compile(code,'<opponent-view>','exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    with a.output.open('x') as f:f.write(build(a.parent.read_text()))
