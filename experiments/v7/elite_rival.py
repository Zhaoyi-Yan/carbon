"""Optional v6 patch: compare three predictions using only observed rival movements."""
import inspect
from pathlib import Path

import numpy as np


def _v7_rival_logprobs(rows):
    z = rows - rows.max(axis=-1, keepdims=True)
    return np.maximum(-7., z - np.log(np.exp(z).sum(axis=-1, keepdims=True)))


def _v7_route_rival(state, players, side, step, other_view):
    foe = players[1-side]
    memory = state.get('elite_rival')
    if memory is None or memory['step'] != step-1:
        memory = dict(step=-1, total=np.zeros(2), count=0., pending={}, selected=0)
        state['elite_rival'] = memory
    past = []
    for uid,(start,logprob) in memory['pending'].items():
        worker = foe[2].get(uid)
        if worker and worker[0] in _ROLLOUT_LANDING[start]:
            action = _ROLLOUT_LANDING[start].index(worker[0])
            past.append(np.clip(logprob[1:,action] - logprob[0,action], -1.5, 1.5))
    memory['total'] *= .88
    memory['count'] *= .88
    if past:
        # One crowded turn cannot singlehandedly determine the opponent.
        weight = min(5, len(past))
        memory['total'] += weight * np.mean(past, axis=0)
        memory['count'] += weight
    baseline = state['opponent']
    memory['legacy_id'] = state['simulation_ids'][1-side]
    memory['step'] = step
    memory['pending'] = {}
    memory['blend'] = 0.
    if not foe[2]:
        memory['selected'] = 0
        return
    # The same legal estimate of opponent fog as the baseline; no rival actions or private carbon.
    # Average opposite rotations so a coordinate preference cannot decide the identity.
    views=[]; rotations=(_V5_ROTATIONS[0], _V5_ROTATIONS[2])
    for _,inverse,_,channels in rotations:
        view=other_view.reshape(NPL,N)[channels][:,inverse].reshape(NPL,SIZE,SIZE)
        views.extend((view,view))
    raw_w,raw_b,_ = _net()(np.stack(views).astype(np.float32), np.array([21,22,21,22],dtype=np.int64))
    workers=np.zeros((2,N,5),dtype=np.float32);bases=np.zeros((2,N,3),dtype=np.float32)
    for j,(cells,_,actions,_) in enumerate(rotations):
        workers+=.5*raw_w[2*j:2*j+2][:,cells][:,:,actions]
        bases+=.5*raw_b[2*j:2*j+2][:,cells]
    for uid,(cell,_,_) in foe[2].items():
        candidates=np.stack([baseline[0][cell],workers[0,cell],workers[1,cell]])
        memory['pending'][uid]=(cell,_v7_rival_logprobs(candidates))
    advantage=memory['total']/max(1.,memory['count'])
    previous=memory['selected']
    best=int(np.argmax(advantage))
    if previous and advantage[previous-21] >= advantage[best]-.025:
        best=previous-21
    gain=float(advantage[best])
    if memory['count'] < 19 or gain < (.008 if previous else .028):
        memory['selected']=0
        return
    style=21+best
    blend=min(.72,.30+2.4*(gain-.008))
    memory.update(selected=style,blend=blend)
    state['opponent'] = ((1.-blend)*baseline[0]+blend*workers[best],
                         (1.-blend)*baseline[1]+blend*bases[best])
    state['root_ids'][1] = style
    state['simulation_ids'][1-side] = style


def build(code):
    marker='def _v5_root(obs, est, x, previous):'
    before="    _shadow_update(obs,x,opponent_x,state,mine,econ,styles,style,rival)"
    roll="    _v5_rollout_set(me,style,rival)"
    tiny="sw, sb, sv = _V6R_TINY(np.ascontiguousarray(enemy), _v5_ids([side]*n), enemy_scalar)"
    for anchor in (marker,before,roll,tiny):
        if code.count(anchor)!=1:
            raise ValueError('Missing unique v6 anchor: '+anchor)
    code=code.replace(marker,'\n\n'.join(inspect.getsource(f) for f in (_v7_rival_logprobs,_v7_route_rival))+'\n\n'+marker)
    code=code.replace(before,before+"\n    _v7_route_rival(state,players,me,step,opponent_x)")
    code=code.replace(roll,roll+"""
    elite_mix = state['elite_rival']['blend']
    if elite_mix:
        w,b = state['rollout'][op]
        state['rollout'][op] = (w*(1.-elite_mix), b*(1.-elite_mix))""")
    # The student has twenty teachers and one generic row. Preserve its previous routing on every branch.
    code=code.replace(tiny,"""tiny_id = _ST['v5_history']['elite_rival']['legacy_id']
    sw, sb, sv = _V6R_TINY(np.ascontiguousarray(enemy), tiny_id, enemy_scalar)""")
    return code


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.write_text(build(args.source.read_text()))
