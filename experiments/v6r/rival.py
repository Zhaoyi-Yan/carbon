"""Recognize v5-like board behaviour and condition the forward model on its searched moves."""
import argparse
import base64
import inspect
from pathlib import Path


def _r6_rival_gate(state, players, side, step):
    enemy=players[side]
    if step==1:
        kinds=[u[2] for u in enemy[2].values()]
        state['rival_prior']=1.35 if kinds==['COLLECTOR','COLLECTOR'] else -2.25
    evidence=state.get('rival_evidence',0.)*.95;observed=[]
    for uid,(pos,advantage) in state.pop('rival_pending',{}).items():
        if uid in enemy[2] and enemy[2][uid][0] in _ROLLOUT_LANDING[pos]:
            dest=_ROLLOUT_LANDING[pos].index(enemy[2][uid][0]);observed.append(advantage[dest])
    if observed:evidence+=float(np.mean(observed))
    state['rival_evidence']=np.clip(evidence,-6.,6.)
    return .96/(1.+np.exp(-np.clip(state.get('rival_prior',-1.5)+2.8*evidence,-8,8)))


def _r6_mix_rival(state,enemy,base,rival,players):
    w,b=base
    fullw,fullb=enemy[0]+state['rival_worker'],enemy[1]+state['rival_base']
    gate=state['rival_gate']
    observations={}
    for uid,(pos,cargo,kind) in players[2].items():
        a=_probs(fullw[pos],.90);prior=_probs(w[pos],.90)
        observations[uid]=pos,np.clip(np.log(np.maximum(a,1e-7))-np.log(np.maximum(prior,1e-7)),-3.,3.)
    state['rival_pending']=observations
    return (1-gate)*w+gate*fullw,(1-gate)*b+gate*fullb


def build(code,weights):
    def change(old,new):
        nonlocal code
        if code.count(old)!=1:raise ValueError(old)
        code=code.replace(old,new)
    data=base64.b64encode(weights.read_bytes()).decode()
    source="with np.load(io.BytesIO(base64.b64decode('"+data+"')),allow_pickle=False) as _r6_foe:\n    _R6_FOE_W,_R6_FOE_B=_r6_foe['weights'],_r6_foe['biases']\n"
    change('def _v5_root(obs,',source+'\n'+inspect.getsource(_r6_rival_gate)+'\n'+inspect.getsource(_r6_mix_rival)+'\n\ndef _v5_root(obs,')
    change("    state['root_ids'] = [expert,enemy_style]","""    gate=_r6_rival_gate(state,players,op,step)
    state['rival_gate']=gate
    state['root_ids'] = [expert,20 if gate>.5 else enemy_style]""")
    change("    ids[me],ids[op] = expert,enemy_style","    ids[me],ids[op] = state['root_ids']")
    change("    state['opponent'] = _v5_mix(enemy,rival,.8)","    state['opponent'] = _r6_mix_rival(state,enemy,_v5_mix(enemy,rival,.8),rival,players[op])")
    change("        for i, (cells, _, actions, _) in enumerate(_V5_ROTATIONS):","""        if side==1:
            fit_w=np.zeros_like(workers);fit_b=np.zeros_like(bases)
        for i, (cells, _, actions, _) in enumerate(_V5_ROTATIONS):""")
    change("                role = 0 if kind == 'COLLECTOR' else 1\n                delta =","""                role = 0 if kind == 'COLLECTOR' else 1
                if side==1:
                    fit_w[pos]+=.25*(_R6_FOE_W[role]@hidden[pos]+_R6_FOE_B[role])[actions]
                delta =""")
    change("                delta = _V5['coef'][:, 2, :3] @ hidden[pos]", """                if side==1:
                    fit_b[pos]+=.25*(_R6_FOE_W[2,:3]@hidden[pos]+_R6_FOE_B[2,:3])
                delta = _V5['coef'][:, 2, :3] @ hidden[pos]""")
    change("        sides.append((workers, bases, offsets))","""        if side==1:
            _ST['v5_history']['rival_worker'],_ST['v5_history']['rival_base']=fit_w,fit_b
        sides.append((workers, bases, offsets))""")
    change("        policies[side] = (mix*weights,mix*biases)","""        gate=_ST['v5_history']['rival_gate'] if side!=own_side else 0.
        policies[side] = ((1-gate)*mix*weights+gate*_R6_FOE_W,(1-gate)*mix*biases+gate*_R6_FOE_B)""")
    compile(code,'<public-rival-model>','exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','weights','output'):p.add_argument(k,type=Path)
    a=p.parse_args();a.output.write_text(build(a.source.read_text(),a.weights))
