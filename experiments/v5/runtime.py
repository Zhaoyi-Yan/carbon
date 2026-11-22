"""Live mixture of twenty teacher policies and a short-horizon economic value."""
def _v5_heads(hidden, player):
    prior = _POLICY.p
    workers = hidden @ prior['hw.w'].reshape(5, -1).T + prior['hw.b']
    bases = hidden @ prior['hb.w'].reshape(3, -1).T + prior['hb.b']
    # Return the offset for each unit only. The 20 experts share one trunk inference.
    offsets = {}
    for uid, (pos, _, kind) in player[2].items():
        role = 0 if kind == 'COLLECTOR' else 1
        offsets[uid] = (pos, _V5['coef'][:, role] @ hidden[pos] + _V5['bias'][:, role], False)
    for uid, pos in player[1].items():
        offsets[uid] = (pos, _V5['coef'][:, 2, :3] @ hidden[pos] + _V5['bias'][:, 2, :3], True)
    return workers, bases, offsets


def _v5_mix(heads, weights, strength):
    w, b, offsets = heads
    workers, bases = w.copy(), b.copy()
    for pos, delta, base in offsets.values():
        target = bases if base else workers
        target[pos] += strength * (weights @ delta)
    return workers, bases


def _v5_root(obs, est, x, previous):
    step, me, players = obs['step'], obs['player'], obs['players']
    op = 1-me
    if step == 0 or 'v5_history' not in _ST:
        _ST['v5_history'] = dict(evidence=np.zeros(20), pending={}, step=-1, style=0)
    state = _ST['v5_history']
    evidence = state['evidence']
    observed = []
    if state['step'] == step-1:
        for uid, (start, predicted) in state['pending'].items():
            unit = players[op][2].get(uid)
            if unit and unit[0] in _ROLLOUT_LANDING[start]:
                move_id = _ROLLOUT_LANDING[start].index(unit[0])
                observed.append(predicted[:, move_id])
    if state['step'] != step:
        evidence *= .93
        if observed:
            logprob = np.mean(observed, axis=0)
            evidence += logprob - logprob.mean()
    econ, styles = _v5_observe(players, obs.get('trees', {}), step, LAST_TURN)
    memory = styles[op] if step == 0 or 'behavior' not in state else .89*state['behavior']+.11*styles[op]
    state['behavior'] = memory
    styles[op] = memory
    value, context = _v5_context(econ, styles, players, step, LAST_TURN, me)
    phase = min(3, step*4//max(1, LAST_TURN+1))
    separation = ((_V5['prototypes'][phase]-memory)/_V5['scale'])**2
    post = .8*evidence - .40*separation.sum(1)
    rival = np.exp(np.clip(post-post.max(),-30.,0.))
    rival /= rival.sum()
    merit = _V5['route'] @ context - .095*np.log1p(np.arange(20))
    if step > 0:
        merit[state['style']] += .08
    order = np.argsort(-merit)
    style = int(order[0]) if step >= 12 else 0
    state['style'] = style
    opponent_x = _quant(encode(players,step,op,est,[1.]*N,previous))
    h = _net()(np.stack((x,opponent_x)).astype(np.float32), _SID, hidden=True)
    own = _v5_heads(h[0],players[me])
    enemy = _v5_heads(h[1],players[op])
    eye = _V5_EYE
    mine = _v5_mix(own, eye[style], _V5_OWN_STRENGTH)
    second = next(int(k) for k in order if k != style)
    state['alternatives'] = [(own[0], own[1]), _v5_mix(own,eye[second],.8)]
    state['opponent'] = _v5_mix(enemy,rival,.8)
    state['pending'] = {}
    for uid,(pos,offset,base) in enemy[2].items():
        if not base:
            logits = enemy[0][pos] + offset
            z = logits - logits.max(axis=1,keepdims=True)
            state['pending'][uid] = pos, np.maximum(-8., z-np.log(np.exp(z).sum(axis=1, keepdims=True)))
    state['step'] = step
    return mine


def _v5_value(board, me):
    state = board.get('_st')
    meta = state.trees_meta if state is not None else board.get('trees', {})
    pl = board['players']
    economic, _ = _v5_observe(pl, meta, board['step'], LAST_TURN)
    delta = economic[me]-economic[1-me]
    phase = board['step']/max(1,LAST_TURN)
    late = max(0.,1.-max(0,LAST_TURN+1-board['step'])/50.)
    diff = pl[me][0]-pl[1-me][0]
    cash = np.clip(diff/1000.,-4.,4.)
    features = np.concatenate((delta,delta*phase,delta*late,[cash,cash*phase]))
    return (diff/1000. + features@_V5['economic'])/3.
