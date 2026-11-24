"""Record the original search's intermediate values without new samples or network calls."""
from instrument import build, insert_after


def instrument(source, sha):
    text = build(source, expected_sha256=sha)
    start = text.index('    shipping = _v7_proposals', text.index('def _audit_pool('))
    end = text.index("\n\ndef _act(", start)
    text = text[:start] + """    _AUDIT.update(estimated_carbon=list(est), fixed=dict(fixed or {}), search_seed=seed,
                  root_styles=list(_ST['v5_history']['root_ids']))
""" + text[end:]
    text = insert_after(text, "    mode_keys={tuple(sorted(cands[i].items())):mode for i,mode in root_modes.items()}",
                        "    if _AUDIT is not None:\n        _AUDIT['before_protection']=[dict(a) for a in cands]")
    text = insert_after(text, '    priors=_r6_action_costs(players,me,lw0,lb0,cands,step)',
                        """    if _AUDIT is not None:
        _AUDIT.update(priors=list(priors), modes=dict(root_modes), requested_depth=D2, reply_budget=M)""")
    text = insert_after(text, '    if first_reply is not None:opp[0]=first_reply',
                        "    if _AUDIT is not None and 'reply_override' in _AUDIT:\n        opp[0] = _AUDIT['reply_override']")
    text = insert_after(text, '    boards = begin(alive, range(M))', """    if _AUDIT is not None:
        _AUDIT['root_simulated'] = [
            dict(candidate=c,reply=m,players=g['players'],trees=g['_st'].trees_meta)
            for c,m,g,_ in boards
        ]""")
    text = insert_after(text, '            for board, value in zip(part, vv):',
                        """                raw = value
                banked = mix = legacy = fc = netcash = None""")
    a = """                    value += _V5_VALUE_MIX * (_fc_leaf(future, me, LAST_TURN, _v5_value(future, me)) - value)
                    value += 0.26 * (_v7_nc_leaf(future,me) - value)
                v.append(float(value))"""
    b = """                    legacy = _v5_value(future, me)
                    fc = _fc_leaf(future, me, LAST_TURN, legacy)
                    value += _V5_VALUE_MIX * (fc - value)
                    netcash = _v7_nc_leaf(future,me)
                    value += 0.26 * (netcash - value)
                if _AUDIT is not None:
                    _AUDIT.setdefault('leaf', []).append(dict(candidate=board[0], reply=board[1],
                        turn=future['step'], both=both, raw_critic=float(raw), banking_weight=mix,
                        banking_value=banked, legacy=legacy, fc=fc, fc_weight=_V5_VALUE_MIX,
                        netcash=netcash, netcash_weight=.26, value=float(value)))
                v.append(float(value))"""
    assert text.count(a) == 1
    text = text.replace(a,b)
    text = insert_after(text, '            alive = [0] + sorted((c for c in alive if c), key=lambda c: -mean[c])[:keep - 1]',
                        """            if _AUDIT is not None:
                _AUDIT.setdefault('pruning', []).append(dict(depth=t, kept=list(alive), scores=dict(mean)))""")
    text = insert_after(text, '    if depth >= 2 and len(alive) >= 2 and _time.perf_counter()-t0 < .69:',
                        """        if _AUDIT is not None:
            _AUDIT['before_tail'] = dict(mean)""")
    text = insert_after(text, '            shared=.5*(sum(mean[c] for c in two)-sum(tail))',
                        """            if _AUDIT is not None:
                _AUDIT['tail'] = dict(candidates=list(two), values=list(tail), shared=shared, weight=.48)""")
    text = insert_after(text,
                        "        _AUDIT.update(final_scores={k:mean[k] for k in alive}, selected=best)",
                        "        if 'force' in _AUDIT:\n            assert _AUDIT['force'] in alive\n            best = _AUDIT['force']")
    text = insert_after(text, '    best = max(alive, key=lambda c: mean[c])',
                        """    if _AUDIT is not None:
        _AUDIT.update(max_before_eps=best, eps=_S_EPS, base_score=mean[0],
                      eval_values=[(i,m,v) for (i,m),v in val.items()])""")
    a = '            state[3],state[2]=pl,_turn(board,config,moves)'
    text = insert_after(text,a,"""            if _AUDIT is not None:
                _AUDIT.setdefault('tail_moves', []).append(dict(candidate=state[0],turn=t, actions=moves))""")
    a = "        valuations.append(p[0]+c+tree+horizon*workforce-float_cost)"
    return insert_after(text,a,"""        if _AUDIT is not None:
            _AUDIT.setdefault('asset_calls', []).append(dict(side=side,team=me,turn=t,cash=p[0],
                cargo_value=c, tree_value=tree, workforce=horizon*workforce, liquidity_cost=float_cost))""")
