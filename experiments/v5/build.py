"""Build a standalone adaptive strategy from the frozen yan-v4 and fitted weights."""
import argparse
import base64
from pathlib import Path
from backbone import exposed_source, replace, ROOT

HERE=Path(__file__).resolve().parent


def build(weights, output, own, valuation, frontier=False, rotated=False, fog=False, territory=False, deep=False, adaptive=False, paired=False):
    source=exposed_source()
    batch=(ROOT/'opponents/yan-v3.py').read_text()
    sample=batch[batch.index('_ROLLOUT_LANDING ='):batch.index('def _deviations(')]
    source=source[:source.index('def _sample(')] + sample + source[source.index('def _deviations('):]
    body=(HERE/'features.py').read_text()+'\n'+(HERE/'runtime.py').read_text()+'\n'
    if frontier:
        body += (HERE/'frontier.py').read_text()+'\n'
    if rotated:
        body += (HERE/'symmetry.py').read_text()+'\n'
        start=body.index('    h = _net()(np.stack((x,opponent_x))')
        end=body.index('    eye = _V5_EYE', start)
        body=body[:start]+"    own, enemy = _v5_rotated_heads(x,opponent_x,players[me],players[op])\n"+body[end:]
        source=replace(source, '                    X[2 * i + 1] = view(b, op)',
                       '''                    X[2 * i + 1] = view(b, op)
                    S[2*i+1] = -S[2*i]
                    S[2*i+1,2] = S[2*i,2]''')
        source=replace(source,'                vv = vv[::w]',
                       '                vv = .5*(vv[::2]-vv[1::2]) if both else vv')
    if territory:
        extra=(HERE/'territory.py').read_text()
        body += extra[extra.index('_V5_GRID ='):]+'\n'
        body=replace(body,'def _v5_value(board, me):','def _v5_cash_only(board, me):')
        body+='\n_v5_value = _v5_territory_value\n'
    if fog:
        body+=(HERE/'fog.py').read_text()+'\n'
        source=replace(source, "    x = encode(players, step, me, est, known, st['prev'])", '''    if step == 0 or 'fog' not in st:
        st['fog'] = _V5Map(players)
    projected = st['fog'].observe(obs,me)
    # The policy was trained with the legacy estimate; the simulator uses the posterior.
    network_est = .68*projected + .32*np.asarray(est)
    x = encode(players, step, me, network_est, known, st['prev'])
    est = projected''')
        source=replace(source,"    st['tr'].set_actions(players, act)",
                       "    st['fog'].moves = act.copy()\n    st['tr'].set_actions(players, act)")
        source=replace(source,"    opp = {}","    opp = {}\n    determinizations = {}")
        source=replace(source,"                out.append([c, m, _turn(root, configuration, acts), players])", '''                if m not in determinizations:
                    board = root.copy()
                    if m:
                        board['carbon'] = _ST['fog'].sample(np.random.RandomState(seed0+2141*m))
                    determinizations[m] = board
                out.append([c, m, _turn(determinizations[m], configuration, acts), players])''')
    if deep:
        body+=(HERE/'rollout.py').read_text()+'\n'
        body=replace(body,"    state['step'] = step",
                     "    _v5_rollout_set(me,style,rival)\n    state['step'] = step")
        source=replace(source,'sid, sc=None, hidden=False):','sid, sc=None, hidden=False, play_sides=None):',2)
        first=source.index('class _TorchNet:')
        second=source.index('class _WinoNet:')
        end=source.index('_S_WINO =',second)
        anchor='        return lw, lb, v'
        def expose(block,layout):
            return replace(block,anchor, '        if play_sides is not None:\n            _v5_rollout_update(X,'+layout+',lw,lb,play_sides)\n'+anchor)
        source=source[:first]+expose(source[first:second], 'hf.transpose(1,2).contiguous().numpy()')+expose(source[second:end], 'hf.permute(1,0,2).contiguous().numpy()')+source[end:]
        source=replace(source,'                lw, lb, vv = net(X, _SID, S)',
                       '                lw, lb, vv = net(X, _SID, S, play_sides=([me,op]*n if both else [me]*n))')
        source=replace(source,'                lw2, lb2, _ = net(Xo, _SID)',
                       '                lw2, lb2, _ = net(Xo, _SID, play_sides=[op]*len(part))')
    if adaptive:
        body=replace(body, "    merit = _V5['route'] @ context - .095*np.log1p(np.arange(20))",
                     "    merit = (_V5['route'] @ context - .80*_V5['route_center']\n             + .20*(_V5['matchup'] @ rival) - .024*np.log1p(np.arange(20)))")
        body=replace(body, "        merit[state['style']] += .08","        merit[state['style']] += .014")
    if paired:
        body+=(HERE/'encode_pair.py').read_text()+'\n'
        source=replace(source, "        return _quant(encode(o['players'], o['step'], who, o['carbon'], full, b[3]))",
                       '''        if 'v5_views' not in o:
            o['v5_views'] = _v5_pair(o['players'],o['step'],o['carbon'],b[3])
        return o['v5_views'][who]''')
    blob=base64.b64encode(weights.read_bytes()).decode()
    body += f'with np.load(io.BytesIO(base64.b64decode("{blob}")), allow_pickle=False) as _v5_file:\n    _V5 = dict(_v5_file)\n'
    body += f'_V5_EYE = np.eye(20)\n_V5_OWN_STRENGTH = {own!r}\n_V5_VALUE_MIX = {valuation!r}\n\n'
    source=replace(source,'def _act(obs,',body+'def _act(obs,')
    source=replace(source,'    lw, lb = _POLICY.forward(x, _SID)',
                   "    lw, lb = _v5_root(obs, est, x, st['prev']) if _SEARCH_OK else _POLICY.forward(x, _SID)")
    anchor='    for T, cap in _S_GEN:'
    source=replace(source,anchor,'''    if most >= 4:
        for aw, ab in _ST['v5_history']['alternatives'][:min(2,(most-1)//3)]:
            proposal = _sample(players, me, aw, ab, rng, step, fixed)
            marker = tuple(sorted(proposal.items()))
            if marker not in keys:
                keys.add(marker)
                cands.append(proposal)
'''+anchor)
    if frontier:
        source=replace(source,anchor,"""    if most >= 3 and len(cands)<most:
        counter = _v5_countermove(root, configuration, act0, fixed or {}, lw0, lb0, seed0, t0)
        if counter is not None and tuple(sorted(counter.items())) not in keys:
            keys.add(tuple(sorted(counter.items())))
            cands.append(counter)
"""+anchor)
    source=replace(source,"    xo = _quant(encode(players, step, op, est, full, prev))[None]\n    lwo, lbo, _ = net(xo.astype(np.float32), _SID)",
                   "    lwo, lbo = _ST['v5_history']['opponent']")
    source=replace(source,'_sample(players, op, lwo[0], lbo[0],','_sample(players, op, lwo, lbo,')
    source=replace(source,'                v.append(float(value))',
                   '''                if not both and horizon > 0 and _V5_VALUE_MIX:
                    value += _V5_VALUE_MIX * (_v5_value(future, me) - value)
                v.append(float(value))''')
    source=replace(source,'Base code: yan-v1', 'Additional yan-v5 work: twenty replay-trained action heads, rival identification,\ncontext-dependent style selection and 48-step cash evaluation.\nBase code: yan-v1')
    compile(source,str(output),'exec')
    output.write_text(source)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('weights',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--own',type=float,default=.75);p.add_argument('--value',type=float,default=.65)
    p.add_argument('--frontier',action='store_true')
    p.add_argument('--rotated',action='store_true')
    p.add_argument('--fog',action='store_true')
    p.add_argument('--territory',action='store_true')
    p.add_argument('--deep',action='store_true')
    p.add_argument('--adaptive',action='store_true')
    p.add_argument('--paired',action='store_true')
    a=p.parse_args();build(a.weights,a.output,a.own,a.value,a.frontier,a.rotated,a.fog,a.territory,a.deep,a.adaptive,a.paired)
