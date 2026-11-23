"""Observe the v7 root search and execution chain without changing policy inputs."""
import hashlib

V7_SHA256 = '4894fe44c601b7360d37de2eed928802c73884f0929a5897604d174e333e02e6'

# Hooks inside the original search observe which sources actually reach their
# generator. The shadow probes below separately cover branches gated on width.
INSERTIONS = (
    ('    cands = [act0]; keys = {tuple(sorted(act0.items()))}',
     "    _audit_event('root',act0)"),
    ('        for mode,action in alternatives[:limit]:',
     "            _audit_event('shipping:'+mode,action)"),
    ('            marker = tuple(sorted(option.items()))\n            if marker not in keys:',
     "            _audit_event('teacher',option)"),
    ('        if tree_action is not None:',
     "            _audit_event('first_tree',tree_action)"),
    ('            proposal = _sample(players, me, aw, ab, rng, step, fixed)',
     "            _audit_event('style',proposal)"),
    ('        if patch is not None:',
     "            _audit_event('patch',patch)"),
    ('                a = _sample(players, me, lwt, lb0, rng, step, fixed)',
     "                _audit_event('sample',a)"),
    ('            for a in _deviations(players, me, lw0, act0, fixed, cap):',
     "                _audit_event('deviation',a)"),
)

POSTPROCESSORS = ('_reply_choice', '_v6_last_worker', '_v7_cash_route', '_v7_keep_float',
                  '_v7_short_deposit', '_v7_corridor', '_v7_solvency', '_r6_opening',
                  '_v7_early_income')

HOOKS = '''
_AUDIT = None


def _audit_event(source, action):
    if _AUDIT is not None:
        _AUDIT['proposed'].append((source, dict(action)))


def _audit_stage(name, action):
    if _AUDIT is not None:
        _AUDIT['stages'].append((name, dict(action)))


def _audit_pool(obs, cfg, est, base, lw, lb, fixed, protected, width, seed, cands):
    if _AUDIT is None:
        return
    side, turn, players = obs['player'], obs['step'], obs['players']
    _AUDIT.update(width=width, admitted=[dict(a) for a in cands],
                  protected=dict(protected or {}))
    shipping = _v7_proposals(obs, cfg, est, base, fixed or {}, _v7_flows({
        'players': players, 'step': turn, 'trees': obs.get('trees', {}),
        'carbon': est, 'player': side}), 3)
    shadow = [('shipping:'+mode,a) for mode,a in shipping]
    teacher = _ST.get('own_search_prior')
    if teacher is not None:
        bases = lb.copy()
        for uid, pos in players[side][1].items():
            a = BACT.index(base[uid]) if uid in base else 0
            bases[pos] = -100.
            bases[pos,a] = 0.
        shadow.append(('teacher', _sample(players, side, teacher*_TEMP, bases,
                                         np.random.RandomState(seed+45289), turn, fixed)))
    first_tree = _r6_first_tree(obs,cfg,est,base,lw)
    patch = _r6_patch_proposal(obs,est,base,fixed or {},lw) if width>=4 else None
    shadow.extend((name,a) for name,a in (('first_tree',first_tree),('patch',patch)) if a is not None)
    shadow.extend(('deviation',a) for a in _deviations(players,side,lw,base,fixed,2))
    for mode,(w,b) in enumerate(_ST['v5_history']['alternatives']):
        shadow.append(('style', _sample(players,side,w,b,np.random.RandomState(seed+47259+mode),turn,fixed)))
    _AUDIT['probes'] = shadow
'''


def insert_after(source, anchor, extra):
    if source.count(anchor) != 1:
        raise ValueError(f'Expected one v7 anchor: {anchor}')
    return source.replace(anchor, anchor+'\n'+extra)


def build(source):
    if hashlib.sha256(source.encode()).hexdigest() != V7_SHA256:
        raise ValueError('Expected frozen yan-v7 source')
    entry = 'def _act(obs, configuration=None, t0=None):'
    start = source.index(entry)
    end = source.index('\ndef agent(', start)
    body = source[start:end]
    lines = []
    for line in body.splitlines():
        lines.append(line)
        names = POSTPROCESSORS if 'act' in line else ()
        for name in names:
            if f'={name}(' in line or f'= {name}(' in line:
                indent = line[:len(line)-len(line.lstrip())]
                lines.append(indent+f"_audit_stage({name!r},act)")
        if 'act = _search(' in line:
            lines.append("            _audit_stage('search',act)")
        if line == '    st[\'tr\'].set_actions(players, act)':
            lines.insert(-1, "    _audit_stage('executed',act)")
    source = source[:start]+HOOKS+'\n\n'+'\n'.join(lines)+'\n'+source[end:]
    for anchor, extra in INSERTIONS:
        if anchor.endswith('if marker not in keys:'):
            source = source.replace(anchor, anchor.replace('            if marker',extra+'\n            if marker'),1)
        else:
            source = insert_after(source,anchor,extra)
    at = '    if len(cands) < 2:\n        _S_STATS'
    hook = ('    _audit_pool(obs,configuration,est,act0,lw0,lb0,fixed,protected,most,seed0,cands)\n'
            "    if _AUDIT is not None:\n        _AUDIT['replies'] = []\n")
    if source.count(at) != 1:
        raise ValueError('Missing candidate pool boundary')
    source = source.replace(at,hook+at)
    source = insert_after(source, "    _S_STATS['last'] = (len(cands), depth, rolls, dt, tuple(_S_COST))",
                          "    if _AUDIT is not None:\n        _AUDIT.update(scored_depth=dict(dep), depth=depth, rolls=rolls, replies=[opp[k] for k in sorted(opp)])")
    return source.replace('    if _S_REC is not None:',
                          "    if _AUDIT is not None:\n        _AUDIT.update(final_scores={k:mean[k] for k in alive}, selected=best)\n    if _S_REC is not None:",1)
