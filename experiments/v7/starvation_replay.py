"""Reconstruct every cash movement and QQ's delivery opportunities in the matched public replay."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

MATCH = '4063a34afedb6d4767a9118e1dee9e5d'
QQ = 1
N = 15
ACTIONS = ('STAY', 'UP', 'RIGHT', 'DOWN', 'LEFT')
SHIFT = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))
LAND = [{action: ((cell // N + dy) % N) * N + (cell % N + dx) % N
         for action, (dy, dx) in zip(ACTIONS, SHIFT)} for cell in range(N*N)]
FLOW = ('base', 'tree', 'collision_tree', 'hire_cost', 'plant_cost', 'seize_cost')
ROOT = Path(__file__).resolve().parents[2]


def distance(a, b):
    dy, dx = abs(a // N - b // N), abs(a % N - b % N)
    return min(dy, N-dy) + min(dx, N-dx)


def read_replay(path):
    raw = path.read_bytes()
    source = gzip.decompress(raw) if raw.startswith(b'\x1f\x8b') else raw
    payload = source.decode()
    if 'window.kaggle =' in payload:
        return json.JSONDecoder().raw_decode(payload.split('window.kaggle =', 1)[1].lstrip())[0]['environment']
    return json.loads(payload)


def landings(players, actions, config):
    """Include successful recruits even if they die before appearing in the next observation."""
    groups = defaultdict(list)
    for side, (money, bases, workers, _) in enumerate(players):
        count = len(workers)
        for wid, (pos, cargo, kind) in workers.items():
            groups[LAND[pos][actions[side].get(wid, 'STAY')]].append((side, wid, cargo, kind))
        for bid, pos in bases.items():
            act = actions[side].get(bid)
            if act not in ('RECCOLLECTOR', 'RECPLANTER'):
                continue
            kind = act[3:]
            cost = config['rec' + kind.title() + 'Cost']
            assert config['rec' + kind.title() + 'IncreaseCost'] == 0
            if count < config['workerLimit'] and money >= cost:
                groups[pos].append((side, '+' + bid, 0., kind))
                money -= cost
                count += 1
    return groups


def cash_flows(before, after, config, tree_origin):
    tick = after[0]['observation']['step']
    old = before[0]['observation']['players']
    obs = after[0]['observation']
    new = obs['players']
    actions = [p['action'] or {} for p in after]
    homes = {pos: s for s, p in enumerate(old) for pos in p[1].values()}
    live_tree = {t[0]: s for s, p in enumerate(old) for t in p[3].values() if t[1] < config['treeLifespan']}
    tree_quote = config['plantCost'] + config['plantCostInflationRatio'] * config['plantCostInflationBase'] ** len(live_tree)
    flow = [Counter(), Counter()]
    deposits, casualties, changes = [], [], []
    for pos, records in landings(old, actions, config).items():
        collectors = [w for w in records if w[3] == 'COLLECTOR']
        light = min((w[2] for w in collectors), default=None)
        tied = [w for w in collectors if w[2] == light]
        winner = records[0] if len(records) == 1 else (tied[0] if len(tied) == 1 else None)
        for s, wid, cargo, kind in records:
            if wid.startswith('+'):
                flow[s]['hires'] += 1
                flow[s]['hire_cost'] += config['rec' + kind.title() + 'Cost']
            if pos in homes and cargo:
                recipient = homes[pos]
                flow[recipient]['base'] += cargo
                flow[recipient]['base_own' if s == recipient else 'base_enemy'] += cargo
                deposits.append([tick, s, wid, round(cargo, 3), recipient, pos])
            removed = len(records) > 1 and (winner is None or winner[1] != wid)
            if removed:
                flow[s]['deaths'] += 1
                if pos in homes:
                    dest = 'our_base' if homes[pos] == s else 'enemy_base'
                elif winner:
                    dest = 'teammate' if winner[0] == s else 'opponent'
                elif pos in live_tree:
                    dest = 'our_tree' if live_tree[pos] == s else 'enemy_tree'
                    flow[live_tree[pos]]['collision_tree'] += cargo
                else:
                    dest = 'ground'
                flow[s]['dead_cargo_' + dest] += cargo
                if cargo:
                    casualties.append([tick, s, wid, round(cargo, 3), dest, pos])
            elif winner is not None and winner[1] == wid and kind == 'COLLECTOR' and pos in homes and s != homes[pos] and not sum(w[2] for w in records):
                flow[s]['deaths'] += 1
    for s, p in enumerate(new):
        plant_price = config['plantCost'] if len(old[s][3]) <= config['playerTreeProtectiveNumber'] else tree_quote
        for tid, (cell, age) in p[3].items():
            if tid not in old[s][3]:
                taken = tid in old[1-s][3]
                field = 'seizes' if taken else 'plants'
                flow[s][field] += 1
                flow[s]['seize_cost' if taken else 'plant_cost'] += config['seizeCost'] if taken else plant_price
                if not taken:
                    tree_origin[tid] = s
                changes.append([tick, s, field, tid, cell, age])
            income = obs['trees'][tid][1]
            flow[s]['tree'] += income
            if tree_origin[tid] != s:
                flow[s]['tree_captured_origin'] += income
        predicted = old[s][0] + sum(flow[s][v] for v in FLOW[:3]) - sum(flow[s][v] for v in FLOW[3:])
        flow[s]['rounding'] = p[0] - predicted
        assert abs(flow[s]['rounding']) <= .00501, (tick, s, p[0], predicted)
        assert len(p[2]) == len(old[s][2]) + flow[s]['hires'] - flow[s]['deaths'], (tick, s, flow[s])
    return flow, deposits, casualties, changes


def threats(players, mine, cargo):
    foe = players[1-mine]
    light = [v[0] for v in foe[2].values() if v[2] == 'COLLECTOR' and v[1] <= cargo]
    danger = {c for pos in light for c in LAND[pos].values()}
    if foe[0] >= 30 and len(foe[2]) < 10:
        danger.update(foe[1].values())
    return danger


def route_stats(frames, first, stop):
    """Count options visible before act. Entering an own base banks cargo even on a lethal collision."""
    count = Counter()
    for t in range(first, stop):
        pl = frames[t][0]['observation']['players']
        own = pl[QQ]
        act = frames[t+1][QQ]['action'] or {}
        for uid, (pos, cargo, kind) in own[2].items():
            if kind != 'COLLECTOR' or cargo < 30:
                continue
            bases = set(own[1].values())
            danger = threats(pl, QQ, cargo)
            teammates = {LAND[v[0]][act.get(k, 'STAY')] for k, v in own[2].items() if k != uid}
            d = lambda v: min(distance(v, b) for b in bases)
            closer = {c for c in LAND[pos].values() if d(c) < d(pos)}
            safe = {c for c in closer if c in bases or (c not in danger and c not in teammates)}
            nxt = LAND[pos][act.get(uid, 'STAY')]
            count['loaded_worker_turns'] += 1
            count['both_bases_within_enemy_reach'] += int(bases <= danger)
            count['no_safe_shortening_step'] += int(not safe)
            count['has_safe_shortening_step'] += int(bool(safe))
            count['safe_shortening_declined'] += int(bool(safe) and nxt not in safe)
            count['any_shortening_chosen'] += int(nxt in closer)
            count['one_step_from_deposit'] += int(d(pos) == 1)
            count['one_step_deposit_declined'] += int(d(pos) == 1 and nxt not in bases)
    return dict(count)


def route_probe(frames, tick, uid, guard_radius, limit=14):
    """Historical sensitivity only. Freeze other movement, including friendly choices; own observation at t determines origin."""
    own = frames[tick][0]['observation']['players'][QQ]
    pos, _, _ = own[2][uid]
    origin = pos
    homes = set(own[1].values())
    options = {pos: [pos]}
    for t in range(tick, min(tick+limit, len(frames)-1)):
        before = frames[t][0]['observation']['players']
        actions = [p['action'] or {} for p in frames[t+1]]
        enemy = before[1-QQ]
        if guard_radius:
            forbidden = {q for w in enemy[2].values() if w[2] == 'COLLECTOR' for q in LAND[w[0]].values()}
            if enemy[0] >= 30 and len(enemy[2]) < 10:
                forbidden.update(enemy[1].values())
        else:
            # Conservatively reject every other opponent collector landing even if we would win.
            forbidden = {LAND[w[0]][actions[1-QQ].get(k,'STAY')] for k,w in enemy[2].items() if w[2]=='COLLECTOR'}
            forbidden.update(pos for k,pos in enemy[1].items() if actions[1-QQ].get(k)=='RECCOLLECTOR')
        forbidden.update(LAND[w[0]][actions[QQ].get(k,'STAY')] for k,w in before[QQ][2].items() if k != uid)
        forbidden.update(enemy[1].values())
        paths = {}
        for cell, path in options.items():
            for q in LAND[cell].values():
                if q in homes:
                    return dict(arrival=t+1, steps=t+1-tick, path=path+[q], origin=origin)
                if q not in forbidden and q not in paths:
                    paths[q] = path+[q]
        options = paths
        if not options:
            break
    return None


def two_step_receipts(frames, config):
    """Certify a positive receipt against all rival movements using the present board only."""
    candidates = []
    for t in range(0,len(frames)-2):
        pl = frames[t][0]['observation']['players']
        own, foe = pl[QQ], pl[1-QQ]
        homes = set(own[1].values())
        # Joint own actions are under our control. Keep all other actual friendly actions on step one.
        action = frames[t+1][QQ]['action'] or {}
        for uid,(pos,cargo,kind) in own[2].items():
            if kind != 'COLLECTOR' or cargo < 30 or min(distance(pos, b) for b in homes) != 2:
                continue
            taken = {LAND[w[0]][action.get(k,'STAY')] for k,w in own[2].items() if k != uid}
            danger = threats(pl,QQ,cargo)
            safe = []
            for middle in LAND[pos].values():
                if middle in taken or middle in danger or middle in foe[1].values():
                    continue
                homes_in_one = sorted(q for q in homes if distance(q,middle)==1)
                if not homes_in_one:
                    continue
                # Each existing tree could tax up to 10%, including if stolen now. New seedlings earn zero.
                covering = 0
                for p in pl:
                    for cell,age in p[3].values():
                        dy,dx = abs(middle//N-cell//N),abs(middle%N-cell%N)
                        if age < config['treeLifespan'] and max(min(dy,N-dy),min(dx,N-dx))<=1:
                            covering += 1
                floor = cargo*max(0,1-config['collectorAbsorptionRate']*covering)
                if floor > 0:
                    safe.append(dict(cell=middle,bases=homes_in_one,receipt_floor=round(floor,3)))
            if safe:
                picked = LAND[pos][action.get(uid,'STAY')]
                candidates.append(dict(step=t,uid=uid,cell=pos,cargo=round(cargo,3),cash=own[0],chosen_cell=picked,
                                       chosen_first_safe=picked in {a['cell'] for a in safe},options=safe))
    return candidates


def verify_paths(frames, cfg, probes):
    """Play a single alternate carrier against recorded other moves, using the competition referee."""
    import sys
    sys.path[:0] = [str(ROOT),str(ROOT/'experiments/v6r')]
    from benchmark import configure_engine,limit_math_threads,new_game
    limit_math_threads()
    configure_engine()
    from zerosum_env.envs.carbon import carbon,idgen
    from close_replay import archive_state,restore,referee_events
    _,_ = new_game(cfg['randomSeed'],len(frames))
    initial = dict(seats=[dict(observation=p['observation'],status=p['status'],reward=p['reward'],action=p['action'] or {}) for p in frames[0]],
                   id_counters=dict(idgen._id_dict))
    state,env=restore(initial,cfg)
    saved={}
    for t in range(165):
        assert state[0].observation.players==frames[t][0]['observation']['players']
        assert state[0].observation.full_carbon==frames[t][0]['observation']['full_carbon']
        if t in (90,164):
            saved[t]=archive_state(state)
        if t!=164:
            for p,src in zip(state,frames[t+1]): p.action=src['action'] or {}
            carbon.compute_next_board(state,env)
    runs=[]
    for case in probes:
        tick = case['step']
        if tick not in saved or not case['all_neighbor']:
            continue
        path=case['all_neighbor']['path']
        uid=case['uid'];state,env=restore(saved[tick],cfg)
        amt=0.
        traced=[]
        for i,(src,dst) in enumerate(zip(path,path[1:])):
            assert state[0].observation.players[QQ][2][uid][0]==src
            to_do=[dict(s['action'] or {}) for s in frames[tick+i+1]]
            chosen=next(k for k,v in LAND[src].items() if v==dst)
            to_do[QQ].pop(uid,None)
            if chosen!='STAY': to_do[QQ][uid]=chosen
            if i==len(path)-2:
                amt=state[0].observation.players[QQ][2][uid][1]
            for p,a in zip(state,to_do): p.action=a
            transitions=[]
            with referee_events(transitions):
                carbon.compute_next_board(state,env)
            traced.extend([tick+i+1,e[0],round(e[2],3)] for e in transitions
                          if e[1]==QQ and e[0] in ('bank','base_collision','tree_collision','tree_yield','seize','plant','hire_collector','hire_planter') and abs(e[2])>1.e-6)
        t=tick+len(path)-1
        actual=frames[t][0]['observation']['players'][QQ][0]
        result=state[0].observation.players[QQ][0]
        runs.append(dict(begins=tick,uid=uid,arrival=t,route=path,credited_carrier=round(amt,3),official_cash=actual,
                         alternative_cash=result,delta=round(result-actual,2),our_cash_events=traced))
    return runs


def board(players, s):
    p = players[s]
    return dict(cash=p[0], cargo=round(sum(w[1] for w in p[2].values()),3),
                collectors=sum(w[2]=='COLLECTOR' for w in p[2].values()), planters=sum(w[2]=='PLANTER' for w in p[2].values()), trees=len(p[3]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT/'analysis/v7')
    ap.add_argument('--verify-paths',action='store_true',help='Also run four short historical alternatives with the bundled competition referee.')
    args = ap.parse_args()
    g = read_replay(args.replay)
    frames, cfg = g['steps'],g['configuration']
    assert cfg['size'] == N and cfg['moveCost'] == cfg['smeltCost'] == 0
    assert g['rewards'] == [6027.01,2013.46] and len(frames)==300
    assert [board(frames[164][0]['observation']['players'],s)['cash'] for s in (1,0)] == [7.53,2079.3]
    assert [round(board(frames[164][0]['observation']['players'],s)['cargo'],1) for s in (1,0)] == [457.6,53.8]
    rows, deposits, casualties, changes = [],[],[],[]
    tree_origin = {}
    for t, fr in enumerate(frames):
        if t:
            flow, dep, dead, changed = cash_flows(frames[t-1],fr,cfg,tree_origin)
            deposits.extend(dep);casualties.extend(dead);changes.extend(changed)
        else:
            flow=[Counter(), Counter()]
        rows.append([dict(**board(fr[0]['observation']['players'],s), **flow[s]) for s in (0,1)])
    def span(a,b):
        state = []
        for s in (1,0):
            counts = Counter()
            for row in rows[a+1:b+1]:
                counts.update({k:v for k,v in row[s].items() if k not in ('cash','cargo','collectors','planters','trees')})
            state.append(dict(end=board(frames[b][0]['observation']['players'],s),
                              net=round(rows[b][s]['cash']-rows[a][s]['cash'],2),
                              flows={k:round(v,3) for k,v in sorted(counts.items())}))
        return dict(start=a,end=b,side_QQ_Master=state)
    snapshots = {}
    for t in (49,75,86,87,88,89,100,164,190,215,250,251,252,299):
        snapshots[t]=[rows[t][s] for s in (1,0)]
    probes=[]
    for t in (50,75,87,90,110,130,150,164,190,210,230,250):
        p=frames[t][0]['observation']['players'][QQ]
        for uid,(cell,cargo,kind) in p[2].items():
            if kind!='COLLECTOR' or cargo<30:continue
            probes.append(dict(step=t,uid=uid,cargo=round(cargo,2),nearest=min(distance(cell,b) for b in p[1].values()),
                               fixed=route_probe(frames,t,uid,False),all_neighbor=route_probe(frames,t,uid,True)))
    args.out.mkdir(parents=True,exist_ok=True)
    data=([dict(t=t,QQ=row[1],Master=row[0]) for t,row in enumerate(rows)])
    line_bytes=''.join(json.dumps(item,separators=(',',':'))+'\n' for item in data).encode()
    (args.out/'starvation-ledger.jsonl.gz').write_bytes(gzip.compress(line_bytes,mtime=0))
    report=dict(match=MATCH,replay=f'https://117.72.210.134/replay/{MATCH}/1',game=1,
                original_seats=['claude-AlphaCarbon-Master7sE','yan-QQ-v6'],official_result_seed=157458776,
                referee_randomSeed=cfg['randomSeed'],official_map_sha256='7a35c318027553b621f101698a6b2ce2f338a4022caed07a78b12d91f34c3b0e',
                source_players_sha256=hashlib.sha256(json.dumps([f[0]['observation']['players'] for f in frames],separators=(',',':')).encode()).hexdigest(),
                phase_order=['QQ','Master'],spans=[span(a,b) for a,b in ((0,25),(25,50),(50,75),(75,88),(88,164),(164,215),(215,252),(252,299),(0,164),(0,299))],
                checkpoint_flows=snapshots,route_checks={f'{a}-{b}':route_stats(frames,a,b) for a,b in ((50,87),(88,164),(164,215),(88,252))},
                route_probes=probes,route_limit=14,probes_note='Conditional on recorded trajectories of every other worker. fixed excludes actual enemy collector landings; all_neighbor excludes every one-step reply reachable by any opposing collector. Cash credits all cargo reaching own base including fatal collisions. Other players may react in a live branch; entered cells can pay tree tax.',
                two_step_certificates=two_step_receipts(frames,cfg),verified_fixed_paths=verify_paths(frames,cfg,probes) if args.verify_paths else [],
                events=dict(columns={'deposits':['arrival','cargo_owner_seat','id','amount','base_owner_seat','cell'],
                                     'casualties':['arrival','seat','id','cargo','destination','cell'],
                                     'tree_changes':['arrival','new_owner_seat','operation','id','cell','age']},
                            deposits=deposits,loaded_casualties=casualties,tree_changes=changes),
                max_single_step_cash_rounding=max(abs(row[s].get('rounding',0)) for row in rows for s in (0,1)))
    (args.out/'starvation.json').write_text(json.dumps(report,ensure_ascii=False,indent=1)+'\n')
    for tick in (164,):
        print('routes',[(p['uid'],p['fixed'],p['all_neighbor']) for p in probes if p['step']==tick])
    print('decision counts',report['route_checks'])
    print('cash error',report['max_single_step_cash_rounding'])
    print('our receipts',[(e[0],e[3],e[1]) for e in deposits if e[4]==QQ])

if __name__ == '__main__':
    main()
