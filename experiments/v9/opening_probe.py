"""Check the reactive-opening candidate on real initial observations before playing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark import configure_engine, load_strategy, new_game, observe, validate_actions  # noqa: E402


def load(path, name):
    source = path.read_bytes()
    return load_strategy(name, path, source), hashlib.sha256(source).hexdigest()


def classify(actions):
    values = list(actions.values())
    return dict(total=len(values), collectors=values.count('RECCOLLECTOR'),
                planters=values.count('RECPLANTER'), actions=actions)


def main(a):
    configure_engine()
    from zerosum_env.envs.carbon import carbon
    cand = [load(a.candidate, f'cand{seat}') for seat in (0, 1)]
    base = [load(a.baseline, f'base{seat}') for seat in (0, 1)]
    rows = []
    for seed in a.seeds:
        state, env = new_game(seed, 300)
        config = env.configuration
        entry = {'seed': seed, 'step0': {}, 'step1': {}}
        for seat in (0, 1):
            obs = observe(state, seat, 0)
            ca = cand[seat][0].agent(obs, config)
            ba = base[seat][0].agent(obs, config)
            validate_actions(ca, state[0].observation.players[seat])
            validate_actions(ba, state[0].observation.players[seat])
            entry['step0'][str(seat)] = {'candidate': classify(ca), 'baseline': classify(ba)}
            state[seat].action = {}
        carbon.compute_next_board(state, env)
        for seat in (0, 1):
            obs = observe(state, seat, 1)
            ca = cand[seat][0].agent(obs, config)
            validate_actions(ca, state[0].observation.players[seat])
            entry['step1'][str(seat)] = {'candidate': classify(ca)}
            state[seat].action = ca
        rows.append(entry)
    checks = {
        'candidate_waits_at_step0': all(r['step0'][s]['candidate']['total'] == 0 for r in rows for s in ('0', '1')),
        'baseline_recruits_at_step0': all(r['step0'][s]['baseline']['total'] >= 2 for r in rows for s in ('0', '1')),
        'candidate_recruits_at_step1': all(r['step1'][s]['candidate']['collectors'] >= 1 and
                                           r['step1'][s]['candidate']['planters'] >= 1
                                           for r in rows for s in ('0', '1')),
        'search_engine_loaded': bool(cand[0][0]._SEARCH_OK),
    }
    out = dict(candidate=dict(path=str(a.candidate), sha256=cand[0][1], bytes=a.candidate.stat().st_size),
               baseline=dict(path=str(a.baseline), sha256=base[0][1]), seeds=list(a.seeds),
               checks=checks, rows=rows)
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(checks, indent=1))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate', type=Path)
    p.add_argument('baseline', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    main(p.parse_args())
