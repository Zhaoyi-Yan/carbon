"""Check the optional embedded fast simulator against the vendored referee."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from benchmark import load_strategy
from env import Game
from policy import MOVES, HIRES


def extract(ref, fast):
    ob=ref.state[0].observation
    return fast.State(ref.turn,list(ob.full_carbon),copy.deepcopy(ob.players),
                      copy.deepcopy(ob.trees),ref.ids.copy())


def state_data(state):
    return dict(players=state.players, stock=state.carbon, trees=state.trees_meta,
                ids=state.counters, step=state.step)


def advance(fast, game, state, actions, seed):
    before = copy.deepcopy(state_data(state))
    candidate = fast.next_state(state, actions, game.config)
    game.step(actions)
    expected = state_data(extract(game, fast))
    actual = state_data(candidate)
    if actual != expected:
        return candidate, dict(seed=seed, actions=actions, before=before,
                               expected=expected, actual=actual)
    return candidate, None


def compare(fast, count=10000, minimum_games=100):
    rng=np.random.default_rng(31024)
    start=0
    finished=0
    for seed in range(3102400,3102900):
        g=Game(seed)
        st=extract(g,fast)
        for _ in range(299):
            obs=g.observations()
            commands=[]
            for i,o in enumerate(obs):
                money,bases,workers,_=o['players'][i]
                actions={}
                for u in bases:
                    action=rng.choice(HIRES) if money>=30 else None
                    if action is not None:actions[u]=action
                for u in workers:
                    action=rng.choice(MOVES)
                    if action is not None:actions[u]=action
                commands.append(actions)
            st, mismatch = advance(fast, g, st, commands, seed)
            start+=1
            if mismatch:
                return dict(passed=False, steps_compared=start, **mismatch)
            if g.done:
                finished+=1
                break
        if start>=count and finished>=minimum_games:
            break
    return dict(passed=start>=count and finished>=minimum_games,
                steps_compared=start,finished_games=finished)


def compare_arena(fast, path):
    """Reconstruct action-recorded arena games by seed under the same referee."""
    steps = 0
    games = []
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            if row['status'] != 'completed' or not row.get('actions_and_post_cash'):
                continue
            seed = row['seed']
            game = Game(seed, row['episode_steps'])
            state = extract(game, fast)
            for actions, cash in row['actions_and_post_cash']:
                state, mismatch = advance(fast, game, state, actions, seed)
                steps += 1
                if mismatch:
                    return dict(passed=False, steps_compared=steps, **mismatch)
                assert np.allclose(game.cash(), cash, atol=1e-8, rtol=0), (seed, game.turn)
            assert game.done and [p.reward for p in game.state] == row['scores'], seed
            games.append(dict(seed=seed, layout_sha256=row['layout_sha256'],
                              strategy_sha256=[p['sha256'] for p in row['players']]))
    return dict(passed=bool(games), steps_compared=steps, finished_games=len(games), games=games)


if __name__=='__main__':
    source=ROOT/'opponents/claude-AlphaCarbon-Master3s2.py'
    module=load_strategy('fast_candidate',source,source.read_text().rsplit('\n_net()',1)[0].encode())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--arena-replay', type=Path)
    args = parser.parse_args()
    result = compare_arena(module, args.arena_replay) if args.arena_replay else compare(module)
    result['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
