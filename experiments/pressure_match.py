"""Play the pressure wrapper against its fixed v5 base; count occupancy after each real move."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena import parse_strategies, run_game
from benchmark import configure_engine


def run_observed(task):
    configure_engine()
    from zerosum_env.envs.carbon import carbon
    advance = carbon.compute_next_board
    early, total = [0, 0], [0, 0]
    zones = []

    def record(state, env):
        outcome = advance(state, env)
        board = state[0].observation
        n = env.configuration.size
        if not zones:
            for player in board.players:
                cells = set()
                for pos in player[1].values():
                    row, col = divmod(pos, n)
                    cells.update(((row+dr)%n)*n+(col+dc)%n for dr in range(-2,3)
                                 for dc in range(-2,3) if abs(dr)+abs(dc) <= 2)
                zones.append(cells)
        for i, player in enumerate(board.players):
            count = sum(t == 'COLLECTOR' and cargo < 1 and pos in zones[1-i]
                        for pos, cargo, t in player[2].values())
            total[i] += count
            if board.step <= 75:
                early[i] += count
        return outcome

    with patch.object(carbon, 'compute_next_board', record):
        result = run_game(task)
    result.update(early_empty_collectors_near_enemy_base=early, empty_collectors_near_enemy_base=total)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    parser.add_argument('--baseline', type=Path, default=ROOT/'opponents'/'yan-v2.py')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    strategies = parse_strategies([f'pressure={Path(__file__).with_name("pressure_opponent.py")}',
                                   f'control={args.baseline}'])
    tasks = [(pair, args.seed, 300) for pair in (strategies, strategies[::-1])]
    with args.output.open('x') as out, ProcessPoolExecutor(args.jobs) as pool:
        for result in pool.map(run_observed, tasks):
            line = json.dumps(result, ensure_ascii=False, allow_nan=False)
            out.write(line+'\n')
            out.flush()
            print(line, flush=True)
