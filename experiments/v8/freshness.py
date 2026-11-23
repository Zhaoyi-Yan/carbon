"""Check preselected layouts against archived public games and prior local arena matches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark import configure_engine, new_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--previous', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    configure_engine()
    known = set(json.loads((ROOT/'analysis/v7/public-layouts.json').read_text()))
    prior_seeds = set()
    for path in (ROOT/'analysis').rglob('*.jsonl'):
        with path.open() as records:
            for line in records:
                entry = json.loads(line)
                if isinstance(entry, dict) and entry.get('layout_sha256'):
                    known.add(entry['layout_sha256'])
                    prior_seeds.add(entry.get('seed'))
    for path in args.previous:
        extra = json.loads(path.read_text())['maps']
        known.update(extra.values())
        prior_seeds.update(map(int,extra))
    protocol = json.loads((args.run/'protocol.json').read_text())
    layouts = {}
    for seed in protocol['screening'] + protocol['confirmation']:
        state, env = new_game(seed,300)
        board = state[0].observation
        layout = [env.configuration.size, board.full_carbon, [sorted(p[1].values()) for p in board.players]]
        digest = hashlib.sha256(json.dumps(layout,separators=(',',':')).encode()).hexdigest()
        if digest in known or digest in layouts.values() or seed in prior_seeds:
            raise ValueError(f'Previously used map {seed}')
        layouts[seed] = digest
    (args.run/'freshness.json').write_text(json.dumps(dict(prior_layouts=len(known),maps=layouts),indent=2)+'\n')


if __name__=='__main__':
    main()
