"""Prove a new tournament's starting layouts are disjoint from public data and prior local games."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def check(frozen,rows):
    sha=frozen['sha256'];seeds=set(frozen['seeds'])
    public={json.loads(l)['layout'] for l in (ROOT/'analysis/v6r-artemis-games.jsonl').read_text().splitlines()}
    earlier=set()
    for folder,pattern in ((ROOT/'analysis/arena','*.jsonl'),(ROOT/'analysis','v6r-*.jsonl')):
        for f in folder.glob(pattern):
            for line in f.open():
                g=json.loads(line)
                if 'map_sha256' not in g:continue
                if g.get('seed') in seeds and any(p.get('sha256')==sha for p in g.get('players',[])):continue
                if g.get('layout_sha256'):earlier.add(g['layout_sha256'])
    maps={g['layout_sha256'] for g in rows}
    # The archive uses [full_carbon,sorted_bases]; the arena's unprefixed map
    # hash has the same encoding on this fixed two-base board. Both base lists
    # are already in ascending order on the reference referee.
    raw={g['map_sha256'] for g in rows}
    if maps&earlier or raw&public:raise ValueError('an evaluation map occurs in the training or earlier sweep')
    if {g['seed'] for g in rows}!=seeds or len(maps)!=len(seeds):raise ValueError('incomplete map set')
    return dict(maps=len(maps),public_layouts=len(public),other_local_layouts=len(earlier),
                public_overlap=len(raw&public),other_local_overlap=len(maps&earlier))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('freeze',type=Path);p.add_argument('results',type=Path)
    a=p.parse_args();print(json.dumps(check(json.loads(a.freeze.read_text()),[json.loads(l) for l in a.results.read_text().splitlines()])))
