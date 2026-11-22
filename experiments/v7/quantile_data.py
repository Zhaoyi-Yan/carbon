"""Align replayed commands and existing observer-only logits to an independent per-seat RNG tape."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from distill_data import configure_engine, load_manifest, local, public
from receipt_data import ACT

SEEDS = (37, 36, 38, 71)
def main(data):
    configure_engine()
    cache = data/'distill-cache'
    positive = json.loads((cache/'index.json').read_text())
    records = list(positive['records'])
    negative = data/'distill-negative'
    for r in json.loads((negative/'index.json').read_text()):
        records.append(dict(r, negative=True, fold='train', extra=True))
    by_game = defaultdict(list)
    for r in records:
        by_game[r['source'],r['id']].append(r)
    manifest,fold,ban = load_manifest(data/'distill-public')
    output = data/'quantile-cache'
    output.mkdir(exist_ok=True)
    for r in manifest:
        rows = by_game[r['source'],r['id']]
        todo = []
        for q in rows:
            key = f"{r['source']}-{r['id']}-{'other' if q['negative'] else 'v6'}.npz"
            q['tape'] = key
            if (output/key).exists():continue
            old = negative if q.get('extra') else cache
            with np.load(old/q['file']) as f:
                y,idx,uid = (f[k] for k in ('y','t','uid'))
            side = 1-r['seat'] if q['negative'] else r['seat']
            todo.append((q,side,idx,y,uid, [np.random.RandomState(s) for s in SEEDS], [], []))
        if not todo:continue
        frames = local(r['payload'],0) if r['source']=='local' else public(r['payload'],0)
        for turn,o,_,commands,_ in frames:
            if turn>max(v[2][-1] for v in todo):break
            for q,side,idx,y,uid,streams,u,cargo in todo:
                player=o['players'][side]; units=player[2]
                nbase=len(player[1]); n=len(units)
                draws=np.stack([s.random_sample(nbase+n)[nbase:] for s in streams],1)
                ix=np.flatnonzero(idx==turn)
                if not ix.size:continue
                assert len(ix)==n
                keys=list(units)
                assert np.array_equal(uid[ix],[int(k.split('-')[-1]) for k in keys])
                assert np.array_equal(y[ix],[ACT[commands[side].get(k) or 'NONE'] for k in keys])
                u.append(draws);cargo.extend(float(w[1]) for w in units.values())
        for q,side,idx,y,uid,streams,u,cargo in todo:
            assert len(cargo)==len(y)
            with (output/q['tape']).open('wb') as f:np.savez_compressed(f,u=np.concatenate(u),cargo=cargo)
    (output/'index.json').write_text(json.dumps(dict(records=records,seeds=SEEDS,excluded=sorted(ban),windows=positive['windows']),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);main(p.parse_args().data)
