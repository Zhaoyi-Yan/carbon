"""Fit visible economic features to future changes in cash. Keep entire matches together."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'v5'))
from features import _V5_DIST, _v5_home
import cash_projection as model

model._V5_DIST, model._v5_home = _V5_DIST, _v5_home
model.np, model.LAST_TURN = np, 298


def rows(path):
    states = [f[0]['observation'] for f in json.loads(gzip.decompress(path.read_bytes()))['steps']]
    money = [s['players'][0][0] - s['players'][1][0] for s in states]
    features, labels = [], []
    end = len(states) - 1
    for t in range(int(path.name[:4], 16) % 5, end, 5):
        features.append(model._v6_position(states[t]['players'], states[t]['trees'], t, 0))
        labels.append([money[min(end, t+n)] - money[t] for n in (16, 44, 88)])
    return np.stack(features), np.array(labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('history', type=Path)
    p.add_argument('online', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    online = sorted(a.online.glob('*.json.gz'))
    assert online
    order = lambda f: hashlib.sha256(('v6-income/' + f.name).encode()).digest()
    historical = sorted(a.history.glob('*.json.gz'), key=order)[:360]
    online = sorted(online, key=order)
    train, valid = historical[:300] + online[:2*len(online)//3], historical[300:] + online[2*len(online)//3:]
    assert not {f.name for f in train} & {f.name for f in valid}
    with ProcessPoolExecutor(3) as ex:
        prepared = dict(zip(train+valid, ex.map(rows, train+valid, chunksize=4)))
    def data(files):
        groups = [prepared[f] for f in files]
        return tuple(np.concatenate([g[i] for g in groups]) for i in (0,1))
    x, y = data(train)
    scale = np.maximum(x.std(0), .25)
    xc, yc = zip(*[(a-a.mean(0),b-b.mean(0)) for a,b in (prepared[f] for f in train)])
    design = np.concatenate([x, .75*np.concatenate(xc)])/scale
    labels = np.concatenate([y, .75*np.concatenate(yc)])
    weights = np.linalg.solve(design.T@design + np.eye(x.shape[1])*400., design.T@labels)/scale[:,None]
    output = {'horizons':[16,44,88], 'weights':(weights@np.array([.15,.60,.25])).tolist(),
              'training':[f.name for f in train], 'validation':[f.name for f in valid]}
    for key, files in (('historical', historical[300:]), ('online', online[2*len(online)//3:])):
        x,y = data(files)
        output[key] = {'states':len(x), 'persist_mae':np.abs(y).mean(0).round(2).tolist(),
                       'forecast_mae':np.abs(y-x@weights).mean(0).round(2).tolist()}
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    print(output['online'], flush=True)

if __name__ == '__main__':
    main()
