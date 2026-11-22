"""Paired CPU search audit on saved positions; no game is played."""
import argparse
import gzip
import importlib.util
import inspect
import json
from pathlib import Path
import time


def load(path, name, fixed):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if fixed:
        code = inspect.getsource(module._search)
        begin = code.index('    if _S_COST[0] > 0.0:')
        end = code.index('    stages =', begin)
        code = code[:begin] + '    most, D2, M = 20, min(D2, 7), 2\n' + code[end:]
        begin = code.index('        need = ',code.index('    def late('))
        end = code.index('\n    def over(', begin)
        code = code[:begin] + '        return False\n' + code[end:]
        exec(compile(code, str(path), 'exec'), vars(module))
    assert module._NET is module._RNET
    network = module._NET
    module.audit_counts = [0, 0]
    def counted(*args, **kwargs):
        module.audit_counts[0] += 1
        module.audit_counts[1] += len(args[0])
        return network(*args, **kwargs)
    module._NET = module._RNET = counted
    return module


def probe(model, replay, turn, seat):
    model._ST.clear()
    tracker = model.BeliefTracker(model._Bel, seat)
    previous = None
    for current, following in zip(replay['steps'][:turn], replay['steps'][1:]):
        players = current[0]['observation']['players']
        tracker.step(players, current[0]['observation']['step'], current[seat]['observation']['carbon'])
        tracker.set_actions(players, following[seat]['action'] or {})
        previous = players
    # Each position starts with the same belief state and same new style posterior.
    model._ST.update(tr=tracker, prev=previous, rng=model.np.random.RandomState(model._SEED))
    model._S_COST[:] = [.0015, .0004, .0015, .0004]
    model.audit_counts[:] = [0, 0]
    model._S_REC = []
    state = replay['steps'][turn]
    obs = {**state[0]['observation'], 'step': turn, 'player': seat, 'carbon': state[seat]['observation']['carbon']}
    start = time.perf_counter()
    action = model.agent(obs, replay['configuration'])
    elapsed = time.perf_counter() - start
    if not model._S_REC:
        return dict(seconds=elapsed, action=action, passes=model.audit_counts[0], rows=model.audit_counts[1])
    _, _, candidates, depths, _, chosen = model._S_REC[-1]
    width, depth, replies, _, _ = model._S_STATS['last']
    return dict(width=width, depth=depth, replies=replies, seconds=elapsed, action=action, candidate=chosen,
                passes=model.audit_counts[0], rows=model.audit_counts[1], evaluated=[i for i,x in enumerate(depths) if x])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('replay', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--fixed', action='store_true', help='20 proposals, 2 replies, 7 plies, no wall-time cutoff')
    p.add_argument('--turns', default='30,60,95,145,195,240')
    args = p.parse_args()
    with gzip.open(args.replay) as f:
        replay = json.load(f)
    mods = {'base':load(args.baseline,'base_speed_probe',args.fixed),
            'candidate':load(args.candidate,'test_speed_probe',args.fixed)}
    with args.output.open('x') as f:
        for i,turn in enumerate(map(int,args.turns.split(','))):
            record = {'seed':replay['configuration']['randomSeed'],'turn':turn,'seat':i%2,'fixed':args.fixed}
            for key in (('base','candidate') if i%2==0 else ('candidate','base')):
                record[key] = probe(mods[key],replay,turn,i%2)
            f.write(json.dumps(record)+'\n')
            f.flush()
            print(turn, {k:(record[k].get('depth'),record[k]['rows'],round(record[k]['seconds'],3)) for k in mods},flush=True)


if __name__ == '__main__':
    main()
