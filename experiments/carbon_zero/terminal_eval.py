"""Evaluate saved terminal-RL arms on shared, independently seeded development games."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import mean, median

from pilot import Actor, FT3_SHA256, ROOT, STYLE, load_ft3, torch, write
from policy import actor_from_checkpoint
from benchmark import load_strategy
from scripts.selfplay import source_for
from terminal_games import Match, TEMPERATURE, keyed_seed, play

RIVALS = ('ZW0', 'ft3-generic-no-search', 'yan-v6-no-search')


def paired_map_panel(rival, count, replicate=False, namespace='terminal-rl-dev-v1', shared_maps=False):
    matches = []
    for i in range(count):
        layout = (keyed_seed(namespace, i, 'layout') if shared_maps
                  else keyed_seed(namespace, rival, i, 'layout')) & 0xffffffff
        for seat in (0, 1):
            keys = (namespace, rival, i, seat)
            matches.append(Match(layout, seat, keyed_seed(*keys, 'AA' if replicate else 'learner'),
                                 keyed_seed(*keys, 'opponent'), i))
    return matches


def opponent_for(ft3, out, name, default):
    if name == 'ZW0':
        return default, None, FT3_SHA256
    if name == 'ft3-generic-no-search':
        raw = load_ft3(ft3, full=True)
        raw._SID = raw._POLICY.sid('CL-top')
        raw._ROOT_TEMP = TEMPERATURE
        digest = FT3_SHA256
    else:
        _, digest, blob = source_for('yan-v6', ROOT)
        raw = load_strategy('terminal_rl_v6', out / 'frozen-v6.py', source=blob)
    raw._search = lambda _o, _c, _e, action, *args: action
    return None, raw, digest


def stats(rows):
    return dict(wins=sum(r['margin'] > 0 for r in rows), draws=sum(r['margin'] == 0 for r in rows),
                losses=sum(r['margin'] < 0 for r in rows),
                mean_margin=mean(r['margin'] for r in rows),
                median_margin=median(r['margin'] for r in rows),
                mean_utility=mean(r['utility'] for r in rows),
                full_games=sum(not r['elimination'] for r in rows),
                eliminations=sum(r['elimination'] for r in rows))


def paired(base, challenger):
    comparison = []
    reference = {(r['seed'], r['seat']): r for r in base}
    for game in challenger:
        other = reference[(game['seed'], game['seat'])]
        comparison.append(dict(seed=game['seed'], seat=game['seat'],
                               margin=game['margin']-other['margin'],
                               utility=game['utility']-other['utility']))
    layouts = sorted({row['seed'] for row in comparison})
    maps = [[mean(row[key] for row in comparison if row['seed'] == seed)
             for seed in layouts] for key in ('margin', 'utility')]
    rng = random.Random(473621)
    draws = [rng.choices(range(len(layouts)), k=len(layouts)) for _ in range(4000)]
    intervals = []
    for values in maps:
        samples = sorted(mean(values[i] for i in batch) for batch in draws)
        intervals.append([samples[100], samples[3899]])
    return dict(rows=comparison, mean_margin_delta=mean(maps[0]), mean_utility_delta=mean(maps[1]),
                map_bootstrap95_margin=intervals[0], map_bootstrap95_utility=intervals[1])


def main(a):
    os.environ['CARBON_DEBUG'] = '1'
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=True)
    teacher = load_ft3(a.ft3)
    style = teacher._POLICY.sid(STYLE)
    assert style == 122
    reference = Actor(teacher._POLICY, style).to(a.device).eval()
    refs = [opponent_for(a.ft3, a.out, name, reference) for name in RIVALS]
    protocol = dict(source_sha256=FT3_SHA256, style=style, temperature=TEMPERATURE, decoder='sequential',
                    shared_maps=a.shared_maps,
                    panels={rival: [m.__dict__ for m in paired_map_panel(
                        rival, a.maps, namespace=a.panel_id, shared_maps=a.shared_maps)]
                            for rival in RIVALS},
                    opponents={rival: ref[2] for rival, ref in zip(RIVALS, refs, strict=True)},
                    AA='same ZW0 states/opponent stream, independent learner stream')
    protocol_path = a.out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Development evaluation protocol changed')
    write(protocol_path, protocol)
    candidates = list(a.candidate)
    if not (a.out / 'ZW0.json').exists():
        candidates.insert(0, 'ZW0')
    if a.aa and not (a.out / 'AA.json').exists():
        candidates.insert(1, 'AA')
    baseline = {}
    for item in candidates:
        name, _, file = item.partition('=')
        actor = reference
        checkpoint_sha = None
        if file:
            blob = Path(file)
            checkpoint_sha = hashlib.sha256(blob.read_bytes()).hexdigest()
            actor, _ = actor_from_checkpoint(teacher, style, blob, a.device)
        elif name not in ('ZW0', 'AA'):
            raise ValueError(f'Missing checkpoint path for {name}')
        path = a.out / f'{name}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['checkpoint_sha256'] != checkpoint_sha:
                raise ValueError(f'Checkpoint digest changed for {name}')
            continue
        if name != 'ZW0' and not baseline:
            baseline = json.loads((a.out / 'ZW0.json').read_text())['rivals']
        report = {}
        for rival, (opponent, raw, _) in zip(RIVALS, refs, strict=True):
            if name == 'AA' and rival != 'ZW0':
                continue
            matches = paired_map_panel(rival, a.maps, replicate=(name == 'AA'),
                                       namespace=a.panel_id, shared_maps=a.shared_maps)
            played, timing = play(teacher, actor, opponent, matches, a.device, raw=raw)
            rows = [row for row, _, _ in played]
            entry = dict(overall=stats(rows), games=rows, costs=timing,
                         seats={str(i): stats([r for r in rows if r['seat'] == i]) for i in (0, 1)})
            if name != 'ZW0':
                entry['paired_ZW0'] = paired(baseline[rival]['games'], rows)
            report[rival] = entry
        write(path, dict(arm=name, checkpoint_sha256=checkpoint_sha, rivals=report))
        print('Evaluated', name, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ft3', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--candidate', action='append', default=[])
    p.add_argument('--aa', action='store_true')
    p.add_argument('--maps', type=int, default=4)
    p.add_argument('--panel-id', default='terminal-rl-dev-v1')
    p.add_argument('--shared-maps', action='store_true',
                   help='Use one map list for every rival (new panels; old panels stay rival-specific)')
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    main(p.parse_args())
