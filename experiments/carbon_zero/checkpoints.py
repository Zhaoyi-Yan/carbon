"""Evaluate PPO exports on a frozen opponent outside the training loop."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark import configure_engine,limit_math_threads,load_strategy

limit_math_threads()
configure_engine()
import numpy as np
import torch

from arena import Strategy,run_game
from env import Game
from evaluate import games,seeds
from scripts.selfplay import source_for
from export import build
from policy import Actor, Encoder, STYLE, commands, coordinated_sample, load_ft3, logits, sample, slots


def capture(panel, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(panel, indent=2) + '\n')
    temporary.replace(path)


def run_search_panel(out, checkpoint, refs):
    if set(refs) != {'ZW0', 'ZW1'}:
        raise ValueError('The pure panel must contain both frozen and learned actors')
    v8_code = subprocess.check_output(
        ['git', 'show', 'v11-yan-v8-elite-refit:submission.py'], cwd=ROOT)
    v8_sha = hashlib.sha256(v8_code).hexdigest()
    assert v8_sha == 'a87f8e6b83656c771cb5a25d6b6f383b245127e7c782e4c6c7c4a0a2cc15d396'
    v8_path = out / 'frozen-v8.py'
    v8_path.write_bytes(v8_code)
    opponent = Strategy('yan-v8', v8_path, v8_sha)
    seed = seeds(f'search-{checkpoint}', 1)[0]
    for role, modes in refs.items():
        search = modes['search']
        path = out / f'zero-{role}-search-{checkpoint}.py'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == search['sha256']
        if 'games' in search:
            assert [(game['seed'], game['seat']) for game in search['games']] == [(seed, 0), (seed, 1)]
            continue
        candidate = Strategy(role, path, search['sha256'])
        matches = [
            ([candidate, opponent], seed, 300, True),
            ([opponent, candidate], seed, 300, True),
        ]
        started = time.perf_counter()
        with ProcessPoolExecutor(2) as pool:
            search['games'] = list(pool.map(run_game, matches))
        search['seconds'] = time.perf_counter() - started
        search['size_bytes'] = path.stat().st_size
        for seat, game in enumerate(search['games']):
            game['seat'] = seat


@torch.inference_mode()
def check_export_actions(teacher, actor, module, temperature, coordinated):
    game = Game(841273)
    trackers = [Encoder(teacher, seat) for seat in (0, 1)]
    streams = [np.random.default_rng(37 + seat) for seat in (0, 1)]
    native_state = [({}, None, None), ({}, None, None)]
    compared = 0
    for _ in range(32):
        actions = []
        for seat, obs in enumerate(game.observations()):
            names, cells, legal = slots(obs, game.config)
            features = trackers[seat].encode(obs)
            scores, _ = logits(actor, torch.as_tensor(features[None]),
                               torch.as_tensor(cells[None]), torch.as_tensor(legal[None]),
                               temperature)
            if coordinated:
                choices, _ = coordinated_sample(scores, streams[seat], cells[None], legal[None])
            else:
                choices = sample(scores, streams[seat])
            action = commands(names, choices[0])
            # The exported agent normally has its own interpreter per seat.
            # Swap its state and random stream to reproduce that isolation.
            module._ST, module._ZTURN, module._ZRNG = native_state[seat]
            native = module.agent(obs, game.config)
            native_state[seat] = module._ST, module._ZTURN, module._ZRNG
            assert action == native, (game.turn, seat, action, native)
            trackers[seat].commit(obs, action)
            actions.append(action)
            compared += 1
        game.step(actions)
        if game.done:
            break
    return compared


def main(ft3, out, checkpoint, full):
    torch.set_num_threads(2)
    dev='mps' if torch.backends.mps.is_available() else 'cpu'
    teacher=load_ft3(ft3)
    style=teacher._POLICY.sid(STYLE)
    base=Actor(teacher._POLICY,style)
    learned=copy.deepcopy(base)
    learned.load_state_dict(torch.load(out/f'checkpoint-{checkpoint}.pt',map_location='cpu',weights_only=True)['parameters'])
    temp=json.loads((out/'calibration.json').read_text())['temperature']
    coordinated=json.loads((out/'protocol.json').read_text())['training'].get('decoder')=='sequential'
    source=ft3.read_text()
    refs={}
    opponent_name, opponent_sha, opponent_code=source_for('yan-v6',ROOT)
    external_path=out/'frozen-v6.py'
    external_path.write_bytes(opponent_code)
    older=load_strategy('zero_v6_external',external_path)
    older._search=lambda o,c,e,action,*args:action
    other=load_ft3(ft3,full=True)
    other._SID=other._POLICY.sid('CL-top')
    other._ROOT_TEMP=.4
    other._search=lambda _o,_c,_e,action,*args:action
    maps=seeds(f'external-policy-{checkpoint}',2)
    other_maps=seeds(f'external-v6-{checkpoint}',2)
    for role,net in (('ZW0',base),('ZW1',learned)):
        refs[role]={}
        for mode in ('pure','search'):
            code=build(source,teacher._POLICY,net,style,temp,mode,coordinated)
            path=out/f'zero-{role}-{mode}-{checkpoint}.py'
            path.write_text(code)
            module=load_strategy('export_'+role+'_'+mode,path)
            game=Game(841271)
            x=Encoder(teacher,0).encode(game.observations()[0])
            native=module._ZERO_ACTOR.forward(x.astype(np.float32)*.01,style)
            with torch.inference_mode():
                w,b,_=net(torch.as_tensor(x[None]))
            errors=[float(np.max(np.abs(v[0].numpy()-r))) for v,r in zip((w,b),native,strict=True)]
            assert max(errors)<.003,errors
            refs[role][mode]=dict(parity=errors,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if mode=='pure':
                refs[role][mode]['verified_actions']=check_export_actions(
                    teacher, net, module, temp, coordinated)
                refs[role][mode]['size_bytes']=path.stat().st_size
            if mode=='search':
                assert module._SEARCH_OK
                original_raw=teacher._POLICY.forward(x.astype(np.float32)*.01,style)
                assert all(np.array_equal(a,b) for a,b in zip(original_raw,module._POLICY.forward(x.astype(np.float32)*.01,style)))
                for seat in (0,1):
                    action=module.agent(game.observations()[seat],game.config)
                    from benchmark import validate_actions
                    validate_actions(action,game.observations()[seat]['players'][seat])
                refs[role][mode]['verified_legal_roots']=2
        net.to(dev)
        panels={}
        for name,rival,paired_maps,digest in (('ft3-generic-no-search',other,maps,hashlib.sha256(ft3.read_bytes()).hexdigest()),
                ('yan-v6-no-search',older,other_maps,opponent_sha)):
            rows=[]
            for seed in paired_maps:
                for seat in (0,1):
                    trial=games(teacher,net,None,[seed],temp,.4,dev,raw=(rival,seat),coordinated=coordinated)
                    rows.extend(trial['games'])
            panels[name]=dict(source_sha256=digest,games=rows)
        net.cpu()
        refs[role]['pure']['external']=panels
    path = out / f'external-{checkpoint}.json'
    capture(refs, path)
    if full:
        run_search_panel(out, checkpoint, refs)
        capture(refs, path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('ft3',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('checkpoint',type=int)
    search = p.add_mutually_exclusive_group()
    search.add_argument('--search-games', action='store_true')
    search.add_argument('--search-only', action='store_true',
                        help='Add complete search games to existing pure evaluation')
    a=p.parse_args()
    if a.search_only:
        path = a.out / f'external-{a.checkpoint}.json'
        refs = json.loads(path.read_text())
        run_search_panel(a.out, a.checkpoint, refs)
        capture(refs, path)
    else:
        main(a.ft3,a.out,a.checkpoint,a.search_games)
