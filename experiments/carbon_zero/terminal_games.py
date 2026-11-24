"""Play complete simultaneous games with independent, reproducible seat streams."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import time

import numpy as np
import torch

from env import Game, potential, reward, utility
from policy import Encoder, action_logits, commands, coordinated_sample, slots
from ppo import joint_logprob

TEMPERATURE = 0.4
HORIZON = 299


def keyed_seed(*keys):
    payload = json.dumps(keys, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


@dataclass(frozen=True)
class Match:
    seed: int
    seat: int
    learner_rng: int
    opponent_rng: int
    group: int


def group_matches(namespace, batch, group_size=4, wide=False):
    """Two layouts x two seats x group_size replicates, or eight layouts x two seats.

    The wide allocation keeps the pair=0,1 / member=0 layouts and random streams
    and adds pair=2..7, still with two balanced seats and 16 games per batch.
    """
    pairs, members = (8, 1) if wide else (2, group_size)
    matches = []
    for pair in range(pairs):
        layout = keyed_seed(namespace, batch, pair, 'layout') & 0xffffffff
        for seat in (0, 1):
            group = 2 * pair + seat
            for member in range(members):
                keys = (namespace, batch, group, member)
                matches.append(Match(layout, seat, keyed_seed(*keys, 'learner'),
                                     keyed_seed(*keys, 'opponent'), group))
    return matches


@dataclass
class RunningGame:
    spec: Match
    game: Game
    trackers: tuple
    rng: tuple
    layout: str
    initial_potential: float
    rows: list = field(default_factory=list)
    action_trace: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    shaped_return: float = 0.0


def opening_digest(game):
    board = game.state[0].observation
    layout = [game.config['size'], board.full_carbon,
              [sorted(team[1].values()) for team in board.players]]
    return hashlib.sha256(json.dumps(layout, separators=(',', ':')).encode()).hexdigest()


def pack(rows, device):
    features, cells, masks, names = zip(*rows, strict=True)
    arrays = tuple(np.stack(field) for field in (features, cells, masks))
    return arrays, tuple(torch.as_tensor(a, device=device) for a in arrays), names


def choose(net, inputs, rng, capture):
    arrays, (features, cells, masks), names = inputs
    spatial = net.spatial(features)
    scores = action_logits(net.workers(spatial), net.bases(spatial), cells, masks, TEMPERATURE)
    noise = np.stack([r.gumbel(size=(12, 5)) for r in rng])
    picked, conditional = coordinated_sample(scores, None, arrays[1], arrays[2], noise=noise)
    moves = [commands(uid, indices) for uid, indices in zip(names, picked, strict=True)]
    if capture == 'policy':
        effective = torch.as_tensor(conditional, device=scores.device)
        old = joint_logprob(scores.masked_fill(~effective, -1.e9),
                            torch.as_tensor(picked, device=scores.device)).cpu().numpy()
        return moves, list(zip(arrays[0], arrays[1], conditional, picked, old, strict=True))
    if capture == 'critic':
        return moves, net.value_input(spatial).cpu().numpy()
    return moves, [None] * len(moves)


@torch.inference_mode()
def play(teacher, learner, opponent, matches, device, *, capture=None, steps=300, raw=None):
    """Capture legal learner data and keep environment transitions at the referee."""
    native_states = {m: {} for m in matches} if raw is not None else None
    timers = dict(encode=0., inference=0., referee=0.)
    start = time.perf_counter()
    active = []
    for spec in matches:
        game = Game(spec.seed, steps)
        encoders = tuple(Encoder(teacher, seat) for seat in (spec.seat, 1 - spec.seat))
        rng = tuple(np.random.default_rng(s) for s in (spec.learner_rng, spec.opponent_rng))
        initial_cash = game.cash()[::1 if spec.seat == 0 else -1]
        active.append(RunningGame(spec, game, encoders, rng, opening_digest(game),
                                  potential(initial_cash)))
    finished = []
    calls = 0
    while active:
        now = time.perf_counter()
        observations = []
        inputs = [[], []]
        for p in active:
            both = p.game.observations()
            ours = (both[p.spec.seat], both[1 - p.spec.seat])
            observations.append(ours)
            for role, (track, obs) in enumerate(zip(p.trackers, ours, strict=True)):
                if role == 1 and raw is not None:
                    continue
                names, cells, mask = slots(obs, p.game.config)
                inputs[role].append((track.encode(obs), cells, mask, names))
        tensors = [pack(rows, device) for rows in inputs if rows]
        timers['encode'] += time.perf_counter() - now

        now = time.perf_counter()
        learner_moves, samples = choose(learner, tensors[0], [p.rng[0] for p in active], capture)
        if raw is None:
            replies, _ = choose(opponent, tensors[1], [p.rng[1] for p in active], None)
            calls += 2
        else:
            replies = []
            for p, obs in zip(active, observations, strict=True):
                raw._ST = native_states[p.spec]
                raw._SEED = p.spec.opponent_rng & 0xffffffff
                replies.append(raw.agent(obs[1], p.game.config))
                native_states[p.spec] = raw._ST
            calls += 1
        timers['inference'] += time.perf_counter() - now

        pending = []
        now = time.perf_counter()
        for i, (p, obs, own, other, sample) in enumerate(zip(
                active, observations, learner_moves, replies, samples, strict=True)):
            game, seat = p.game, p.spec.seat
            before = game.cash()
            p.gaps.append(float(before[seat] - before[1 - seat]))
            sent = (own, other) if seat == 0 else (other, own)
            if capture == 'policy':
                p.rows.append(sample)
                names = tensors[0][2][i]
                p.action_trace.append((names, sent))
            elif capture == 'critic':
                gap = before[seat] - before[1 - seat]
                diagnostic_input = tensors[0][0][0][i].copy() if game.turn in (40, 140, 240) else None
                p.rows.append((sample, gap, game.turn, diagnostic_input))
            p.trackers[0].commit(obs[0], own)
            if raw is None:
                p.trackers[1].commit(obs[1], other)
            eliminated, scores = game.step(sent)
            p.shaped_return += reward(before, game.cash(), game.done, scores, eliminated, seat)
            if not game.done:
                pending.append(p)
                continue

            margin = scores[seat] - scores[1 - seat]
            terminal_u = utility(margin, eliminated)
            # One common initial offset per group. This also checks early finishes.
            if abs(p.shaped_return - (terminal_u - p.initial_potential)) > 1e-9:
                raise AssertionError('The full shaped return did not telescope')
            row = dict(**asdict(p.spec), layout_sha256=p.layout, turns=game.turn,
                       scores=scores, cash=game.cash(), margin=margin, utility=terminal_u,
                       elimination=eliminated, gaps=list(p.gaps))
            finished.append((row, p.rows, p.action_trace))
        timers['referee'] += time.perf_counter() - now
        active = pending
    elapsed = time.perf_counter() - start
    return sorted(finished, key=lambda item: (item[0]['group'], item[0]['learner_rng'])), {
        **timers, 'elapsed': elapsed, 'forwards': calls
    }
