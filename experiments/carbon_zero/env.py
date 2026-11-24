"""Authoritative simultaneous Carbon transitions with isolated ID counters."""
import math

from benchmark import configure_engine, new_game, observe, validate_actions

configure_engine()
from zerosum_env.envs.carbon import carbon, idgen


def utility(delta, eliminated=False, beta=.5, scale=1500.):
    sign = (delta > 0) - (delta < 0)
    return float(sign if eliminated else beta * sign + (1. - beta) * math.tanh(delta / scale))


class Game:
    def __init__(self, seed, steps=300):
        self.state, self.referee = new_game(seed, steps)
        self.ids = idgen._id_dict.copy()
        self.steps = steps
        self.turn = 0
        self.done = False

    @property
    def config(self):
        return {k: v for k, v in self.referee.configuration.items() if k != 'randomSeed'}

    def observations(self):
        return tuple(observe(self.state, seat, self.turn) for seat in (0, 1))

    def cash(self):
        return tuple(team[0] for team in self.state[0].observation.players)

    def step(self, actions):
        if self.done:
            raise ValueError('Finished game')
        players = self.state[0].observation.players
        for seat, action in enumerate(actions):
            validate_actions(action, players[seat])
        for actor, action in zip(self.state, actions, strict=True):
            actor.action = action
        idgen._id_dict.clear()
        idgen._id_dict.update(self.ids)
        carbon.compute_next_board(self.state, self.referee)
        self.ids = idgen._id_dict.copy()
        self.turn += 1
        eliminated = any(actor.status != 'ACTIVE' for actor in self.state)
        self.done = eliminated or self.turn == self.steps - 1
        return eliminated, tuple(actor.reward for actor in self.state)


def potential(cash, alpha=.5):
    return alpha * math.tanh((cash[0] - cash[1]) / 1000.)


def reward(before, after, done, scores, eliminated, seat):
    shaped = -potential(before)
    if done:
        shaped += utility(scores[0] - scores[1], eliminated)
    else:
        shaped += potential(after)
    return shaped if seat == 0 else -shaped
