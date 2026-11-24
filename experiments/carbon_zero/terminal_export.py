"""Export a checked terminal-RL checkpoint with the frozen pure decoder."""
import argparse
import hashlib
from pathlib import Path

from checkpoints import Actor, STYLE, check_export_actions, load_ft3, load_strategy, torch
from export import build
from pilot import write
from terminal_games import TEMPERATURE


def main(a):
    torch.set_num_threads(2)
    teacher = load_ft3(a.ft3)
    style = teacher._POLICY.sid(STYLE)
    assert style == 122
    actor = Actor(teacher._POLICY, style).eval()
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    actor.load_state_dict(state['parameters'])
    code = build(a.ft3.read_text(), teacher._POLICY, actor, style, TEMPERATURE, 'pure', True)
    a.output.write_text(code)
    module = load_strategy('terminal_rl_export', a.output)
    verified = check_export_actions(teacher, actor, module, TEMPERATURE, True)
    write(a.output.with_suffix('.json'), dict(
        E=state['E'], actions_verified=verified, size_bytes=a.output.stat().st_size,
        source_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('ft3', 'checkpoint', 'output'):
        p.add_argument(name, type=Path)
    main(p.parse_args())
