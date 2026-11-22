"""Extract a portable public replay, verify every referee transition, and restore turn 190."""
import argparse
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "analysis" / "v7"
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "v6r")]
from benchmark import configure_engine, limit_math_threads, new_game, observe
from arena import engine_fingerprint
from close_replay import archive_state, restore

BRANCH = 190
INPUT = OUT / "branch-original.json"
SNAPSHOT = OUT / "branch-190.json"


def compact(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(compact(data) + "\n")


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fields(observations, rewards):
    board, other = observations
    return [board["players"], board["trees"], board["full_carbon"], board["carbon"], other["carbon"], rewards]


def extract(html):
    raw = html.read_bytes()
    text = (gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw).decode()
    env = json.JSONDecoder().raw_decode(text.split("window.kaggle =", 1)[1].lstrip())[0]["environment"]
    frames = env["steps"]
    _, _ = new_game(env["configuration"]["randomSeed"], len(frames))
    from zerosum_env.envs.carbon import idgen
    initial = dict(seats=[dict(observation=s["observation"], status=s["status"], reward=s["reward"],
                               action=s["action"] or {}) for s in frames[0]], id_counters=dict(idgen._id_dict))
    assert all(not p[2] and not p[3] for p in initial["seats"][0]["observation"]["players"])
    assert all(v == len(initial["seats"][0]["observation"]["players"][i][1]) for i,v in enumerate(initial["id_counters"].values()))
    return dict(match_id="03100a95e8647c51a4d642fe08d856dd", game=1, seats=["QQ-v6", "Artemis"],
                source_gzip_sha256=hashlib.sha256(raw).hexdigest(), configuration=env["configuration"], initial=initial,
                actions=[[s["action"] or {} for s in f] for f in frames[1:]],
                expected=[digest(fields([s["observation"] for s in f], [s["reward"] for s in f])) for f in frames],
                final_rewards=[s["reward"] for s in frames[-1]])


def board_digest(state):
    return digest(fields([s.observation for s in state], [s.reward for s in state]))


def step(state, env, actions):
    from zerosum_env.envs.carbon import carbon
    for s, a in zip(state, actions):
        s.action = a
    carbon.compute_next_board(state, env)


def prefix(data):
    """Only historical own decisions and seat-specific observations enter teacher forcing."""
    state, env = restore(data["initial"], data["configuration"])
    frames = []
    for t, actions in enumerate(data["actions"][:BRANCH]):
        frames.append(([observe(state, p, t) for p in (0, 1)], actions))
        step(state, env, actions)
    return frames


def verify(data):
    from zerosum_env.envs.carbon import carbon
    cfg = data["configuration"]
    state, env = restore(data["initial"], cfg)
    snapshot = None
    for t, expected in enumerate(data["expected"]):
        actual = board_digest(state)
        if actual != expected:
            raise ValueError(f"full replay diverged at frame {t}: {actual} != {expected}")
        if t == BRANCH:
            snapshot = archive_state(state)
        if t < len(data["actions"]):
            step(state, env, data["actions"][t])
    assert snapshot is not None
    final = digest(archive_state(state))
    state, env = restore(snapshot, cfg)
    for t, actions in enumerate(data["actions"][BRANCH:], BRANCH + 1):
        step(state, env, actions)
        if board_digest(state) != data["expected"][t]:
            raise ValueError(f"restoration diverged at frame {t}")
    assert digest(archive_state(state)) == final
    return dict(step=BRANCH, snapshot=snapshot, configuration=cfg, verified_frames=len(data["expected"]),
                verified_restored_steps=len(data["actions"]) - BRANCH, final_rewards=data["final_rewards"],
                referee_sha256=engine_fingerprint(carbon), final_state_and_counters_sha256=final,
                board_sha256=data["expected"][BRANCH], representation="Full stock is referee-only; both fog arrays checked each step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, help="Read original official gzip/HTML and reconstruct portable replay.")
    args = parser.parse_args()
    limit_math_threads()
    configure_engine()
    data = extract(args.html) if args.html else json.loads(INPUT.read_text())
    checkpoint = verify(data)
    if args.html:
        write(INPUT, data)
    write(SNAPSHOT, checkpoint)
    print({k: v for k, v in checkpoint.items() if k not in ("snapshot", "configuration")})
