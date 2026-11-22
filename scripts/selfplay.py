"""Play a candidate against a frozen Carbon League strategy on both seats of every seed.

The default opponent is the submitted yan-v3 account strategy. Use yan-v1, yan-v2, yan-v3 or a
single-file Python strategy as the candidate or baseline. The arena writes one JSONL
record per game, with both names, file hashes, seat order, scores and timing.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
# Account handle -> immutable repository revision, verified source digest, shallow-clone copy.
FROZEN = {
    "yan-v3": ("791c2f23f7dfa46228f6578a5df8e3391a334a7a",
               "ee2b1a51039aaea356e28c6e326d3ec4eb4c06cda054808f068f132447add737",
               "opponents/yan-v3.py"),
    "yan-v2": ("e4c8cbc5f4e9abf7fa5e410fd4fb2ec31073824d",
               "0483eca2a5e8031270dfb89b87cd5d901ee81f823383f6094f8ed839dec6a7dd",
               "opponents/yan-v2.py"),
    "yan-v1": ("66cae433e3b36ccb5fcc95757963e62a7411fae9",
               "29aa472ef0484b65bb3ffdc85113aef2d1daa1e25377bfe5355a4d5e3567a11c",
               "opponents/local-before-apollo.py"),
}


def source_for(spec, repo=ROOT):
    if spec in FROZEN:
        revision, expected, fallback = FROZEN[spec]
        try:
            git = subprocess.run(["git", "-C", str(repo), "show", revision + ":submission.py"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            git = None
        if git is not None and git.returncode == 0:
            source = git.stdout
        else:
            path = repo / fallback
            source = path.read_bytes() if path.is_file() else b""
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"Cannot recover verified {spec} ({expected[:12]}). "
                             f"Fetch its published history with 'git fetch origin --tags'.")
        return spec, expected, source
    path = Path(spec)
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    for name, (_, expected, _) in FROZEN.items():
        if expected == digest:
            return name, digest, source
    return path.stem, digest, source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="working strategy file or frozen account name")
    parser.add_argument("--baseline", default="yan-v3", help="opponent file or frozen account name (default: yan-v3)")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="holdout map seeds; each runs twice")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True, help="new arena JSONL file")
    args = parser.parse_args(argv)
    try:
        candidate, opponent = [source_for(spec) for spec in (args.candidate, args.baseline)]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    cname, cdigest, candidate_bytes = candidate
    bname, bdigest, baseline_bytes = opponent
    if cdigest == bdigest:
        parser.error(f"Candidate and baseline are byte-identical ({cdigest[:12]}). "
                     "Choose an edited candidate or a different --baseline.")
    if cname == bname:
        cname += "-candidate"
    if any("=" in name for name in (cname, bname)):
        parser.error("Strategy filenames must not contain '='")
    with tempfile.TemporaryDirectory(prefix="carbon-selfplay-") as folder:
        cpath, bpath = Path(folder) / "candidate.py", Path(folder) / "baseline.py"
        cpath.write_bytes(candidate_bytes)
        bpath.write_bytes(baseline_bytes)
        command = [sys.executable, str(ROOT / "arena.py"), "--strategies", f"{cname}={cpath}",
                   f"{bname}={bpath}", "--seeds", *map(str, args.seeds), "--steps", str(args.steps),
                   "--jobs", str(args.jobs), "--output", str(args.output)]
        print(f"{cname} ({cdigest[:12]}) vs {bname} ({bdigest[:12]})", flush=True)
        return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
