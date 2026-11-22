import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import selfplay


class SelfPlayTests(unittest.TestCase):
    def test_frozen_revision_ignores_working_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "checkout"
            checkout.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-C", str(checkout), *args])
            git("init", "--quiet")
            frozen = b"def agent(o,c): return {}\n"
            path = checkout / "submission.py"
            path.write_bytes(frozen)
            git("add", "submission.py")
            git("-c", "user.name=test", "-c", "user.email=test@example.invalid",
                "commit", "--quiet", "-m", "snapshot")
            revision = git("rev-parse", "HEAD").decode().strip()
            digest = hashlib.sha256(frozen).hexdigest()
            path.write_text("# different candidate\n")
            with patch.dict(selfplay.FROZEN, {"fixture": (revision, digest, "submission.py")}):
                self.assertEqual(selfplay.source_for("fixture", repo=checkout), ("fixture", digest, frozen))
            self.assertEqual(path.read_text(), "# different candidate\n")

    def test_shallow_fallback_must_match_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            path = repo / "submission.py"
            source = b"def agent(o,c): return {}\n"
            digest = hashlib.sha256(source).hexdigest()
            with patch.dict(selfplay.FROZEN, {"fixture": ("missing-ref", digest, "submission.py")}):
                path.write_bytes(source)
                self.assertEqual(selfplay.source_for("fixture", repo=repo), ("fixture", digest, source))
                path.write_text("not the verified baseline")
                with self.assertRaisesRegex(ValueError, "verified fixture"):
                    selfplay.source_for("fixture", repo=repo)

    def test_default_baseline_detects_identical_renamed_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "renamed.py"
            source = (selfplay.ROOT / "example.py").read_bytes()
            candidate.write_bytes(source)
            frozen = ("missing-fixture-ref", hashlib.sha256(source).hexdigest(), "example.py")
            output = Path(tmp) / "games.jsonl"
            with (patch.dict(selfplay.FROZEN, {"yan-v2": frozen}),
                  contextlib.redirect_stderr(io.StringIO()) as stderr,
                  self.assertRaises(SystemExit) as exc):
                selfplay.main(["--candidate", str(candidate), "--seeds", "42", "--output", str(output)])
            self.assertEqual(exc.exception.code, 2)
            self.assertIn("byte-identical", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_custom_baseline_delegates_to_paired_arena(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / "left.py", root / "right.py"
            left.write_text("def agent(observation, configuration): return {}\n")
            right.write_text("# baseline\ndef agent(observation, configuration): return {}\n")
            result = root / "paired.jsonl"
            subprocess.run([sys.executable, str(selfplay.ROOT / "scripts/selfplay.py"),
                            "--candidate", str(left), "--baseline", str(right), "--steps", "3",
                            "--seeds", "99", "--output", str(result)], capture_output=True, check=True)
            games = [json.loads(line) for line in result.read_text().splitlines()]
            self.assertEqual([g["status"] for g in games], ["completed", "completed"])
            self.assertEqual([[p["name"] for p in g["players"]] for g in games],
                             [["left", "right"], ["right", "left"]])
            self.assertEqual(games[0]["map_sha256"], games[1]["map_sha256"])


if __name__ == "__main__":
    unittest.main()
