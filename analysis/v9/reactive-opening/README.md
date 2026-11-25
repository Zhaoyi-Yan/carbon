# v9 reactive opening: remove the early-income override

2026-11-25. Protocol per [v9 leader study §5](../../v9-leader-study-20261125.md). One-line candidate, bounded to 48 games.

## Fixed identities

| Item | Value |
| --- | --- |
| Baseline | `e97a386b290a1727f26331470094ab57cd452948:yan-v8-own-prior.py`, SHA-256 `83bc00ca52ba42ece876830649c9bc488105e763179b62018558f555bb94fa9c`, 9,137,552 B |
| Candidate | `yan-v9-reactive-opening.py`, SHA-256 `e69766bdb72994cfbc01dc685155f5c2e9a087c86e1582c5317bfe9d82d06dab`, 9,137,472 B |
| Change | removed exactly one line in `_act()`: `if configuration is not None:act=_v7_early_income(obs,configuration,lb,act)`; the function definition and everything else stay |
| Kept fixed | network, prior parameters, search/opponent models, scoring, candidate budget, remaining post-processing |

## Pre-flight probe

`opening_probe.json`, three seeds, both seats, real initial observations:

- candidate returns `{}` at step 0;
- baseline recruits a collector and a planter at step 0;
- candidate recruits one collector and one planter at step 1 (it does not wait forever when the opponent also waits);
- the search engine loads and the file compiles.

## Maps and gates

`protocol.json` freezes 12 screen + 12 confirm maps, disjoint from the 33 v8-prior replay layouts, the previous v8-prior screen/confirm seeds, the v7 panels and the 57 v9-study games (arena `layout_sha256` formula).

- U = 0.5·sign(candidate cash − baseline cash) + 0.5·tanh(margin/1500).
- Screen: 24 games, mean U > 0, all games finish normally, no candidate failure.
- Confirm: 24 independent games, same SHA, mean U > 0 and 95% map-bootstrap lower bound > 0 (fixed RNG 20261125, 1,000,000 resamples).
- Failure at either stage ends the candidate; no third batch, no wait-time scan, no CP/PC tuning.

## Result (screen, 24 games)

| Metric | Value |
| --- | --- |
| W–D–L | 11–0–13 |
| Mean U | **−0.0617** (sd 0.642) |
| Mean margin | −82.1 |
| Seat 0 / seat 1 | 5–7, U −0.112 / 6–6, U −0.012 |
| Finishes | 24/24 at 299 steps, 0 failures, slowest move 0.819 s |

**The screen gate requires mean U > 0; it failed (−0.0617), so the candidate stops here.** Per the pre-registered protocol: no confirm batch, no third batch, no wait-time scan, no CP/PC tuning on the same confirmation maps.

This only rejects "remove the early-income override on top of frozen v9" at this 24-game budget; it does not prove every reactive opening is useless. The negative result and the per-game JSONL are retained.
