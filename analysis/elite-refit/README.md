# Refit from the frozen yan-v7 networks

The experimental candidates were trained locally from `v10-yan-v7-selfplay-prior` using the latest exact leaderboard teacher IDs in the frozen public board. The public snapshot commit is `88717e63390fb8a3531c290b42bae2e01f5e9540`. The union of the snapshot, locally retained archives and recent official public matches provided 494 completed games on 494 different layouts. The six main teachers contributed 34 Artemis, 134 Apollo, 40 Master7sE, 27 Prometheus, 29 RyanX and 65 SpongeBaby v4 games, including losses. Both teacher seats can come from one game. There were 426 training, 35 checkpoint-selection and 33 held-test layouts. Across all ten IDs, the training fold contains 154,757 team-turns. The optimizer samples 18,432 team-turns per network in each experiment. The 17 prospective maps were obtained after the data freeze and after model selection.

| Matchup | Wins–losses | Seat 0 wins | Seat 1 wins | Mean candidate margin |
|---|---:|---:|---:|---:|
| Top six vs v7 | 2–6 | 1/4 | 1/4 | −350.94 |
| Top six with supplemental teachers vs v7 | 5–3 | 4/4 | 1/4 | +155.05 |
| Top six vs v6 | 3–1 | 2/2 | 1/2 | +251.76 |
| Top six with supplemental teachers vs v6 | 2–2 | 1/2 | 1/2 | +66.94 |
| Frozen v7 vs v6 control | 1–3 | 1/2 | 0/2 | −246.02 |

All 28 games reached turn 299, cash balances were checked at every step, and all recorded steps took less than one second. The screening criterion required three wins from each seat on four new maps against v7 and a positive mean. Neither candidate qualified for confirmation. The supplemental candidate's four seat-0 games averaged +380.67 and its four seat-1 games averaged −70.58. From seat 1 its tree net was 254.5 below v7 per game, with 90 more spent on hires, while depot receipts were 273.9 higher. The four maps give descriptive accounting and limited evidence about whether seat effects persist.

The following losses are equally weighted across the six core teacher identities on the 17 prospective maps. Lower is better. There was only one new Artemis map. Recovery and tree event definitions and counts are in the training code and `forward-test.json`.

| Action slice | Frozen v7 | Top six | Top six with supplemental teachers |
|---|---:|---:|---:|
| Worker action | 0.545 | 0.544 | 0.541 |
| Eligible base action | 0.598 | 0.560 | 0.571 |
| Teacher actually recruited | 0.823 | 0.769 | 0.701 |
| Loaded collector advanced toward depot | 0.483 | 0.449 | 0.474 |
| Plant or seize | 0.261 | 0.293 | 0.312 |

These supervised metrics use the identity of the player who moved. At deployment the primary teacher and an alternative are routed by similarity to teacher economic-state centroids. The new trunk, refitted rollout network and teacher routing changed together relative to frozen v7. B and C share the same architecture, centroids and training budget. Neither these metrics nor local games measure performance against the actual closed-source top six. Frozen v7 remains the reference and the game leaderboard received no submission.

The directory contains deployable weights, architecture-independent validation metrics, file hashes, source-game provenance, losses and action logs. Recreate either standalone Python candidate in a copy of this directory with `python experiments/elite_refit/network.py DIRECTORY top6-plus` or `top6`. Use the arena interpreter with PyTorch and NumPy. The candidate SHA-256 values are in `protocol.json`.
