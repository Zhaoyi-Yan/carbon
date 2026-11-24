# Rollout network and primary style

Both candidates were built from the frozen top6-plus refit. The rollback aliases new teacher IDs to the original v7 small net's styles. A CPU check found identical logits and values for all ten alias rows. The promoted candidate sets the primary own style to general style 20. The second root proposal still uses the old router's second choice. It uses the refitted full network, critic and small rollout network.

| Candidate vs frozen v7 | Initial four maps | Average margin | Four independent confirmation maps | Average margin |
|---|---:|---:|---:|---:|
| Original top6-plus | 7–1 (3/4, 4/4 by seat) | +317.38 | 3–5 (1/4, 2/4) | +188.80 |
| Frozen small rollout net | 2–6 (1/4, 1/4) | −96.37 | | |
| Fixed primary general style | 6–2 (3/4, 3/4) | +427.47 | 6–2 (3/4, 3/4) | +151.23 |

Four initial games timed out, including two v7 failures. The retry runs used the original map, seat, file hashes and referee deadlines, with two parallel games instead of three. All four reached turn 299. The original failures and retry records are preserved. The original refit was added to the independent confirmation panel after its completed screening passed the same gate. All 40 games retained for the comparisons reached 299 and reproduced their per-turn cash. In confirmation the generic candidate's matchup had two over-one-second steps covered by the normal overage allowance, and all games finished.

Across confirmation games the selected candidate collected 235.41 more carbon at bases and spent 33.75 less on hiring per game than v7. Tree net was 117.93 lower. Both losses were on the same confirmation map, seed 1347358. The final margins on that map were −296.45 and −1519.67.

The split tree audit on new public maps tracked 60 successful plants and 71 successful captures from the six core teachers. The main-network cross-entropy changed from 0.266 to 0.395 for plants and from 0.293 to 0.244 for captures after the top6-plus refit. The small network changed from 0.686 to 0.845 on plants and from 0.439 to 0.468 on captures. The reported plant and capture means weight equally the teachers with observed actions for each slice. The original training drew 18,432 samples per network from 16,263 distinct team-turns for top6 and 16,806 for top6-plus. The audit and checkpoint logger now record these slices separately.

The tested standalone file uses SHA-256 `a87f8e6b83656c771cb5a25d6b6f383b245127e7c782e4c6c7c4a0a2cc15d396`. The baseline is `v10-yan-v7-selfplay-prior` with SHA-256 `4894fe44c601b7360d37de2eed928802c73884f0929a5897604d174e333e02e6`.
