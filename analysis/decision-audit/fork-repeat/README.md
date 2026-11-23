# Repeated decisions from twelve public v7 maps

All 72 branch continuations reached step 299. The frozen baseline is the delivery file tagged `v10-yan-v7-selfplay-prior`, SHA-256 `4894fe44c601b7360d37de2eed928802c73884f0929a5897604d174e333e02e6`. Both seats use that version after the intervention under a fixed search budget. Each map contributes one historical v7 decision. The two arms share the opponent's first action, then both sides replan. Six maps use each historical v7 seat. An additional A/A pair from turn 258 matched every joint action and referee board to turn 299.

| Visible condition | Maps | Mean change in terminal margin | Maps favoring v7 in all three streams | Maps favoring the alternative in all three |
|---|---:|---:|---:|---:|
| Below one recruit's price | 3 | −258.74 | 1 | 0 |
| Loaded collector threatened | 3 | −4.00 | 0 | 0 |
| Friendly tree threatened | 3 | −233.17 | 1 | 0 |
| None of these conditions | 3 | −134.18 | 1 | 0 |
| All | 12 | −157.52 | 3 | 0 |

Nine maps changed which finalist performed better across the three continuation streams. The median map effect was −138.04. In a retrospective check, two streams agreeing on the runner-up triggered six switches on the third stream. All six lost terminal margin. Averaged over all 36 held-stream choices including the 30 unchanged defaults, this rule lost 62.58 points. This within-root diagnostic uses information unavailable in a live match. The number of independent maps is twelve.

Six raw finalist pairs have identical old five-feature representations. The real execution chain modified at least one finalist on two maps. After postprocessing, ten pairs differed by one worker instruction and two pairs differed by two instructions. At turn 66 on the low-cash map from the RyanX replay, the alternative moved a collector carrying 116.07 one tile closer to home. Its terminal margins relative to keeping the v7 choice were −99.53, −438.33 and −878.50. The change in a single unit's home distance does not establish its team-level payoff.

Roots come from five ranked opponent entries. Their historical actions stop at the branch. Both agents' private memories are reconstructed from legal observations and recorded past moves. Local enemy reactions come from v7. This panel measures alternatives inside the frozen search and provides no win-rate estimate for a new strategy or for online opponents. There was no fit, new release or leaderboard submission.

The manifest contains all candidate turns in the deterministic order used before observing outcomes. The low-cash root on the Apollo map used its second eligible turn because the first lacked two scored finalists. The root archive records the actual turn, both raw and executed moves, opponent move, worker-level changes, map seed, replay identity, original opponent, acting seat, stream, scores, intermediate cash components, terminal board hash and action-plus-state trajectory hash for each arm. The pilot source report separately counts actual generator admissions and shadow probe matches, with historical result margins preserved.
