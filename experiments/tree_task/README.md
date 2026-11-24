# Persistent new-tree task

Run against the eight public games listed in `analysis/tree-task/replay-sources.json`. Save each public replay in the input directory as `MATCH-1.html.gz` and copy the manifest to `INPUT/games.json`. The input parser accepts the published HTML with either plain or gzip encoding.

```sh
caffeinate -i python experiments/tree_task/run_task.py INPUT NEW_OUTPUT_DIRECTORY
python experiments/tree_task/analyze.py NEW_OUTPUT_DIRECTORY
```

An affordable chance is a planter already on an empty non-base tile with cash for the plant and two hires, no visible enemy collector one move away, and an enemy planter within four steps. Within turns 65–244, source matches and opportunities are allocated by hash before comparing outcomes. The recorded command separates staying from leaving. Four opportunities from each group use different maps. Replay prefixes verify both sides' board observations and rewards against the actual referee. Proxy memory is reconstructed from earlier observed states and commands.

The first arm runs frozen v8. The second arm changes only the target planter's first action to stay and then resumes v8. The third uses that same first action and keeps a planter near the new tree for up to 18 turns. It stops once the tree has paid its nominal price, the worker or tree is lost, or there is no safe adjacent position. The planter covers the tree when an enemy planter approaches and avoids known collector landing cells, allied destinations and idle spending on a second tree. The task influences actual simulated decisions after the root. It does not modify the inner search's evaluation of alternatives.

Each snapshot branches against frozen v7 and v8 using one deterministic stream for each proxy. The two arms that plant share an identical joint first step and deploy the same histories. Every continuation uses the published referee, fixed width and depth limits, valid actions and reconciled cash flows. Per-tree ownership and ordinary income are tracked separately from all-team tree income, collision transfers, recruitment and depot receipts. The output reports both differences, at common milestones and terminal. The proxy reactions and these limited cases define the scope of the result.

## Payback screening

The second batch trains a map-weighted model on actual newly planted trees from the first batch's eight archived games. It uses both seats. The reward target is the tree's regular income during its first ownership spell, stopped at first loss, expiration, or the end of the game. Input features use legal belief-state carbon, current price, remaining tree life, nearby visible units and forest density. The opponent's past observed state is available. Future moves, replay carbon outside the acting seat's observation model, and future ownership are never input features.

```sh
caffeinate -i python experiments/tree_task/payback_batch.py TRAIN_REPLAYS HOLDOUT_REPLAYS NEW_OUTPUT_DIRECTORY
```

Use the corresponding manifests from `analysis/tree-payback` as the two input folders' `games.json`. Whole layouts are disjoint. The predeclared filter requires both a modeled payback probability of at least 0.65 and a projected median surplus of at least five. We take at most one root per holdout map. Roots failing either gate serve as diagnostic probes on maps with no eligible root. All investment outcomes are checked in closed loop against the same public starting state and local proxies as the prior batch. The probe and eligible groups reflect different boards, so their contrast is descriptive.
