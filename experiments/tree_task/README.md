# Persistent new-tree task

Run against the eight public games listed in `analysis/tree-task/replay-sources.json`. Save each public replay in the input directory as `MATCH-1.html.gz` and copy the manifest to `INPUT/games.json`. The input parser accepts the published HTML with either plain or gzip encoding.

```sh
caffeinate -i python experiments/tree_task/run_task.py INPUT NEW_OUTPUT_DIRECTORY
python experiments/tree_task/analyze.py NEW_OUTPUT_DIRECTORY
```

An affordable chance is a planter already on an empty non-base tile with cash for the plant and two hires, no visible enemy collector one move away, and an enemy planter within four steps. Within turns 65–244, source matches and opportunities are allocated by hash before comparing outcomes. The recorded command separates staying from leaving. Four opportunities from each group use different maps. Replay prefixes verify both sides' board observations and rewards against the actual referee. Proxy memory is reconstructed from earlier observed states and commands.

The first arm runs frozen v8. The second arm changes only the target planter's first action to stay and then resumes v8. The third uses that same first action and keeps a planter near the new tree for up to 18 turns. It stops once the tree has paid its nominal price, the worker or tree is lost, or there is no safe adjacent position. The planter covers the tree when an enemy planter approaches and avoids known collector landing cells, allied destinations and idle spending on a second tree. The task influences actual simulated decisions after the root. It does not modify the inner search's evaluation of alternatives.

Each snapshot branches against frozen v7 and v8 using one deterministic stream for each proxy. The two arms that plant share an identical joint first step and deploy the same histories. Every continuation uses the published referee, fixed width and depth limits, valid actions and reconciled cash flows. Per-tree ownership and ordinary income are tracked separately from all-team tree income, collision transfers, recruitment and depot receipts. The output reports both differences, at common milestones and terminal. The proxy reactions and these limited cases define the scope of the result.
