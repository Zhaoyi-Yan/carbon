# Same-root team decisions

Run using the replay inputs documented under `experiments/tree_task`. The old and recent input directories correspond to the archived tree-task and payback panels.

```sh
caffeinate -i python experiments/team_branches/run.py OLD_REPLAYS RECENT_REPLAYS NEW_OUTPUT_DIRECTORY
python experiments/team_branches/analyze.py NEW_OUTPUT_DIRECTORY
```

The investment cohort uses eight maps, with four roots per historical seat and four recorded plant attempts and four departures. Turns are hash-selected after excluding neighborhoods of previous tree experiment turns. The late-return cohort includes the Master7sE board at turn 250 from the Word review and three other maps with a cash lead and an opposing cargo advantage. The selectors do not use final margin or cash after the selected decision. Some replay maps contribute to both cohorts at different turns.

Frozen v8 replays its memory on both historical streams. The referee verifies the historical public observations against each prefix. Each arm acts from the same board, fixed search budget and per-turn random stream. The opponent begins with either frozen v8's natural first choice or the historical first choice and reacts with v8 afterward. The historical opponent action is withheld from all candidate generators. Repeated first actions count once in the aggregate. The A/A repetition checks complete trajectory equality.

The original action is taken from the real execution chain. Planting and deferral alter only the target planter's first command. Banking tries the existing v8 planner, then an unobstructed closing move if the planner did not increase loaded progress. Interception uses the existing planner. Neither edit conditions on the opponent's simultaneous move. Synthetic edits inherit reconciled execution memory. The native runner-up enters at final selection and then goes through the entire v8 postprocessing and memory chain. Its outcome isolates this actual alternative with the same budget.

The trace distinguishes a whole joint action from its constituent unit move. An exact outside-pool improvement does not establish that the individual move was absent. Cases where a finalist's raw action was altered by postprocessing remain flagged. Team results compare the paired relative cash change, depot receipts, tree net and hiring cost after 10 turns, 20 turns and at the official end. Both sides' absolute cash changes and individual economic totals are retained. Aggregate effects weight each root equally. A better proxy result conditional on two first actions is evidence about these roots rather than a claim about online win probability.
