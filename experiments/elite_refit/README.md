# Full-network multi-teacher refit

Freeze the public board in `OUT/board.json` and clone the public replay snapshot to `OUT/public`. The collector combines that checkout, previously downloaded archives and the public recent-match lists for each exact top-ten ID. Run with the local arena Python.

```sh
python experiments/elite_refit/collect.py OUT EXISTING_ARENA_CACHE
caffeinate -i python experiments/elite_refit/batch.py OUT
```

Both seats of an eligible game stay together. The encoder uses the acting seat's partial carbon observation, historical commands, and v7's tree-carbon belief update. All frames of each selected seat are encoded, including losses. Metadata, final scores and future events enter labels and diagnostics only. Each repeated layout contributes one game. Whole newest maps covering each teacher are held for testing, followed by separate maps for checkpoint selection. Archived validation/test maps may have been seen during earlier versions' training. The final check also queries for matches that finished after this run's data freeze and excludes all previously selected layouts.

Both candidate arms start from the same v7 full policy, critic and small rollout network. B samples the six exact teacher IDs equally. C allocates 83% to the same six, 10% to the two local versions and 7% to the two earlier SpongeBaby versions. Sampling is uniform across games of a teacher, then frames. Each loss normalizes per active team and unit head. Shared-trunk optimization retains an action KL to the frozen network and calibrates values against the prior and actual future cash margins. Each network receives three epochs of 192 updates of 32 team observations. Root and rollout models are trained separately with identical sampling and budget in B and C. A general style receives identity dropout on 18% of examples.

Ten teacher embeddings extend the existing table. Search uses the two top-six profiles closest to its public cash, cargo, time, tree and worker state. These profiles are training-only empirical centroids shared by both candidates. Their IDs reach the full root, alternative proposal, full search and small rollout network. Existing v7 opponent identification and downstream cash/transport/tree logic continue to execute. Small and large packaged CPU backends are checked against the exported PyTorch models. Offline metrics use the known teacher identity, including for supplemental teachers, with the frozen comparison using each new row's initial v7 embedding. They measure imitation with a known identity. The arena exercises the actual router and complete policy.

The fixed four-map screening panel plays both seats against frozen v7. A separate two-map panel compares both arms and v7 against frozen v6. Candidates with at least three wins per seat and a positive average margin against v7 proceed to four further fresh paired maps. All normal games run with the original wall-clock resource budget and recorded actions. The process does not submit to the game leaderboard.
