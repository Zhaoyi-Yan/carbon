# Carbon-Zero warm-start pilot

Use the arena Python with PyTorch and a copy of Claude's exact ft3 source. The runner checks SHA-256 and the Apollo name-to-embedding mapping. The external source and learned weight files live outside the public repository. The analysis directory records their digests and experiment provenance.

```sh
python experiments/carbon_zero/decoder_check.py FT3_PATH DECODER_RESULT
python experiments/carbon_zero/checks.py FT3_PATH CHECK_RESULT
python experiments/carbon_zero/pilot.py FT3_PATH OUTPUT --steps 200000 --coordinated --lr .00003
python experiments/carbon_zero/checkpoints.py FT3_PATH OUTPUT 200000 --search-games
```

The collector delegates transitions to the vendored referee and restores the correct ID counters for each game. Both actors obtain separate legal observations before either action reaches the referee. The encoder reproduces ft3's quantized 54 planes and replays each seat's last commands through its own tracker. Terminal rewards combine a sign and bounded cash margin. Intermediate rewards telescope through a public cash potential. Partial rollout windows bootstrap with the unchanged behavior policy.

The sequential decoder samples bases first and then workers in fixed ID order. When an earlier action has reserved a destination, a later worker's distribution masks it while other valid moves remain. The trainer stores the sampled conditional masks, positions, command indices, and summed old team log-probability. Only the learner's side participates in PPO. Two value-only updates precede joint learning. Checkpoints store live game action prefixes and RNG states so the collector can recover the same window.

The standalone export implements the same decoder using NumPy. The old-search export uses the original ft3 search, critic, opponent and continuation networks, with the only learned replacement at its own root logits. Its pre-search decoder comes from the original ft3 search entry point for both arms. The old-search diagnostic is a local integration experiment.

The fast engine is a diagnostic candidate. Compare its state transitions on random games or on a local arena file produced with action recording.

```sh
python experiments/carbon_zero/fast_env.py FAST_RESULT
python experiments/carbon_zero/fast_env.py REPLAY_RESULT --arena-replay ARENA_JSONL
python experiments/carbon_zero/diagnose.py FT3_PATH OUTPUT ARENA_JSONL ACTOR_DRIFT_RESULT
```

For each saved checkpoint, run `checkpoints.py FT3_PATH OUTPUT CHECKPOINT` to create the deployment exports and pure external panel. Add `--search-games` for a timed, complete v8 swap on one map, or use `--search-only` to add that panel to an existing result.

Pass `--anchor-kl .1` for the short controlled comparison. It adds a KL term using the frozen ZW0 action probabilities on the learner's stored conditional masks. The reference remains fixed for the entire run.
