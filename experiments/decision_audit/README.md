# v7 decision audit

Run this from the experimental checkout, using the arena Python interpreter.

```sh
python3 experiments/decision_audit/run.py arena-runs/decision-audit-first
```

Use `--smoke` for one source match. The runner refuses to overwrite the output directory. It builds a traced copy from the frozen v7 hash and uses the vendored referee. Each historical prefix is replayed with the recorded joint actions to steps 90, 185 and 245. The runner checks the initial layout hash and both cash balances after each replayed move. It warms v7 and the opponent independently on their historical observations and actions.

The trace records proposals that reach their original generator, the admitted pool, the depth scored for each candidate, final comparable scores, the selected action, and a snapshot after each postprocessor. Identical joint actions can have multiple source labels. Supplemental probes run separately from the admitted pool and include up to one sampled alternative in the payoff comparison per root. The supplemental style sampler has its own random stream. It measures the potential of such an alternative without claiming that the original sampler generated it.

At each root the first opponent action is held identical across continuations. The original search, final execution, admitted finalists and one outside candidate can contribute multiple labels to the same continuation. From the next step, both agents react independently with restored private memory, a common per-turn random seed and equal fixed search settings. The auditor corrects the stored destination and liquidity action to the executed alternative. It checks the untraced root action against the traced move. Match scores, actual bank receipts and net tree income are recorded at common horizons and at the terminal state.

The first run is a pipeline pilot. It uses four historical local maps, with two v7 versus v6 and two v7 mirror games. It inherits the existing fixed-width and fixed-depth harness, which differs from tournament wall-clock search. Analyze roots within their original map. A tested candidate's result is conditional on the selected opponent and seed, and score comparisons apply only to candidates evaluated at comparable depths. The traces and paired payoffs are inputs for the closed-loop ranking experiment in the design document.

The first batch finished all 36 continuations at step 299. The trace changed the search output at one of 12 roots, where the executed move improved the terminal margin by 295.72. Five of 11 outside candidates improved on the original search choice, with an average of −140 when weighted equally by source game. The candidate ranked first beat the second in five of 12 comparisons. Its map-weighted advantage was +2.49. Both ranking and outside-candidate results varied substantially by opponent and map. All four historical roots came from games observed from seat 0. Broader coverage is needed before fitting a ranking head or changing the search or execution chain.
