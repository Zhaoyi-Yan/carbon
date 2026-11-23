# Closed-loop finalist ranker first cycle

The candidate applies a bounded five-feature residual correction to v7's final search scores from steps 180 through 284. It leaves the existing proposals and rollout budget intact. The fit uses 24 decisions at steps 215 and 260 from twelve archived v7 versus v6 games. Four separate v7 mirror maps provide eight held-out comparisons. Their original v7 ranking agreed with the eventual ordering zero times across seven unequal outcomes. The head changed one choice and improved that continuation by 99.49. Five pairs had the same five features for both moves. Only depot progress and visible harvest received nonzero fitted weights.

| Matchup | Wins–losses | Seat 0 wins | Seat 1 wins | Mean margin |
|---|---:|---:|---:|---:|
| Candidate vs v7 | 2–6 | 1/4 | 1/4 | −238.98 |
| Candidate vs v4 | 8–0 | 4/4 | 4/4 | +1185.92 |
| v7 vs v4 | 8–0 | 4/4 | 4/4 | +1273.69 |

The candidate's paired mean against v4 was 87.77 below v7 on the same maps and seats. The difference was positive on two of the four map means. All 24 normal arena games reached step 299, with all recorded cash balances verified under replay and no steps exceeding one second. Directly facing v7, the candidate averaged 466.0 ms and v7 470.5 ms. These times include the normal adaptive search and host effects.

The training and test branch phases used fixed search budgets, one observed starting seat per historical layout and one common random stream per root. Four new maps and a single external opponent family give an initial transfer check. The two scores and end-of-game returns can differ because the branch changes the first action and both sides then replan. The five features describe immediate economic effects and often fail to distinguish distinct joint actions. This candidate did not outperform the frozen v7 reference and stays experimental.
