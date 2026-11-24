# Team decisions on public v8 replay roots

Twelve fixed public positions were replayed to their recorded states. Eight covered planting at turns 90–223 on eight maps, balanced by seat and prior recorded planting action. Four covered a late cash lead with an opposing cargo advantage, including the Word report's Master7sE position at turn 250. The two panels share some source maps. Both local seats use frozen v8 under the fixed width-4 search budget, with historical opening memory. The enemy plays either the natural v8 or historical first action at the root, and reacts to the changed state with v8 thereafter. The portfolio contains 96 arm-response rows and 65 distinct terminal trajectories; repeated replies and identical moves reuse verified trajectories. All distinct continuations reached turn 299. The independent A/A comparison reproduced the full trajectory.

The paired margin changes below are candidate minus original, using equal weight per root. An unchanged or unavailable action contributes zero. The runner-up passes through v8's normal postprocessing.

| Panel and action | After 10 turns | After 20 turns | Terminal |
|---|---:|---:|---:|
| Tree panel: plant | −47.66 | +25.41 | −54.51 |
| Tree panel: defer | +35.20 | −86.48 | −62.60 |
| Tree panel: native runner-up | −15.71 | −121.45 | −14.19 |
| Late panel: bank sooner | −89.44 | −89.41 | −112.75 |
| Late panel: intercept | +64.23 | +42.73 | −68.33 |
| Late panel: native runner-up | +83.76 | −103.85 | +81.63 |

The tested planting and deferral changes lacked repeatable terminal improvements across two distinct enemy first moves. At 20 turns, planting improved relative cash under both responses on two roots, while deferral did so on one. Each corresponding alternative was missing from the candidate pool. These were horizon-specific opportunities. The 10-turn positive average for deferral came from one map, with three maps worse and four ties.

The native second-ranked tree-panel actions on Prometheus replay `b02fe380` at turn 209 and Artemis replay `422eb406` at turn 90 improved terminal outcomes under both enemy first moves. The first gained 67.87 and 350.52, the second gained 98.57 and 1,206.24. Neither action was altered by postprocessing. The v8 score deficits were 0.00045 and 0.00959 respectively. The 20-turn result was positive under both responses only on the Artemis root. This is a case-specific ranking error under these two evaluated responses, while the runner-up was down 14.19 across the tree panel.

At Master7sE turn 250 the cash lead was 247.81 and cargo was 378.73 versus 653.77. Compared with the original action, banking sooner changed the margin by +21.35 and −147.94 at 10 turns, −390.28 and −458.43 at 20 turns, and −278.86 and −100.98 at the end, for the natural-v8 and historical first replies respectively. By turn 270 the relative base receipts were 219.80 and 399.70 lower. The banking action was absent from the pool and both branches verified that admitting this exact action would have hurt in these conditions. No distinct interception proposal was found by the tested generator on this root. On the other three late maps an interception change appeared once, improved the cash margin by 256.93 after ten turns, and lost 273.30 at terminal. That map's first replies were identical. The positive mean of the late runner-up across four roots did not reproduce on any root across two distinct first replies.

The record stores the whole team action, actual first reply, candidate admission and finalist score evidence, both teams' cash and economic accounting at 10 and 20 turns and terminal, all losses, and trajectory hashes. No candidate was promoted.
