# 策略版本索引

主分支的六个标签保存达到阶段目标的单文件策略。下表还原了本地留存的实验快照。相同文件按 SHA-256 去重。归档提交的日期是整理日期，排列顺序不代表每个实验都基于上一行继续开发。

| 阶段 | 策略 | 原有本地记录 |
| --- | --- | --- |
| [v1-heuristic](https://github.com/Zhaoyi-Yan/carbon/blob/v1-heuristic/submission.py) | 部分可见地图启发式，协调工人走位与回程 | 启发式基线 |
| [v2-master2](https://github.com/Zhaoyi-Yan/carbon/blob/v2-master2/submission.py) | Master2 权重、落点冲突处理与终盘交付 | 对示例、juzi、DeepSeek v3 共 36 胜 0 负 |
| [v3-search](https://github.com/Zhaoyi-Yan/carbon/blob/v3-search/submission.py) | 基于 Master3s2 的自适应搜索 | 对公开原版 Master3s2 为 3 胜 1 负，平均领先 312.28 |
| [v4-replay-policy](https://github.com/Zhaoyi-Yan/carbon/blob/v4-replay-policy/submission.py) | 用 Apollo 回放学习工人行动、招募和开局 | 对 v3 为 4 胜 0 负，平均领先 1223.57 |
| [v5-tree-value](https://github.com/Zhaoyi-Yan/carbon/blob/v5-tree-value/submission.py) | 搜索中加入未来 18 步树木收益预测 | 对 Master3s2、v3 各 4 胜，其他三个对手合计 6 胜，共 14 胜 0 负 |
| [v6-tree-guard-batch](https://github.com/Zhaoyi-Yan/carbon/blob/v6-tree-guard-batch/submission.py) | 线上 yan-v3，护树时改换种植点或由空载采集员协防，CPU 批量采样 | 对冻结的 yan-v2 四图换边 5 胜 3 负，平均领先 176.63。[八局与提交记录](analysis/arena/yan-v3-qualification.md) |

## 冻结的线上版本与新候选

从本人账户下载的 yan-v2 与标签 v5-tree-value 完全相同（SHA-256 `0483eca2a5e8031270dfb89b87cd5d901ee81f823383f6094f8ed839dec6a7dd`），yan-v1 与 v3-search 完全相同（`29aa472ef0484b65bb3ffdc85113aef2d1daa1e25377bfe5355a4d5e3567a11c`）。下表均以冻结的 yan-v2 为对手，每张地图从两个出生位各打一局，分差从候选一侧计算。

| 候选 | 种子 | 换边战绩 | 平均现金分差 | 记录 |
| --- | ---: | ---: | ---: | --- |
| [种植改点与空载协防](https://github.com/Zhaoyi-Yan/carbon/blob/9e99a57fb2d16783c53ab351c197f82b877877ff/submission.py) | 35831、41893、47269 | 4 胜 2 负 | +37.08 | [两张独立复测图 2 胜 2 负，逐局分差](experiments/tree_guard_search.md) |
| [路线概率温度校准](https://github.com/Zhaoyi-Yan/carbon/blob/c61c413fd39b751fff4f0207b38e520634625db9/submission.py) | 17471 | 0 胜 2 负 | −455.73 | [隔离竞技场](analysis/arena/temperature-17471.jsonl) |
| [Apollo 回程特征](https://github.com/Zhaoyi-Yan/carbon/blob/bbf3d0badaec6ecefe9c5d1a49920ea117882833/submission.py) | 17471 | 1 胜 1 负 | −443.12 | [原 benchmark 计时](experiments/route_forecast.md) |
| [搜索中复用己方行动头](https://github.com/Zhaoyi-Yan/carbon/blob/08a8137e578db46a2ab4d5a06a2cd735009d5a9e/submission.py) | 26117 | 1 胜 1 负 | −7.955 | [逐局记录](analysis/arena/own-rollout-26117.jsonl) · [18 场线上动作核对](experiments/own_rollout.md) |
| [争夺中树木的收益校准](https://github.com/Zhaoyi-Yan/carbon/blob/69f549d6b0f4e3995f339e3d325f03ebc4b690dd/submission.py) | 31427 | 0 胜 2 负 | −141.56 | [换边实测和留图误差](experiments/contested_trees.md) |
| [单采集员前压](experiments/pressure_opponent.md) | 20481 | 0 胜 2 负 | −1187.515 | [隔离竞技场](experiments/pressure_20481.jsonl) |

## 实验快照

在 GitHub 点击文件名可查看和下载当次的完整单文件。探索分支包含早期草稿、搜索变体、风格和采样温度试验。joint 是未做整局评测的实验。逐局评测及已知败局见[策略记录](strategy-notes.txt)。

| 本地快照 | Git 提交 | 策略 SHA-256 |
| --- | --- | --- |
| [first_draft.py](https://github.com/Zhaoyi-Yan/carbon/blob/e536124bf913914275c8dc8fe898e2b970eee39e/submission.py) | [`e536124b`](https://github.com/Zhaoyi-Yan/carbon/commit/e536124bf913914275c8dc8fe898e2b970eee39e) | `03aeffce18947934c45be2f00794da07628b6e83c02e9c53e1ed9b1d7bd4e763` |
| [before_routes.py](https://github.com/Zhaoyi-Yan/carbon/blob/631b418a6a62ea5c4af5fa01abca1df208ba6d7a/submission.py) | [`631b418a`](https://github.com/Zhaoyi-Yan/carbon/commit/631b418a6a62ea5c4af5fa01abca1df208ba6d7a) | `65e308fe7b8ee35d354017a036c23d571f313c11363004a70a6d002c99d7f200` |
| [before_safety.py](https://github.com/Zhaoyi-Yan/carbon/blob/9249b305b006d93b820a469b421a56667d507da0/submission.py) | [`9249b305`](https://github.com/Zhaoyi-Yan/carbon/commit/9249b305b006d93b820a469b421a56667d507da0) | `ff1d3e5c932db05cd07118f48db4c0544c15b50b506b869d06a74ee18733672e` |
| [master2_t40.py](https://github.com/Zhaoyi-Yan/carbon/blob/c63ba061409c582613dfbe4a1a42c3ea673699b8/submission.py) | [`c63ba061`](https://github.com/Zhaoyi-Yan/carbon/commit/c63ba061409c582613dfbe4a1a42c3ea673699b8) | `cc0edbaef2d573fc3e0d74da057538618fed201963cf0883553a530084656a46` |
| [master2_t75.py](https://github.com/Zhaoyi-Yan/carbon/blob/76790282889db7b548ad9446900f6db4dbb64f8f/submission.py) | [`76790282`](https://github.com/Zhaoyi-Yan/carbon/commit/76790282889db7b548ad9446900f6db4dbb64f8f) | `7c8f8985bef230287b972251e634fad85da3adb54dbb422414be81751ebf59d1` |
| [master2_t100.py](https://github.com/Zhaoyi-Yan/carbon/blob/0546c5bcee9fb82e90dde702b60fd9b0ac1d7126/submission.py) | [`0546c5bc`](https://github.com/Zhaoyi-Yan/carbon/commit/0546c5bcee9fb82e90dde702b60fd9b0ac1d7126) | `06b8456a88fe7d14ef8c2df6136744d49eeef303b59fb6736cdcde61e16e8707` |
| [master3_t40.py](https://github.com/Zhaoyi-Yan/carbon/blob/4e0d9dcb7386917242eeca0710e9c36e6bf5cd9d/submission.py) | [`4e0d9dcb`](https://github.com/Zhaoyi-Yan/carbon/commit/4e0d9dcb7386917242eeca0710e9c36e6bf5cd9d) | `46896503a9a2f382e56835eca29b893f430f86db7024871510417435946c0fbf` |
| [master3_t75.py](https://github.com/Zhaoyi-Yan/carbon/blob/10e69218bef233f62b2e7538f43314802d39b060/submission.py) | [`10e69218`](https://github.com/Zhaoyi-Yan/carbon/commit/10e69218bef233f62b2e7538f43314802d39b060) | `35e107417391751e234b00d6e02c70b919b8109623d4739080f4c9d82ba89b28` |
| [master3_t100.py](https://github.com/Zhaoyi-Yan/carbon/blob/1f5bdbf19eff09461ac01df00ea0070202514243/submission.py) | [`1f5bdbf1`](https://github.com/Zhaoyi-Yan/carbon/commit/1f5bdbf19eff09461ac01df00ea0070202514243) | `da1fee36fdf635969872f8a6c8a03dcb9da8edc90d6d679366e7d9dd3d9d9474` |
| [hybrid_master3.py](https://github.com/Zhaoyi-Yan/carbon/blob/3de702c4db6a81d25f71955539da16efcf2f44bf/submission.py) | [`3de702c4`](https://github.com/Zhaoyi-Yan/carbon/commit/3de702c4db6a81d25f71955539da16efcf2f44bf) | `0eeda1b4888bfa01562498d0401f858ff62f61f704c502ce72cce4ad48a67a3c` |
| [paired_search.py](https://github.com/Zhaoyi-Yan/carbon/blob/aaec45c8edf4947fa5c4520b41ac4cf0de6dd4cc/submission.py) | [`aaec45c8`](https://github.com/Zhaoyi-Yan/carbon/commit/aaec45c8edf4947fa5c4520b41ac4cf0de6dd4cc) | `ff1c26ef78b39f3dfad0039d31bed19b86af779219a467be7d106d064d953cc9` |
| [deep_search.py](https://github.com/Zhaoyi-Yan/carbon/blob/86c963a838c809465ed0ccca90553f2ad5e2b21c/submission.py) | [`86c963a8`](https://github.com/Zhaoyi-Yan/carbon/commit/86c963a838c809465ed0ccca90553f2ad5e2b21c) | `ef71298c31559f6b7d0d5c54aae461d5bda2e3346f4c5c932cd460989c76ab8f` |
| [rotated_search.py](https://github.com/Zhaoyi-Yan/carbon/blob/f0deee62a63fa0e2cc01e0b20024ebf6af008329/submission.py) | [`f0deee62`](https://github.com/Zhaoyi-Yan/carbon/commit/f0deee62a63fa0e2cc01e0b20024ebf6af008329) | `3770edbf32ec9069294cc96265208c819f1963dd8939ee4514d48ce190ac4e93` |
| [best_reply_search.py](https://github.com/Zhaoyi-Yan/carbon/blob/a04c60ac4eeebaac1f3e75dd4a910695021e4a9e/submission.py) | [`a04c60ac`](https://github.com/Zhaoyi-Yan/carbon/commit/a04c60ac4eeebaac1f3e75dd4a910695021e4a9e) | `2386ff29046280ce47670856da6ba2793aa0f73374cf5e21eedd6d3773294a55` |
| [lean_search.py](https://github.com/Zhaoyi-Yan/carbon/blob/5501a9a93538cec42604e84cc3687917e0fea690/submission.py) | [`5501a9a9`](https://github.com/Zhaoyi-Yan/carbon/commit/5501a9a93538cec42604e84cc3687917e0fea690) | `3023ef51daf313e705404b8b395f6a742e6ea3f2abb4d8cd1b589c3340574901` |
| [adaptive_search.py](https://github.com/Zhaoyi-Yan/carbon/blob/79f76c5ed006800df7db066b5d37ef932762f073/submission.py) | [`79f76c5e`](https://github.com/Zhaoyi-Yan/carbon/commit/79f76c5ed006800df7db066b5d37ef932762f073) | `7df4bdafa6a2de259a64331998b69001324889245b2f68fcfedaccf0c13e9511` |
| [value_ensemble_search.py](https://github.com/Zhaoyi-Yan/carbon/blob/0669b2b460e218c53d522ef0fa924ee99fe0406e/submission.py) | [`0669b2b4`](https://github.com/Zhaoyi-Yan/carbon/commit/0669b2b460e218c53d522ef0fa924ee99fe0406e) | `9fcf94058a7e531d720e06147b2b54bb4824fcdeff102aac504c0f4b81c43241` |
| [style-rush.py](https://github.com/Zhaoyi-Yan/carbon/blob/41eb11349dfb54bd14d08fb28767bc8e94e46506/submission.py) | [`41eb1134`](https://github.com/Zhaoyi-Yan/carbon/commit/41eb11349dfb54bd14d08fb28767bc8e94e46506) | `f853280837310e0ecae62af5d87a50c2bb4e1778494c41ed2ca9984c99c67db9` |
| [style-seize.py](https://github.com/Zhaoyi-Yan/carbon/blob/ae5a843fae24cd17f696440cc0d284e0d85cce6c/submission.py) | [`ae5a843f`](https://github.com/Zhaoyi-Yan/carbon/commit/ae5a843fae24cd17f696440cc0d284e0d85cce6c) | `813b1dee64f3d6f6d9601382555904c1855b7b62a4328a002458a35a36452ff5` |
| [style-guard.py](https://github.com/Zhaoyi-Yan/carbon/blob/09912c2568050e63ec12197101ce406ecdb89a88/submission.py) | [`09912c25`](https://github.com/Zhaoyi-Yan/carbon/commit/09912c2568050e63ec12197101ce406ecdb89a88) | `2f1a1da372efce4aae09a4564ee3aef4d8f2871a2b5b667bdacb9fdd13e4a967` |
| [style-zenith.py](https://github.com/Zhaoyi-Yan/carbon/blob/f1e7825d631f08fda2abc59d7f943d0e125c16f7/submission.py) | [`f1e7825d`](https://github.com/Zhaoyi-Yan/carbon/commit/f1e7825d631f08fda2abc59d7f943d0e125c16f7) | `2dbd161514f49cc28415c73355409247d62e1f2fc3d7872726de11a5283610b8` |
| [style-arg.py](https://github.com/Zhaoyi-Yan/carbon/blob/6553dc3da64c58fe6c9344f8bc379ce5d9a68817/submission.py) | [`6553dc3d`](https://github.com/Zhaoyi-Yan/carbon/commit/6553dc3da64c58fe6c9344f8bc379ce5d9a68817) | `71d6f9f577e2717752b3d617b9f7250689e877e2edbe791c5d81e198578cf45e` |
| [style-forecast.py](https://github.com/Zhaoyi-Yan/carbon/blob/8dfc09d76a11c65e8c8fc7b5a708385eb671c480/submission.py) | [`8dfc09d7`](https://github.com/Zhaoyi-Yan/carbon/commit/8dfc09d76a11c65e8c8fc7b5a708385eb671c480) | `ad7ea34e39fa9fcdb3b7c281e4ca537776f1a6b4cd14b127ff80a25292cc05bd` |
| [style-master.py](https://github.com/Zhaoyi-Yan/carbon/blob/1b7cc18b6e98346c018ebc010c8762d2b647e20f/submission.py) | [`1b7cc18b`](https://github.com/Zhaoyi-Yan/carbon/commit/1b7cc18b6e98346c018ebc010c8762d2b647e20f) | `98b0eebbb1a5bbef10893ede94e9a7adf76304ae1dbc530c78c7c639bccf89a2` |
| [temp-plain0.py](https://github.com/Zhaoyi-Yan/carbon/blob/c484e3739214b7515894a253ae5da20033ecb74c/submission.py) | [`c484e373`](https://github.com/Zhaoyi-Yan/carbon/commit/c484e3739214b7515894a253ae5da20033ecb74c) | `ff2cf233d56a52491b381d134382cbfc78b621a413cc3e846287ad84b7f80232` |
| [temp-plain035.py](https://github.com/Zhaoyi-Yan/carbon/blob/09851e61d60b43d8bfb4e6df8b448916725e2c95/submission.py) | [`09851e61`](https://github.com/Zhaoyi-Yan/carbon/commit/09851e61d60b43d8bfb4e6df8b448916725e2c95) | `9eee0889305b5a4815dd1590f4280edf9d992c3c89d1a759cb48ca22020814e1` |
| [temp-plain055.py](https://github.com/Zhaoyi-Yan/carbon/blob/855ac587ff9b2fee0ad70085f79aa516946b1fd1/submission.py) | [`855ac587`](https://github.com/Zhaoyi-Yan/carbon/commit/855ac587ff9b2fee0ad70085f79aa516946b1fd1) | `9a52eda1cc8c3e8bcb8a4e60d71ac3e2bd10da00a960822f7b896b8efe1ea98d` |
| [temp-plain105.py](https://github.com/Zhaoyi-Yan/carbon/blob/7f42752dbbbc904544e2d2cb3dcc564d35df4e67/submission.py) | [`7f42752d`](https://github.com/Zhaoyi-Yan/carbon/commit/7f42752dbbbc904544e2d2cb3dcc564d35df4e67) | `2775cd93623537fe2dbc75c8aed7b225bc929209ac49b466273ae2204cbdae70` |
| [temp-rotate035.py](https://github.com/Zhaoyi-Yan/carbon/blob/9a62a0ce09bfb6793742e892b04aebf51d5f85a3/submission.py) | [`9a62a0ce`](https://github.com/Zhaoyi-Yan/carbon/commit/9a62a0ce09bfb6793742e892b04aebf51d5f85a3) | `3cc75f6b5c6813e2c4ef474ea01c0e3d829c5be1cc9f26628551a5280b5c4d5c` |
| [temp-rotate072.py](https://github.com/Zhaoyi-Yan/carbon/blob/9278976d896d09ed6f04d876235a4bdda6b3da0a/submission.py) | [`9278976d`](https://github.com/Zhaoyi-Yan/carbon/commit/9278976d896d09ed6f04d876235a4bdda6b3da0a) | `8224c792b48b031a7e98648972f04dadb30037a214fc66910073c07d960201af` |
| [temp-rotate110.py](https://github.com/Zhaoyi-Yan/carbon/blob/50497f766163d26f56b48560911e25da858d96de/submission.py) | [`50497f76`](https://github.com/Zhaoyi-Yan/carbon/commit/50497f766163d26f56b48560911e25da858d96de) | `89bb7458276af67337e77a002487b6d7f6927566284d9ebd87111ef7176ff49f` |
| [wide_search.py](https://github.com/Zhaoyi-Yan/carbon/blob/a9d8cc348061aea2cbe935e8a8be294340c3deba/submission.py) | [`a9d8cc34`](https://github.com/Zhaoyi-Yan/carbon/commit/a9d8cc348061aea2cbe935e8a8be294340c3deba) | `d5bdbdc1bd3f2281871dd2293c070fd5ba8e75e02f051c17f96f39ed2168bed6` |
| [wide_rotated_search.py](https://github.com/Zhaoyi-Yan/carbon/blob/d3f935dc2ce725038067e98dddc30ad05b01ef0a/submission.py) | [`d3f935dc`](https://github.com/Zhaoyi-Yan/carbon/commit/d3f935dc2ce725038067e98dddc30ad05b01ef0a) | `21597b8d9fc11756fa573fe20095c7d26bbc331dd5898fa2ef9fbe7ef1ada953` |
| [joint.py](https://github.com/Zhaoyi-Yan/carbon/blob/4d5bfad7e9b661dbaa7aaeed37964beb61fa6fa3/submission.py) | [`4d5bfad7`](https://github.com/Zhaoyi-Yan/carbon/commit/4d5bfad7e9b661dbaa7aaeed37964beb61fa6fa3) | `9616c1292990f9deb9b786dcdee95c7b42b1a055f3294ecfa3d030bf618f5450` |
