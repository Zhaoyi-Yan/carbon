# Carbon League 策略

## 当前研究交接：v9 已上线，冠军回放与一步响应式开局验证

**最新 Codex 交接：[v9 上线后的冠军回放研究与下一步协议](analysis/v9-leader-study-20261125.md)。** 维护者确认 v8 own-prior 已作为 yan-QQ-v9 上线。11 月 25 日 10:59 官方导出快照记录其 30 局 16 胜、对当前前六 1 胜 7 负，尚无强敌突破。已分析同一 Artemis ID 的 28 场完整回放，并封存原始输入和复算程序。冠军 28/28 场首步等待；v9 的现有响应式开局被随后立即招募规则覆盖。这是值得验证的旧策略取舍，不是已证实的冠军胜因。下一步只从冻结 v9 恢复现有一步响应式开局，以新地图换边做最多 24+24 局，未达门槛即停止；联合队伍候选训练保留为条件性方向，不同步启动。

当前线上 v9 ID 为 `0f9896ed3dc3cd7bfad29fe2dd38d6fb`；本地对应文件见 [`e97a386` 的 yan-v8-own-prior.py](https://github.com/Zhaoyi-Yan/carbon/blob/e97a386b290a1727f26331470094ab57cd452948/yan-v8-own-prior.py)，SHA-256 `83bc00ca…`。`selfplay.py` 目前没有 `yan-v9` 别名，比较 v9 必须显式传冻结文件路径。线上回放不含源码 SHA，文件对应依赖维护者的部署说明。

[研究方案 v2](analysis/carbon-zero-research-plan-20261124.md) 定义了复用 ft3 Apollo、合法观测、固定温度、混合终局效用的 Zero-W 路线。[首轮 200k PPO 实验](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/analysis/carbon-zero/README.md) 已完成但棋力退化，冻结 v8 不变。

上一轮已完成：[v8 own-prior 复审：本地验收通过，冻结候选并交付](analysis/v8-own-prior-review-20261125.md)。`experiment/v8-own-prior` 的 `e97a386` 只更新完整 v8 的两个已有 prior 参数槽，同一候选初筛 14–10、独立确认 15–9；确认平均 U +0.1562，地图区间下界约 +0.006，通过原定门槛。合并 48 局 29–19，但区间跨零、席位贡献不均，不能称稳定普遍优势或已能击败前六。这轮开发已结束，不追加第三批对 v8 的验收；后续部署与线上结果以上方最新交接为准。

[前轮路线调整与原协议](analysis/carbon-route-reset-20261125.md)已完成本次有限尝试；Zero 的 M/G/M-wide 继续暂停。[M/G 首种子复审](analysis/carbon-zero-mg-review-20261125.md)、[B1-fixed/M-G 协议](analysis/carbon-zero-b1fixed-review-20261124.md)、[B1 复审](analysis/carbon-zero-b1-review-20261124.md)和[更早 PPO/RLOO 方案](analysis/carbon-zero-rl-followup-20261124.md)保留为历史记录。

以后每轮新的研究分析、建议和实验协议写入 GitHub 的 analysis 文档并更新这里的入口；实施结果保留在对应实验分支。

## yan-v8 实验交接

前代已发布基线为 [yan-v8.py](yan-v8.py)，冻结标签为 `v11-yan-v8-elite-refit`。[2026-11-24 实验结论与执行计划](analysis/yan-v8-next-experiments-20261124.md)保留了持续树任务、回本筛选和联合动作实验的结果、回放要求及评分审查案例。当前 v9 及最新执行优先级以上方冠军回放研究为准。

## 历史主力 yan-v7

[yan-v7.py](yan-v7.py) 保存前代本地晋级策略，冻结标签为 `v10-yan-v7-selfplay-prior`，内含全部模型权重。对冻结的线上 yan-v6，八张新图换边 14 胜 2 负，另外四张预留图换边 7 胜 1 负。两批使用同一个文件，合计 21 胜 3 负，场均资金分差 +409.63。它根据双方未来 5 和 10 步的预计入库比较晚盘走法，资金断流时优先检查能安全兑现的货物。普通局面会从自己的历史搜索中学一套工人动作，作为原搜索多比较的一种选择。赛事平台由维护者另行上传。[八图初筛](analysis/v7/13-qualifying-summary.json) · [四图复核](analysis/v7/13-confirmation-summary.json) · [策略与败局](experiments/v7/README.md)

## 历史主力 yan-v6

[yan-v6.py](yan-v6.py) 是冻结的前代本地晋级文件，包含策略网络、价值网络、压缩的 v5 对手模型与快速推演模型。标准 15×15 棋盘用 NumPy 和 PyTorch CPU 推理。对冻结的线上 [yan-v5](yan-v5.py)，首批新图换边 12 胜 4 负，另批四图换边 6 胜 2 负。两批合计 12 张新图、18 胜 6 负，场均领先 682.19。本次交付记录本地成绩，赛事上传由维护者决定。[首次八图](analysis/v6r-qual-05-summary.json) · [独立四图](analysis/v6r-confirm-05-summary.json) · [策略与逐轮评测](experiments/v6r/README.md)

以下是历史版本 **yan-v3**（`v6-tree-guard-batch`）。它沿用公开回放训练的行动和招募模型，在争夺树木时比较改种邻格、派空载采集员协防等走法，并在 CPU 上批量采样模拟行动。

yan-v3 在四张预先留出的新图上与 yan-v2 换边对战，得到 **5 胜 3 负，场均资金领先 176.63**。已按该文件提交线上评测。可用 `--baseline yan-v3` 复测上代。上一版 yan-v2 对应 `v5-tree-value`，yan-v1 对应 `v3-search`。[八局逐图结果](analysis/arena/yan-v3-qualification.md) · [自对战用法](ARENA.md)

## yan-v2 的历史验证结果

| 本地对手 | 地图种子 | 换边战绩 | 平均资金领先 |
| --- | --- | --- | --- |
| 上代 yan-v1（v3-search） | 811、853 | 4 胜 0 负 | 866.72 |
| claude-AlphaCarbon-Master3s2 | 733、809 | 4 胜 0 负 | 895.94 |
| 官方示例、juzi、DeepSeek v3 | 887 | 各 2 胜 0 负 | 见逐局记录 |

以上 14 场均打满 299 步。上线版本核实后，另以隔离竞技场在 19417 换边对打，yan-v2 两局分别领先 yan-v1 1786.33 和 771.82。[本次两局原始记录](analysis/arena/self-19417.jsonl)

Apollo 的 21 场公开回放用于训练和选择权重，与 Apollo 本体的直接对战成绩仍待验证。[逐局比分及实验条件](strategy-notes.txt)

## 离线竞技场与前五研究

[竞技场使用说明](ARENA.md)包括本机和 Codex Cloud 的安装、换边循环赛与多机结果汇总。[前五策略回放研究](analysis/top-five.md)保留了 77 张地图的原始回放和观测指标。[己方 21 场线上回放](analysis/yan-v2-economy.md)定位了争夺新树的中盘亏损。[对手路线实验](experiments/route_forecast.md)比较了对手预测的准确度和真实对抗结果。

## 版本与复现

- [迭代索引](ITERATIONS.md)列出了主要版本、实验候选和对应代码；[主分支提交记录](https://github.com/Zhaoyi-Yan/carbon/commits/main/)可逐版比较。
- [训练脚本](train_replays.py)读取[随仓库保存的 21 场回放](replays/apollo/manifest.json)，16 场训练、5 场验证。安装 NumPy 和 PyTorch 后，在仓库根目录执行以下命令即可生成包含权重的 candidate.py。
- [本地评测脚本](benchmark.py)使用原版 zerosum_env。原版裁判已随仓库保存，可交换出生位测试当前文件与保存的公开对手。

```sh
python train_replays.py --output /tmp/carbon-retrained
CARBON_DEBUG=1 python scripts/selfplay.py \
  --candidate /tmp/carbon-retrained/candidate.py --seeds 733 809 \
  --jobs 1 --output arena-runs/retrained-vs-yan-v4.jsonl
```

每轮开发用独立 Git 提交保存策略和评测结果，记录对手、种子、出生位和已知败局。沿用主线的版本打标签，未验收的候选可保存在实验分支。GitHub 用于公开代码和复现实验，比赛平台提交由维护者另行决定。

## 来源

策略与价值网络基于 Claude 公开的 Master 系列，沿用原文件对 Ryanx 快速模拟器的署名。回放来自赛事官方公开导出仓库。[Master2 源码](https://117.72.210.134/source/e591b77f3f8c72b4e243c16a4d2e60e0) · [Master3s2 源码](https://117.72.210.134/source/d679ed45ab595e8b7bb1ea51c3451154) · [回放仓库](https://github.com/JUZI1/carbon-league-replays) · [原版环境](https://github.com/moliqingwa/carbon_challenge)
