# Carbon League 策略

[yan-v6.py](yan-v6.py) 和 [submission.py](submission.py) 是同一个本地晋级文件，包含策略网络、价值网络、压缩的 v5 对手模型与快速推演模型。标准 15×15 棋盘用 NumPy 和 PyTorch CPU 推理。对冻结的线上 [yan-v5](yan-v5.py)，首批新图换边 12 胜 4 负，另批四图换边 6 胜 2 负。两批合计 12 张新图、18 胜 6 负，场均领先 682.19。本次交付记录本地成绩，赛事上传由维护者决定。[首次八图](analysis/v6r-qual-05-summary.json) · [独立四图](analysis/v6r-confirm-05-summary.json) · [策略与逐轮评测](experiments/v6r/README.md)

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
