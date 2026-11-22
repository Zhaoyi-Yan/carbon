# 本地与云端对战

竞技场直接使用仓库内的原版 Carbon 裁判。每个策略在独立进程内持有自己的局内状态，两个座位收到各自可见的碳量。给出三个以上的策略时，每一对都会在相同地图上交换座位。

## 每轮先挑战自己的主力

默认对手是线上 **yan-QQ-v6**，本地通过 `yan-v6` 复原冻结标签 v9-yan-v6-estuary。yan-v5 使用标签 v8-yan-v5-multistyle。[线上实战与新候选](experiments/v6/README.md)记录了复盘和换边验证。回归对手 **yan-v4** 对应标签 `v7-claude-style`。回归对手 **yan-v3** 对应标签 `v6-tree-guard-batch`。它在四张新图换边挑战 yan-v2，取得 5 胜 3 负。可显式传入 `--baseline yan-v2` 或 `--baseline yan-v1` 做跨代回归。前两版分别对应 `v5-tree-value` 和 `v3-search`，均已用本人账户下载文件的 SHA-256 核实。[yan-v3 评测及上传记录](analysis/arena/yan-v3-qualification.md)

新候选先挑战冻结的主力，再用历史主力及不同风格对手复查弱点。每张图双方都打两个出生位。单张图用于筛选，晋级前需要在开发时没看过的多张地图上复测。

已有 NumPy、jsonschema 和 CPU 版 PyTorch 的 Python 环境可直接使用。需要对齐赛事库版本时，用 Python 3.12 执行安装脚本。Linux 安装 CPU 专用 PyTorch。默认虚拟环境在仓库根目录。

```sh
bash scripts/setup-arena.sh
CARBON_DEBUG=1 .arena-venv/bin/python scripts/selfplay.py \
  --candidate yan-v2 --baseline yan-v1 \
  --seeds 19417 --jobs 1 --output arena-runs/seed-19417.jsonl
python3 arena_report.py arena-runs/seed-19417.jsonl
```

上面命令可还原两代线上策略的测试。平时只需传入新候选文件，基线由入口自动从已核实的 Git 版本读取并校验哈希。以下命令中的种子换成本轮提前留出的地图。

```sh
CARBON_DEBUG=1 .arena-venv/bin/python scripts/selfplay.py \
  --candidate ./candidate.py --seeds 19417 --jobs 1 --output arena-runs/challenger.jsonl
```

输出文件须使用新名字，以便保留旧战绩。每局结算后立即写一行。记录包含策略哈希、裁判哈希、地图指纹、出生位、双方现金、搜索次数和失败原因。输出顺序可以随并行任务的完成顺序变化。普通平局与正常胜负均返回退出码 0；存在弃权或环境错误会返回 2。

初始化限时 20 秒，每步基准 1 秒，单步最多 3 秒，累计额外时间最多 5 秒，最多 3 次慢步。计入进程通信与本机调度时间。仅运行可信策略；本工具不隔离网络、磁盘和机器 CPU/内存配额。环境配置沿用上游 randomSeed，留图评估适用于按正常观测决策且未用种子反推资源图的策略。

## 线上两代策略的直接对战

同一地图 `19417` 完整换边，yan-v2 从 0 号位以 9887.60 比 8101.27 胜，从 1 号位以 9067.12 比 8295.30 胜，平均现金分差 +1279.08。两局均为 299 步，每方搜索 298–299 次。此前在旧 benchmark 使用种子 811、853 换边，yan-v2 另取得四胜。三个种子是三张地图。[隔离竞技场逐局记录](analysis/arena/self-19417.jsonl) · [旧 benchmark 的四局](strategy-notes.txt)

## Claude 本地最终版与 yan-v3

截至 2026-11-22，Claude 本地最终版（SHA-256 `a3d54e4672458e53f5dfd47444b59a25326f5d1592575f968a1a59f26c98b4b0`）与已提交的 yan-v3（`ee2b1a51039aaea356e28c6e326d3ec4eb4c06cda054808f068f132447add737`）在三张地图换边直接交手。Claude 共 5 胜 1 负，场均资金领先 1042.18。六局全部打满 299 步，双方均无超过一秒的慢步。表中分差均从 Claude 一侧计算。

| 地图种子 | Claude 坐 0 号位 | Claude 坐 1 号位 |
| --- | ---: | ---: |
| 897419481 | +1559.10 | +2044.68 |
| 283554314 | +784.10 | −227.51 |
| 365960692 | +1142.65 | +950.07 |

首张图来自三方循环赛，第二批两张新图独立换边复测，Claude 为 3 胜 1 负、场均领先 662.33。复测种子取 ASCII 字符串 `claude-final-vs-yan-v3-confirm-0` 和 `claude-final-vs-yan-v3-confirm-1` 的 SHA-256 前四字节，按大端取模十亿。[首张图原始记录](https://github.com/Zhaoyi-Yan/carbon/blob/2292dc463ba3f5c4a8d84eb4139c0e63acd2d674/analysis/arena/live-adaptation-897419481.jsonl) · [新增四场原始记录](analysis/arena/claude-final-vs-yan-v3-confirm.jsonl)

## 多个 AI 如何一起改策略

1. **观察对手**。使用[前五回放逐局数据](analysis/top-five-games.jsonl)区分对手。按整张地图切分示范数据，双方与同局所有步数放在同组。遇到同类局面，分开检验空载前压、护送满载工人和经济扩张等选择。
2. **分别提出能对打的候选**。一名开发者或 AI 改己方走法，另一名专门寻找它的弱点，第三方保管本轮留用种子和对手版本。保存当前主力、历史版本和不同风格的对手。可用[对手路线实验](experiments/route_forecast.md)生成两个概率不同的独立候选。
3. **在同一批地图比较三方表现**。用 `--strategies v5=opponents/yan-v2.py temperature=/tmp/temperature.py tactical=/tmp/tactical.py` 运行三方循环赛。每个新种子会产生六局。先核对 `search_enabled`、耗时、异常，再看成绩。若只知道对某一个风格的胜率，下一轮保留该候选作练习对手。
4. **留出新图给晋级评测**。在开发前固定种子清单。对每个强对手分别看完整换边地图的胜负、平均资金分差、是否有单边崩盘，再决定是否更新主力。本仓库示例 19417 已用于调试及实测，后续当作回归图。

不同电脑或云任务可分不同种子，分别生成 JSONL。使用 `python3 arena_report.py shard-1.jsonl shard-2.jsonl --json` 合并；重复拷贝去重，同一策略名存在不同代码或同一场出现两套结果时会提示。报告只在两个出生位都正常结算后计入成对战绩与分差，剩余正常场次及弃权/环境错误单列。不同机器若为同一份已校验代码使用不同文件名，可加 `--by-hash` 按完整 SHA-256 合并。独立复跑同一种子时应各自汇总。相同棋盘并不固定神经网络搜索的时间预算或每个随机策略的结果。

## Codex Cloud

在 Cloud 任务中检出同一仓库提交，使用上面的安装命令和对局命令。CPU 包和裁判就绪后，对局执行本身可以离线；运行时不需要比赛网站账号。默认环境也可以请求访问已连接的 GitHub 仓库。[Codex Cloud 指南](https://learn.chatgpt.com/docs/cloud)

本次本地完整双座位对照用了提交 `7658efc4e6eeeef4066844a646b0c411e784645c`、种子 19417、本机 Apple M4、Python 3.13.11、NumPy 2.4.2 和 PyTorch 2.12.1。这次历史对手为公开的 Master3s2。v5 从 0 号位赢 121.69，从 1 号位赢 155.41；双方每场搜索 298–299 次。本机另在全新 Python 3.12 / NumPy 1.26.4 / PyTorch 2.5.1 环境中做了短局兼容性验证。[完整逐局记录](analysis/arena/local-19417.jsonl)

## 本地 yan-v6 对照

线上已验证的对手可用 `--baseline yan-v5`。本轮本地晋级文件可用 `--baseline yan-v6`，两者均校验完整 SHA-256。八图初筛和四图独立复核各自记录地图、座位、搜索用时与全部败局。[初筛](analysis/v6r-qual-05-summary.json) · [复核](analysis/v6r-confirm-05-summary.json)
