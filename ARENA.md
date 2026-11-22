# 本地与云端对战

竞技场直接使用仓库内的原版 Carbon 裁判。每个策略在独立进程内持有自己的局内状态，两个座位收到各自可见的碳量。给出三个以上的策略时，每一对都会在相同地图上交换座位。

## 开始一场比赛

已有 NumPy、jsonschema 和 CPU 版 PyTorch 的 Python 环境可直接使用。需要对齐赛事库版本时，用 Python 3.12 执行安装脚本。Linux 安装 CPU 专用 PyTorch。默认虚拟环境在仓库根目录。

```sh
bash scripts/setup-arena.sh
CARBON_DEBUG=1 .arena-venv/bin/python arena.py \
  --strategies v5=submission.py Master3s2=opponents/claude-AlphaCarbon-Master3s2.py \
  --seeds 19417 --steps 300 --jobs 1 --output arena-runs/seed-19417.jsonl
python3 arena_report.py arena-runs/seed-19417.jsonl
```

输出文件须使用新名字，以便保留旧战绩。每局结算后立即写一行。记录包含策略哈希、裁判哈希、地图指纹、出生位、双方现金、搜索次数和失败原因。输出顺序可以随并行任务的完成顺序变化。普通平局与正常胜负均返回退出码 0；存在弃权或环境错误会返回 2。

初始化限时 20 秒，每步基准 1 秒，单步最多 3 秒，累计额外时间最多 5 秒，最多 3 次慢步。计入进程通信与本机调度时间。仅运行可信策略；本工具不隔离网络、磁盘和机器 CPU/内存配额。环境配置沿用上游 randomSeed，留图评估适用于按正常观测决策且未用种子反推资源图的策略。

## 多个 AI 如何一起改策略

1. **观察对手**。使用[前五回放逐局数据](analysis/top-five-games.jsonl)区分对手。按整张地图切分示范数据，双方与同局所有步数放在同组。遇到同类局面，分开检验空载前压、护送满载工人和经济扩张等选择。
2. **分别提出能对打的候选**。一名开发者或 AI 改己方走法，另一名专门寻找它的弱点，第三方保管本轮留用种子和对手版本。保存当前主力、历史版本和不同风格的对手。可用[对手路线实验](experiments/route_forecast.md)生成两个概率不同的独立候选。
3. **在同一批地图比较三方表现**。用 `--strategies v5=submission.py temperature=/tmp/temperature.py tactical=/tmp/tactical.py` 运行三方循环赛。每个新种子会产生六局。先核对 `search_enabled`、耗时、异常，再看成绩。若只知道对某一个风格的胜率，下一轮保留该候选作练习对手。
4. **留出新图给晋级评测**。在开发前固定种子清单。对每个强对手分别看完整换边地图的胜负、平均资金分差、是否有单边崩盘，再决定是否更新主力。本仓库示例 19417 已用于调试及实测，后续当作回归图。

不同电脑或云任务可分不同种子，分别生成 JSONL。使用 `python3 arena_report.py shard-1.jsonl shard-2.jsonl --json` 合并；重复拷贝去重，同一策略名存在不同代码或同一场出现两套结果时会提示。报告只在两个出生位都正常结算后计入成对战绩与分差，剩余正常场次及弃权/环境错误单列。本地与云端在同一种子的独立复跑应各自汇总。相同棋盘并不固定神经网络搜索的时间预算或每个随机策略的结果。

## Codex Cloud

在 Cloud 任务中检出同一仓库提交，使用上面的安装命令和对局命令。CPU 包和裁判就绪后，对局执行本身可以离线；运行时不需要比赛网站账号。默认环境也可以请求访问已连接的 GitHub 仓库。[Codex Cloud 指南](https://learn.chatgpt.com/docs/cloud)

本次本地完整双座位对照用了提交 `7658efc4e6eeeef4066844a646b0c411e784645c`、种子 19417、本机 Apple M4、Python 3.13.11、NumPy 2.4.2 和 PyTorch 2.12.1。v5 从 0 号位赢 121.69，从 1 号位赢 155.41；双方每场搜索 298–299 次。本机另在全新 Python 3.12 / NumPy 1.26.4 / PyTorch 2.5.1 环境中做了短局兼容性验证。[完整逐局记录](analysis/arena/local-19417.jsonl)
