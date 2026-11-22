# 对手能抢到的树值多少钱

线上 yan-v2 的部分败局中，不少新树在种下后 18 步内易主。v5 训练树价值时已经对易主后的收入反号，因此这一实验检验更具体的问题：前五选手交手时，v5 的 18 步收益预测是否对争夺中的树有系统偏差。

从 77 场前五回放中排除 v5 原训练档案里出现过的地图布局或种子，剩下 63 张。按地图布局哈希固定划分，47 张用于拟合，16 张留出。每局取一个固定观测方，用对局当时可见的碳量重建碳图；每三步采样一次。若敌种植员到树的最短距离在四步内，按距离、双方最近种植员谁先抵达、剩余可收款期估算原模型的平均偏差。目标是未来最多 18 步实际发给两边的净现金，从当前所有者视角以 0.96 逐步折现。补偿量向零收缩，减少稀疏分组带来的波动。

训练有 17,677 个树状态，留出的 6,434 个树状态中，v5 的平均预测减实际为 +1.24 碳，校准后为 +0.72。均方根误差从 15.69 降到 15.65，平均绝对误差从 9.80 升到 9.88。逐树结果说明整体偏差很小。是否值得改变每轮的走法要看整局现金和换边结果。[独立地图与误差记录](contested_trees_results.json)

使用 `python experiments/contested_trees.py --output /tmp/carbon-contest-fit` 从冻结 v5 生成单文件，然后运行 `CARBON_DEBUG=1 python scripts/selfplay.py --candidate /tmp/carbon-contest-fit/contested-trees.py --seeds 31427 --jobs 2 --output arena-runs/contested-first.jsonl`。拟合目录和对局输出都使用新路径。

在种子 31427 的完整换边赛中，候选在 0 号位以 6404.29 比 6612.58 落败，在 1 号位以 5739.18 比 5814.01 落败。候选平均分差 −141.56，两局双方均无慢步。这种估值校准未达到晋级条件。[逐局实测](../analysis/arena/contested-trees-31427.jsonl) · [完整候选](https://github.com/Zhaoyi-Yan/carbon/blob/69f549d6b0f4e3995f339e3d325f03ebc4b690dd/submission.py)
