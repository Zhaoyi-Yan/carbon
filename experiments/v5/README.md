# yan-v5 对战实验

对手为维护者已提交的 yan-v4。其原文件保存在 `opponents/yan-v4.py`，SHA-256 为 `a3d54e4672458e53f5dfd47444b59a25326f5d1592575f968a1a59f26c98b4b0`。比赛当前主力仍是 yan-v4。实验分支上的 submission.py 对应下表第一版自适应候选。

[实验预登记](design.json)固定了 20 个策略 ID、当时排名和开发、验收种子。[官方快照一](https://github.com/JUZI1/carbon-league-replays/tree/685b50c0d4acedc12d67370b58b3e311f24faed8)、[官方快照二](https://github.com/JUZI1/carbon-league-replays/tree/1bbadc777c42f20537e93637a1109270b0b18b23)及仓库 replays/top-five 共提供 272 张不同地图的完整对局。前二十共出场 471 次。每步只用本方能看到的信息建立碳量估计。训练动作取下一帧的已执行动作；另以未来真实资金分差训练估值。训练和验证按地图和种子隔离。

| 候选 | 构建参数 | 已完成的开发测试 |
| --- | --- | --- |
| portfolio | heads.npz，默认参数 | 两图换边 3 胜 1 负，场均 +456.16 |
| countersearch | heads.npz，--frontier --own .7 --value .85 | 相同两图换边 3 胜 1 负，场均 +282.56 |
| rotation | heads.npz，--rotated --own .8 --value .5 | 以独立 JSONL 记录 |
| cartographer | landscape.npz，--fog --territory --rotated --own .8 --value .57 | 以独立 JSONL 记录 |

portfolio 在同一神经网络上拟合 20 组采集、种植和招募动作。根据对面尚存单位的真实移动和场上采集、种植比例估计对方风格，并按剩余回合、双方资产、人员和对手阵型选择己方策略。搜索同时保留原始策略和另一种风格提出的动作。countersearch 加入估计两种敌方应对的单步战术候选。rotation 在四个朝向评估当前局面，反向映射行动方向。cartographer 另加入资源控制和公开树收入推断。所有开发局及其失败均保存在 analysis/arena 下。

早期对照只纠正了 v4 风格名称与嵌入的偏移。旧模型有 120 个名字和 121 行嵌入，Claude 在两者末尾分别追加数据，导致其命名的 Apollo 使用前一位 1C-Testv2 的行。仅切到真正的 Apollo 行在开发图 128484274 换边两负，分差为 −757.88、−735.99。

## 训练与复现

安装仓库的 arena 环境。下列参数中的路径指向两个固定快照解压后的目录。extract 会忽略不完整、与前二十无关的对局，本地缓存每局只需提取一次。

```sh
python experiments/v5/extract.py /tmp/v5-shards /path/to/current /path/to/older replays/top-five
python experiments/v5/fit.py /tmp/v5-shards /tmp/heads.npz /tmp/validation.json
python experiments/v5/build.py experiments/v5/heads.npz /tmp/portfolio.py
CARBON_DEBUG=1 python scripts/selfplay.py --candidate /tmp/portfolio.py --seeds 128484274 611214248 --jobs 4 --output /tmp/portfolio.jsonl
```

[验证明细](validation.json)包括 214 张训练地图、58 张验证地图。按已知教师选择动作头时，验证动作准确率由 76.89% 升至 77.64%。其中 29 张地图出现在 v4 的旧训练池中。在剔除这些地图的 [独立复核](validation-independent.json)中，剩余 24,015 个单位动作的准确率从 79.56% 升至 80.47%。未来 48 步资金分差变化的平均绝对误差为 268.46，保持当前分差的预测误差为 505.64。真实换边对局单独计分。

完整验收要求冻结同一文件，在开发时封存的四张地图的两个出生位都正常战胜 yan-v4。
