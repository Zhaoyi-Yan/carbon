# 第190步分叉的对手近似与独立验证

Artemis 的公开策略 ID 为 `226891089ba7f54b198c8f8dc771dfc2`，原 v6r 共享网络对应嵌入行22。Apollo 的 ID 为 `dd1f117a64c676225f0c6333b1cdd1e7`，对应行21。推理权重逐项与冻结的 `elite.npz` 一致，其 SHA-256 为 `b353dc54334334f6284f7016329a39b59a5bb955b1841b8427e9a9e2a5919283`。每步读取本席迷雾可见碳与公开棋子，按网络给出的动作分数同时分配己方落点，并检查招募额度。每个分叉都在第0至189步用该侧真实动作预热，再随当前盘面重新决策。

验证取冻结训练和原选模验证集之后的9张独立 Artemis 地图。逐局校验参赛席位和地形，并从全程验证排除本次分叉的目标地图。第190至298步共981次决策、9440个工人动作。验证每次决策之后恢复专家的实际历史，适合测局部动作和交货误差；接手对弈109步后的分布偏移仍须单列。[锁定的公开回放快照](https://github.com/JUZI1/carbon-league-replays/tree/3d96b8d032a102c297968e40911d893c9abc160d)

| 同九张 Artemis 图 | Artemis 替身 | Apollo 加固定种子采样 |
|---|---:|---:|
| 单工人动作完全相同 | 79.82% | 77.42% |
| 载货至少40、离基地最多3格的行动一致率 | 83.26% | 82.65% |
| 招工精确率 / 召回率 | 92.31% / 81.25% | 90.36% / 78.12% |
| 有至少一笔本方交货的回合，精确率 / 召回率 | 88.03% / 93.98% | 88.73% / 94.74% |
| 预测本方工人向本方基地交货总额（实局36042.44） | 37156.38 | 36291.64 |

直接切换两个确定性嵌入仅改变69/9440个工人决策。改为 Apollo 行加固定温度0.55的 Gumbel 采样后，923/9440个工人决策与 Artemis 替身不同，619/981个回合的联合决策不同。随机数由地图种子、回合和座位决定，同一地图上各个策略分支得到相同随机数。两者共享网络，结果反映专家近似对手的敏感度。两组都低估部分招工，Artemis 各图自有交货额预测/实局在0.84至1.20之间。官方策略源码未知，独立验证无法证明近似对手在长期对抗中的行为与官方完全相同。

调用 `surrogate_variation.make(name, seed, config)` 可得到 `artemis`、`apollo-varied` 或原始确定性 `apollo`。调用 `observe(obs, own_historical_action)` 累积历史，再用 `act(obs)` 取实时决策。包内验证文件另列12张曾用于旧模型挑选训练轮次的图。

```sh
python experiments/v7/surrogate_fetch.py --data ../v7-data/surrogate-holdout
python experiments/v7/surrogate_check.py --historical ../v6r-data/artemis --recent ../v7-data/surrogate-holdout --output analysis/v7/surrogate-validation.json
python experiments/v7/surrogate_variation_check.py --recent ../v7-data/surrogate-holdout --output analysis/v7/surrogate-variation.json
```
