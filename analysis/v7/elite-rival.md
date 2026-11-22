# 根据对手已走动作选择高段预测

QQ-v6 有 23 行教师网络策略嵌入。旧版在我方使用 Apollo 第21行和 Artemis 第22行，给敌方分配的根部及模拟身份只取0–20。本实验把高段两行加入敌方动作预测。敌方每个仍存活工人的上步动作由两次公开位置推断。比较旧预测和两个高段预测给予该动作的对数概率，累积近期证据后才启用获胜的高段行。两位老师共享网络，预测时选择 Apollo 行不代表确认了对手身份。

由 `elite_rival.build(code)` 可生成与既有候选独立的单文件策略。根部按证据融合旧预测与高段预测，模拟的大网络使用选中嵌入。模拟的小网络只有21行，始终沿用同一席位原有的0–20行，再用大网络参考棋盘校正。测试在仿真搜索的正常入口实际运行了这个组合。

| 独立确认集 | 原 QQ-v6 敌方预测 | 含高段两行的预测 |
|---|---:|---:|
| 6,282个工人动作命中率 | 80.707% | 80.675% |
| 工人平均负对数似然，越低越好 | 0.49657 | 0.49654 |
| 316个可招工基地决策命中率 | 81.329% | 81.013% |

6张地图均未参加该教师网络的原训练，也未用于本实验参数选择。逐图双席评分，包含6个 Artemis 侧以及 RyanX、SpongeBaby、SiegeGuard、Master7sE。实际共648个回合决策，选择 Apollo 行15次、Artemis 行32次、旧预测601次，仅改变6个工人的第一选择。额外耗时中位5.9毫秒，P95为14.7毫秒。单步证据没有显示可采纳的预测增益。云上或本地的整局胜率仍未测定。

初版在另3张新地图、1张旧网络曾用于选模的 Apollo 地图、以及 QQ-v6 对 Master7sE 截图局上评估，486个决策只切换7次，没有改变工人第一选择。之后为两个高段候选添加180度对称，要求至少19个衰减后的有效动作，近期每动作优势达到0.028 nat才切换。确认集结果按该固定版本一次计算。每图取第48、166、246步开始的24步，各用前6步预热路由；两侧各自的迷雾估计从开局按真实自身历史预热。所有比赛和种子见数据文件。

```sh
nice -n 15 python experiments/v7/elite_rival_check.py --recent ../v7-data/surrogate-holdout --historical ../v6r-data/artemis --screenshot ../v7-data/starvation-4063.html.gz --initial --output analysis/v7/elite-rival-initial.json
nice -n 15 python experiments/v7/elite_rival_check.py --recent ../v7-data/surrogate-holdout --historical ../v6r-data/artemis --confirm --output analysis/v7/elite-rival.json
```
