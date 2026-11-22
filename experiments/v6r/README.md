# yan-v6 换边评测与高手回放

本轮的冻结对手是线上已验证的 yan-v5，提交文件 SHA-256 为 `0a413a11f80f6872a5c8f0d313061d7bf11121ae16fbdad97247ff3cb15242e5`。上一轮 yan-v6 的 SHA-256 为 `1995bbcc07592296e6fe6f816412acccb2d39ea2c79787cd937a210513f69afa`，可从 Git 修订 `9907fc5` 的 `yan-v6.py` 提取。

- [两张旧图的现金、树收入、采集与回库核账](../../analysis/v6r-diagnostic.md)
- [榜首、第二名与 yan-v5 的 217 场公开录像](artemis.md)
- [开发地图、逐轮留出地图和预设胜负门槛](panels.json)

## 第一份候选

[grove.py](checkpoints/grove.py) 是第一轮留图赛前冻结的完整策略。它在回放训练中给 Apollo 和 Artemis 各自独立的风格向量。搜索用双方盘面识别是否要切换风格，用小网络加上大网络对同一对手应手的校正，推演多条后续走法。两座基地开局先观察一个回合，再分别招一采集与一种植。比较候选终局时，使用载碳兑现价值和已有树剩余收益。模型对树的生命、当前产出及周围工种使用合法的盘面信息。

经济估值使用当时下载的 215 份回放，训练和验证文件逐一列在 [assets-model.json](assets-model.json)。留出集上，载碳兑现比例的加权绝对误差从常量预测的 0.199 降到 0.131，树未来收益从 17.44 降至 11.34。预测未来收益的误差不能代替实战。

两个高手的全网微调按初始地图去重，验证集保留对每位高手较晚的六场。四轮后，在 8,364 个有标签的高手动作上，命中率从 77.51% 上升到 79.29%。旧底座曾用较早的公开高手录像，完整对打使用本轮新生成的地图。[训练与验证清单](elite-validation.json)

以下命令从仓库根目录执行。生成和训练可使用 PyTorch。提交文件自带权重，比赛推理使用 CPU。

```sh
git show 9907fc5:yan-v6.py > /tmp/carbon-parent-v6.py
python experiments/v6r/artemis_fetch.py --data /tmp/carbon-artemis
python experiments/v6r/data.py /tmp/carbon-boards /tmp/carbon-artemis
python experiments/v6r/train.py /tmp/carbon-boards /tmp/carbon-parent-v6.py /tmp/elite.npz /tmp/elite-report.json
CARBON_DEBUG=1 python scripts/selfplay.py --candidate experiments/v6r/checkpoints/grove.py --baseline yan-v5 --seeds 481710710 --jobs 1 --output /tmp/carbon-grove-check.jsonl
```

[第一轮冻结清单](qualification-1.json) 记录候选哈希、八个新地图种子和通过条件。每张图换边，必须取得至少 12 场正常胜利、两个出生位至少各胜 5 场、平均分差为正。开发成绩与留图成绩单列。策略按墙钟时间调整搜索，复跑同一地图时，CPU 负载也可能改变走法。

## 第二份候选

[canopy.py](checkpoints/canopy.py) 在两个新开发种子（180798099、605220840）换边对 v5 四战全胜。第一张两个出生位分差 +154.63、+1020.06，第二张为 +1104.17、+1210.54。第二份候选在第一份候选之外保留了两个高手风格的联合动作供搜索比选。在最有希望的两个分支上，它用便宜的小模型再推演至多 14 步，比较后续可兑现的碳与树收入。在携碳工人遇敌时，另用游戏引擎模拟对方的几种应手。Artemis 面对首轮双采集时，19 局全从格号较大的基地出种植员，这条观察用于双采开局的基地分工。[四局比分](../../analysis/v6r-dev-canopy.jsonl) · [第二轮八图及冻结哈希](qualification-2.json)

新的对局可在 selfplay 命令上加 `--record-actions`。结果行的 `actions_and_post_cash` 保留原版裁判每回合收到的两组动作和结算后资金，可用地图种子在同版裁判中逐步重放。动作记录在两位策略均已结束该回合的决策后写入。

[利用公开树收入推断雾区碳量](sensing.md) · [接触时应手的一步反事实](reply-stress.jsonl) · [己方旧版本在公开比赛中的行动预测](rival-validation.json)

第一份候选在八图换边确认中为 9 胜 7 负，场均 +27.30。0 号位 3 胜、1 号位 6 胜。两张图双胜，一张图双败。全部正常结束，未达到赛前的 12 胜及单座位胜数要求。[16 局原始比分](../../analysis/v6r-qual-01-grove.jsonl) · [逐图、分座位与用时](../../analysis/v6r-qual-01-summary.json)

## 第三份候选

[zenith.py](checkpoints/zenith.py) 专门处理新种的树尚无历史收入时的估值。根据合法观测估计四邻碳量，扣除相邻树争用和静止采集员占位，预测结算前归属这棵树的净收益。45 张整图留出的 2,560 棵新树上，平均绝对误差从 33.46 降到 25.16。局内方案还允许推迟一笔招聘，在同一回合用 20 资金种下第一棵值得种的树；其他走法仍交由搜索比较。[幼树经济估值](sapling.md)

前两份候选在新图的 0 号位遇到更多败局。第三份候选在 0 号位把坐标水平反射后交给同一网络，并在返回裁判前还原左右方向。玩家编号和结算顺序保持原样。30 个回放局面的原始动作与反射动作经同一模拟器推演，工人、钱、树和地面碳均核对一致。这次检验覆盖坐标变换，胜负须用独立比赛判断。[第三轮冻结清单](qualification-3.json)
