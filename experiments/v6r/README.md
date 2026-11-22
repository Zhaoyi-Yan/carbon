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
python experiments/v6r/artemis_fetch.py --help
python experiments/v6r/data.py /tmp/carbon-boards /tmp/carbon-artemis
python experiments/v6r/train.py /tmp/carbon-boards /tmp/carbon-parent-v6.py /tmp/elite.npz /tmp/elite-report.json
CARBON_DEBUG=1 python scripts/selfplay.py --candidate experiments/v6r/checkpoints/grove.py --baseline yan-v5 --seeds 481710710 --jobs 1 --output /tmp/carbon-grove-check.jsonl
```

[第一轮冻结清单](qualification-1.json) 记录候选哈希、八个新地图种子和通过条件。每张图换边，必须取得至少 12 场正常胜利、两个出生位至少各胜 5 场、平均分差为正。开发成绩与留图成绩单列。策略按墙钟时间调整搜索，复跑同一地图时，CPU 负载也可能改变走法。
