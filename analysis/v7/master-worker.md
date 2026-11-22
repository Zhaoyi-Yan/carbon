# 给根搜索增加 Master 工人走法

Soundings 和 Fairlead 共享同一套动作主干权重。新插件复用每步已算好的四个旋转隐藏图，保留各方向的区别。在工人当前位置计算五个方向的分数，又读取五种落点各自的隐藏通道。将两者合在一个小型非线性网络中，按公共局面混合 Master7sD/E 的动作偏好，最后把四次旋转还原并平均。新增参数21,405字节；实测新增推理中位0.47毫秒、P95约1.31毫秒。

使用原先的42张训练图、13张整图留出。D为17/5、E为9/3、B2为16/5，B2提供共享动作训练样本。用训练集内部按图选择训练轮次后再在全部42图拟合；两场目标图均排除在新模型训练外。旧工人头使用四向平均后的隐藏特征和线性评分，表中让它直接知道真实教师标签作为强对照。新专家运行只看本席位的公开状态和迷雾碳信息。

| 8张 D/E 留出图上的动作一致率 | 冻结 Fairlead | 上一版工人头，给定真实身份 | 新方向专家，观测软分类 |
|---|---:|---:|---:|
| 全部工人，3682次 | 74.34% | 74.66% | 74.55% |
| 重载、距自己基地≤4格，264次 | 80.30% | 81.82% | 81.44% |
| 轻车靠近敌基地或载货车，1051次 | 71.17% | 71.46% | 71.08% |
| 富矿可原地开采，762次 | 80.05% | 80.71% | 79.79% |
| 第75步后重载且两格内有可致死轻车，201次 | 77.61% | 77.11% | 76.62% |

新专家在D的2282个工人动作上为75.55%，原根部74.85%；E的1400个动作为72.93%，原根部73.50%。按整图配对，相对用训练集温度校准后的原根部，动作负对数似然平均改善0.01779，95%按图重采样区间为0.00862至0.03007。第一选择命中率的区间跨越零。D/E软分类器平衡准确率54.71%，原始准确率50.56%。只靠同一局面的队伍经济和棋子分布，很难判断作者具体用了D还是E。

轻车施压的二分类指标按「下一手接近敌基地或一辆比自己至少重35的采集员」定义；新专家该动作的精确率90.55%、召回率91.63%。在富矿上确实留下采矿的召回率89.91%，冻结原根部为95.58%。这些指标衡量行为复现，多步封路或保护送货的经济成效仍要由对局验证。

在 QQ-v6 对 Master7sE 的独立截图局，采样工人命中从65.15%提高至66.60%；轻车施压61.04%→65.06%，重载近库19/21→18/21。使用原裁判局面与正常搜索入口，仅将测试宽度固定为5、深度为2：第90步和第164步各有一个新联合走法实际进入搜索，分别改变1名、3名工人。两次搜索最终都选择了别的走法。候选沿用原本的基地招工、工人落点协调和已锁定的回库动作。这个验证证明搜索能用它；整局胜率未测。两份资格候选均保持冻结。

复现用 `master_worker_data.py` 读取此前55图清单和官方回放，`master_worker_fit.py` 拟合与验证，`master_worker_audit.py` 做按图对照，`master_worker_smoke.py` 验证两个导出及目标图分支。单文件生成器为 `master_worker.build(code)`，参数保存在同目录 `master-worker-model.npz`。

```sh
nice -n 15 python experiments/v7/master_worker_data.py --cache ../v7-data/master-cache --out ../v7-data/master-worker --screenshot ../v7-data/starvation-4063.html.gz
nice -n 15 python experiments/v7/master_worker_fit.py --data ../v7-data/master-worker --cache ../v7-data/master-cache --output analysis/v7/master-worker.json --model experiments/v7/master-worker-model.npz
nice -n 15 python experiments/v7/master_worker_audit.py --data ../v7-data/master-worker --cache ../v7-data/master-cache --model experiments/v7/master-worker-model.npz --report analysis/v7/master-worker.json --output analysis/v7/master-worker-audit.json
nice -n 15 python experiments/v7/master_worker_smoke.py --target ../v7-data/starvation-4063.html.gz --output analysis/v7/master-worker-smoke.json
```
