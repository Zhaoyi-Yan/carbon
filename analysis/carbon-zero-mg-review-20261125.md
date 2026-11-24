# Carbon-Zero M/G 首种子复审：MC 相对优势、泛化短板与 M-wide 对照

> **状态更新（2026-11-25）：** `e8a4151` 已补齐第二种子和 M-wide；M 未复现净增益，M-wide 未通过筛选。本文件的后续执行安排已由[最新路线调整](carbon-route-reset-20261125.md)替代。暂停当前 Zero 配方，不追加 wide 第二种子、40 图确认或百万步训练。以下内容保留为当时证据与原计划。

日期：2026-11-25（北京时间）。证据快照：`experiment/carbon-zero-b1-residual` 的 **`be792f64e3b25b392b1cd502e5b55cfbf11e0950`**。前置：[B1-fixed 复审及 M/G 协议](carbon-zero-b1fixed-review-20261124.md)。

**当前判断：M 比 G 更有希望的证据明显增强；M 超过冻结起点的证据尚未成立。继续已启动的第二种子 M/G，保持本轮设置。下一次新增训练，优先比较同样 16 局/批下的 2 图×4 复本与 8 图×1 复本，单独检验地图覆盖，而不是马上延长训练或增加一组超参数。**

本快照只有 seed `24000000` 的两臂结果。实验报告说 `24010000` 正在运行，本文没有把尚未提交的结果当作事实。本文完成了代码审阅、原始 JSON 复算及已保存校准轨迹的 NumPy 核验；没有运行新模型训练、重新对局或验证未入库的权重。发布基线仍为 `v11-yan-v8-elite-refit`。[本轮报告][report]

## 1. 本轮最重要的结果

### 1.1 主目标：M 自身有没有比 ZW0 更强？

以下均为候选与 ZW0 起点在**相同地图、席位、对手及随机流**上的配对差。每个对手 12 图×2 席；主指标是三个对手等权的终局效用变化 ΔU。区间用 100,000 次地图聚类 bootstrap，三对手、两席一起留在地图簇内。它们是条件于本次固定 checkpoint 的探索性区间，不包含重新训练带来的种子不确定性。[原始面板][panel] · [复算附件](carbon-zero-mg-review-20261125/statistics.json)

| 候选相对 ZW0 | 对冻结 ZW0 | 对 generic 无搜索 | 对 v6 无搜索 | 三对手等权 ΔU |
| --- | --- | --- | --- | --- |
| **M50** | +0.224 [−0.086,+0.539] | **−0.211 [−0.490,+0.022]** | +0.128 [−0.208,+0.456] | **+0.047 [−0.095,+0.184]** |
| **G50** | −0.202 [−0.380,−0.045] | −0.478 [−0.783,−0.164] | −0.058 [−0.265,+0.138] | **−0.246 [−0.381,−0.102]** |

M 的宏均值为正，但区间仍包含零。G 在当前面板的宏表现明显低于起点。应当继续区分“相对另一种训练方法更好”与“训练产生了净棋力增益”。

### 1.2 方法对照：M 比 G 好多少？

| M50 − G50 | ΔU | 95% 地图区间 | 地图均值为正 |
| --- | ---: | --- | ---: |
| 对 ZW0 | +0.426 | [+0.149,+0.738] | 10/12 |
| 对 generic | +0.267 | [−0.031,+0.527] | 10/12 |
| 对 v6 | +0.186 | [−0.078,+0.430] | 9/12 |
| **三对手等权** | **+0.293** | **[+0.212,+0.376]** | **12/12** |

这是比原报告只强调某一个对手更有用的发现：**每张地图先平均三对手和两席，M−G 的宏差在 12 张图上全部为正。** 它并非由一张极端大胜图支撑。仍不能把这 12 图当作 12 个训练种子，或把单种子结果升级为算法普遍性结论。

### 1.3 generic 的代价不能被宏均值掩盖

M 相对 ZW0 对 generic 的 ΔU 在 **10/12 张图为负**，两席分别为 **−0.254、−0.167**。当前负向结果分布较广，不只是某个出生位或一局翻车。与此同时，其区间上界尚略大于零，不能声称已经以当前样本确定退化的真实幅度。

M 的三对手宏 ΔU 是 7/12 图正、5/12 图负。删除任意一图，宏点估计仍为正，范围 +0.007 到 +0.088；删除两图的 66 种组合中，6 种不再为正。这是事后影响分析，**不替代原预算筛选规则**。

| 候选 | 对 ZW0 W–D–L / 平均分差 | 对 generic | 对 v6 |
| --- | --- | --- | --- |
| ZW0 | 13–0–11 / +383 | 16–0–8 / +551 | 5–0–19 / −932 |
| M50 | 17–0–7 / +510 | 13–0–11 / −7 | 7–0–17 / −618 |
| G50 | 10–0–14 / +206 | 8–0–16 / −146 | 4–0–20 / −941 |

M 相对 ZW0 的宏原始分差为 **−39.08**，宏胜率变化为 **+4.17 个百分点**；它们与 ΔU +0.047 并不矛盾。继续使用预定 U 排序，同时保留这些副指标。M 对 ZW0 有 1 局提前淘汰，已按真实裁判结果计入，其余面板对局正常结束，没有删掉提前终局。

## 2. 这次 M/G 对照本身基本成立

代码审阅未发现需要叫停第二种子的优势计算、动作概率、价值时序或参数共享错误。[M/G 实现][mgcode] · [完整对局采集][gamescode] · [优化器实现][optcode]

- 两臂从同一 ft3 Apollo/style 122 起步，T=.4，对手与 reference 都为冻结 ZW0；同 U/Phi、同优化器、学习率、正则和 decoder。
- critic 是单独新建的完整 Actor：ft3 trunk 加锁定的 `critic.pt` head/w，整体冻结。baseline 使用独立 critic 的表示，不借用正在变化的 learner trunk。
- 每批完整采完 16 局后才更新；两臂各采自己的 on-policy 轨迹。首批保存的逐局结果和现金轨迹完全一致，之后轨迹分开符合预期。
- 当前动作的 baseline 使用动作前观测和现金差。实际 sequential decoder 的条件 mask、动作和 old joint log-prob 在内存训练数据中一起保存并重算校验。
- 每局检查完整 GAE 在 λ=1 时等于 MC。两臂没有不同的 value loss、warmup 或优势标准化。

| 首种子训练事实 | M | G |
| --- | ---: | ---: |
| 实际采样 E | 52,329 | 52,329 |
| 完整批 / 对局 | 11 / 176 | 11 / 176 |
| **独立训练地图** | **22** | **22** |
| optimizer steps | 109 | 109 |
| 训练 W–D–L | 80–1–95 | 71–1–104 |
| 训练平均分差 | −60.14 | −64.08 |
| 最大重放 log-prob 误差 | 9.78e−6 | 7.15e−6 |

两臂都有同一局第 4 步双灭平局，其余 175 局正常结束。训练表现来自不断变化的行为策略，不是固定 checkpoint 的泛化成绩。

两个 10k smoke 实际各 14,352 E / 48 局，保存的逐局记录恰好是各自主跑的前缀，**不是额外种子或独立复验**。若它们确实按独立命令重跑，另计重复 smoke 的采样成本 28,704 E；两臂主跑合计 104,658 E。不能把确定性重复当作增加证据。

### 2.1 地图面板有一项需要记录的协议偏离

上轮文档写“沿用当前 12 张共同开发图”，实际新建的 `terminal-rl-mg-v1` 与旧 `evaluation-shared` 12 图**重叠为零**，ZW0 基线也重新评估了。

新面板内部配对有效；复算还确认，它与当前 M/G 训练图、所核对的 critic 拟合/留出图和早期 terminal 开发/确认图均无布局重叠。应如实称为**新的三对手共同开发面板**。它已参与首种子筛选，第二种子继续用它不增加独立地图数量，之后不能重新称为独立确认集。

## 3. 对机制的判断应如何更新

### 3.1 critic 用作减噪基线与用于未来预测，是两种要求

令 `B_t = Phi_t + b_t`，即把 shaped-return critic 转成对最终 U 的状态预测。当前 M 的优势为：

\[
A^M_t=U_T-B_t.
\]

在行为策略点、未裁剪的 on-policy 策略梯度中，B_t 与当前动作无关，满足 `E[grad log pi(a_t|s_t) B_t]=0`。因此，即使冻结 critic 随策略更新而不再准确，它首先改变的是估计的方差；不需要 baseline 完美，终局反馈仍来自真实 U。**这不表示当前 PPO clipped surrogate 加正则的整个多步优化过程对 baseline 完全不敏感。**

对 G，令剩余动作步数为 n，可由当前代码直接展开：

\[
A^G_t=-B_t+(1-\lambda)\sum_{j=1}^{n-1}\lambda^{j-1}B_{t+j}+\lambda^{n-1}U_T.
\]

如果 `e=B−V^pi_U`，其中 `V^pi_U` 是当前策略对最终 U 的真实状态价值（等价于 b 减去对应 shaped-return value），GAE 因 value 误差产生的差别为：

\[
-e_t+(1-\lambda)\sum_{j=1}^{n-1}\lambda^{j-1}e_{t+j}.
\]

当前状态的误差项可以作为动作无关基线消去，未来状态的误差则可能随当前动作变化。**一个能降低 MC 波动的冻结 critic，并不自动就是能在策略漂移后可靠 bootstrap 的 critic。** 这是 M 比 G 更有效的合理机制线索，尚未通过本轮实验与尺度、方差、裁剪和正则等因素分离。GAE 本身就是偏差与方差的权衡，而非只能正确或失效的二元选择。[GAE 原论文][gae] · [PPO 原论文][ppo]

λ=.95 时，显式终局样本的系数在剩 50 步时约 .081，剩 100 步时约 .0062。未来 B 同时携带长期信息，所以这**不是**“只剩这些比例的有效梯度”或“必然短视”的证明。附录在已保存校准轨迹上用实际 helper 验证了上述展开和 λ=1 恒等式，最大数值误差约 1.3e−15。

当前合理更新是：**在这套冻结 critic、固定对手和训练预算下，优先保留完整终局 MC 状态基线路线。** 不据此否定全部 GAE/PPO，不倒推历史 B1 失败的唯一原因，也不因 critic 开局 EV 低就砍掉开局策略梯度。

### 3.2 “优势均值差十倍”应从解释中撤下

报告的 −.13 与 −.015 是有符号均值，不能当作优势 std/RMS 或有效更新幅度。正负相互抵消就能使均值接近零。

首批相同数据给出了更直接的证据：

| 指标 | M | G |
| --- | ---: | ---: |
| signed mean advantage | −0.13059 | −0.01540 |
| 首个审计 microbatch 的 policy trunk gradient norm | 2.85580 | 1.23429 |
| 首批平均 old KL | 0.01940 | 0.01837 |
| 首批 clip rate | 0.22791 | 0.21998 |

梯度范数的这次比值约 **2.31**，并非均值比的 8.48；它又只是一个审计 microbatch，不是整个训练梯度方差。AdamW 与统一 gradient clip 进一步影响实际参数移动，当前证据不足以把差异解释为“G 更新弱十倍”，也不足以据此改 lr 或 anchor。

作为可离线核验的例子，**另一个零步校准数据集**上的 M/G signed means 是 −0.00188/+0.00220，RMS 却为 0.49186/0.17021，比值 2.89。该例证明均值与尺度必须分开报告，不能拿校准集的 RMS 替代缺失的训练 RMS。[优势复算附件](carbon-zero-mg-review-20261125/advantage_audit.json)

现有 `reference_KL_scaled` 也是按 microbatch 总和与 `rows_scale` 缩放后再平均的指标，不能直接与旧 B1 的 `anchor_kl` 数值比较约束强弱。

### 3.3 上轮诊断错误已修好

非残差分支已统一一维，长度不等的相关计算会报错；新输出有 checkpoint/critic/script SHA，以及逐局 V/Phi/U，能离线复算。真正的零步 `ZW0 actor + 预训练 critic` 已加入：全盘 MSE 为 **0.24192**，现金比较器为 **0.25248**；EV 为 **0.21006 vs 0.17887**。它支持当前 critic 在该校准样本上提供额外预测信息，不意味着对后续 M/G 策略仍然同样准确。[修复代码][calcode] · [修复后的数据][caldata]

ZW0/A10/A50 的 GAE–MC 相关现在分别为 .435/.417/.465。以前的 null 是广播错误，不能再解释为“GAE 没有定义”。跨 actor 的 EV 来自不同轨迹分布，也仍不能直接量化“冻结 w 因果修好了多少”。

## 4. 现在先把第二种子做完，不改变本轮问题

首轮预定预算规则仍然成立：M 的三对手宏 ΔU > 0，且每个对手区间上界都未低于零。100,000 次复算后 generic 上界为 +.022，原报告 4,000 次为 +.019，没有改变筛选结论。因此**复验 M/G 两臂**是遵守协议，不是给已经退化的 G 追加无限预算。

第二种子提交时只需追加同目录候选，保留当前 ZW0/M50/G50，不重跑旧对局、不更换地图或随机流；固定约 50k checkpoint，逐种子报告以下三项：

1. **M−ZW0 的三对手宏 ΔU**：净增益是否在两个训练种子方向一致？
2. **M−G 的三对手宏 ΔU**：估计器对照是否复现？
3. **逐对手与两席**：generic 的负向是否稳定，v6 改善是否复现？

| 第二种子结果 | 对本轮的决策 |
| --- | --- |
| M 两种子宏 ΔU 都为正 | 允许固定 M 两个 checkpoint 进入独立地图确认；仍不等于晋级。 |
| M 仍优于 G，但相对 ZW0 宏 ΔU 不为正 | 记录为“相对方法优势可复现，净棋力增益未复现”；不确认当前 M，不自动跑 100k。 |
| M−G 的优势也未复现 | 下调对 MC 相对优势的信心；保留两种子，继续停留在受控研究层面。 |

G 首种子宏均值已为负，单靠第二种子的一次反转不能满足原“同一臂两个种子都正”的确认门槛。所有比较应呈现两种子，不取最好 seed。两种子若共用 12 图，合并仍按 12 个地图簇重采样，且明确区间条件于这两个已训练 checkpoint。

独立确认的建议预算为**新冻结的 40 图×2 席×3 对手**，评估 ZW0 与 M 的两个固定 checkpoint，共 720 局。确认前提交 map/RNG/source/checkpoint manifest，并把成功条件写清：两种子宏点估计均正、两种子等权宏 ΔU 的地图区间下界 > 0，才称该固定组合的宏效用改善得到确认。逐对手退步另列；这仍不证明每个对手都改善。若区间未排除零，只记录未确认，不自动反复添图直到显著。

本面板的 generic/v6 均关闭搜索，也没有前六对手本体。因此即便确认通过，仍需再挑战冻结发布版 v8、验证 actor-only 导出一致性与运行约束，才有讨论替换发布文件的依据。当前 17/24 胜 ZW0 不能换算成线上排名提升。

## 5. 下一次新训练优先做 M-wide：扩大布局覆盖，不扩大对局数

### 5.1 新假设来自采样结构，而非观察一次败局后加规则

当前 collector 为 RLOO 设计：每批 2 张图，每图 2 席，每席 4 个随机复本，共 16 局。同起点复本服务于 leave-one-out return baseline。M 已换成冻结的状态 baseline，**不再要求同一地图/席位凑四局才能计算优势**，却继续继承了这个分组。

因此首种子的 52,329 E 虽然包含 176 局，却只覆盖 **22 个布局**。现在出现的种子敏感和跨对手差异，使“地图覆盖不足”值得用一次直接对照检验。它不是已被证明的失败原因；减少同图复本也会减少每张地图平均回报的估计精度，扩大覆盖并不保证获胜。

| 新对照 | 每批地图 | 每图席位 | 每席复本 | 每批对局 | 首种子 11 批布局数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 已有 M（控制） | 2 | 2 | 4 | 16 | 22 |
| **M-wide（新臂）** | **8** | **2** | **1** | **16** | **88** |

### 5.2 固定实施协议

这是**第二种子完成后的下一项研究对照**，不改正在运行的 M/G。若当前 M 已达到确认门槛，先完成并记录原候选确认；新增训练不能用来替代或回避原结论。

- 从同一 ZW0 actor 起点重新训练，只改变地图/复本分配。保持 M 优势、完整冻结 critic、ZW0 对手、U/Phi、T=.4、decoder、anchor=.1、entropy=.01、lr=3e−5、PPO clip=.2、gradient clip=.5、原 optimizer 与 loss 归一化公式。
- 不同时增加对手混合、在线 critic 更新、阶段门控、优势标准化或温度调度。M/G 没有必要再为 M-wide 追加一臂 G-wide。
- 控制用对应 seed 的既有 M checkpoint 和结果。第一 seed 为 `24000000`；只有新臂筛选通过，再用 `24010000`。若采集/优化器或来源发生实质变化，旧控制便不可直接复用，必须先记录并匹配这一变化。
- **按对应控制的完整批数匹配。** 先读取该 seed 既有 M 的 `B_seed`，新臂也运行 `B_seed` 批、每批 16 局，完成真实终局。首 seed 是 11 批；第二 seed 以其最终记录为准，不机械假设也是 11。
- 报告实际 E、早终局、优化步数和总时间。首 seed 的控制有一局 4 步平局，故即使新臂也是 11 批，E 和 minibatch 数也未必相同。准确称为“相同完整对局/批次预算的采样分配对照”，**不要称 E、局数、optimizer steps 三者全都严格匹配**；不为凑 E 临时再加一整批。
- 地图/随机流沿用原键：`namespace=('terminal-mg-train', seed)`，`layout=keyed_seed(namespace,batch,pair,'layout') & 0xffffffff`，`group=2*pair+seat`，两方随机流分别为 `keyed_seed(namespace,batch,group,member,'learner')` 与 `keyed_seed(namespace,batch,group,member,'opponent')`。旧 `pair=0,1`、`member=0` 的地图与流在 wide 中保留；新加入 `pair=2..7`，均只取 member 0。每批仍平衡两席。若分开采集子批，必须整批保持同一行为参数，再统一更新。
- 所有轨迹由新臂自身 on-policy 生成；共同随机流不代表可以把旧 M 轨迹拿来更新 wide。保存预先生成的布局和 RNG manifest，避免与开发/确认图重叠。
- 复用当前 `terminal-rl-mg-v1` 开发面板、既有 ZW0 和 M 结果，追加 `Mwide_s1`，不重跑整个历史候选表。

### 5.3 预先写入新臂的预算筛选

第一 seed 后，要求同时满足：

1. `M-wide − 同 seed M` 的三对手宏 ΔU 点估计 > 0；
2. `M-wide − ZW0` 的宏 ΔU 点估计 > 0；
3. 相对同 seed M，没有单对手 ΔU 的 95% 地图区间上界 < 0。

通过才运行第二 seed M-wide，并同既有第二 seed M 配对；否则这一数据分配对照停止，不追加 100k。这是下一轮的探索性预算规则，不追溯修改当前 M/G 的 gate，也不把未显著退化当作非劣证明。

两个 seed 都对 M 和 ZW0 方向积极，才讨论新臂独立确认。若旧 M 的 40 图确认已经打开，M-wide 后续必须另冻结未使用的确认图，不能把已用于判断/选型的图重新称为新测试集。

**为什么优先于立即混合对手：**它利用已经不再需要的 RLOO 分组约束，在对局预算内直接增加布局覆盖，且只动一个明确因素。若扩大布局后 generic 负向仍跨 seed 复现，再单独比较固定 50% ZW0 + 50% generic 的 M；v6 继续保持不入训练。那是后续另一轮问题，不把它与 wide 混在同一臂，也不直接启动大规模 league。

## 6. 低成本记录改进与 Codex 交付清单

以下记录缺口**不否定当前 M/G 结果，也不要求为补日志重跑长训**。在下一次新运行时顺手补齐即可：[实现位置][mgcode]

- 保存按 step 统计的 mean/std/RMS/mean-absolute advantage、正负比例及阶段汇总；当前每局算了 std 却没有写出。移除硬编码 `zero_groups=0`，或标记不适用。
- 保存独立 critic 组合参数 hash、训练前后不变检查；当前 source SHA + style + head SHA 已明确来源，但尚无整体参数 hash。
- 在训练机保留压缩 commands/conditional mask/old log-prob/value 轨迹，仓库提交 manifest/hash 与紧凑统计。当前 GitHub 只有 episode 结果、现金轨迹和随机流，不能单靠 JSON 完全重放训练梯度。
- 单独记录 baseline 前向与优势构造耗时，并记端到端时间。现有 collection + optimization 的计时没有覆盖这部分。
- 协议把首 seed gate、第二 seed gate、确认预算和选型规则完整落盘。修正文档中“固定现金基线不可低于”“冻结 w 已因果恢复 critic”“均值就是尺度”等过强表述；旧结果保留并注明被后续复审修正。

| 顺序 | 下次应交付的内容 | 决策用途 |
| --- | --- | --- |
| 1 | 当前 seed 24010000 的 M/G 固定终点、完整协议、同面板结果 | 完成本轮已有承诺，区分 M−G 与 M−ZW0 的复现。 |
| 2 | 两种子分别及等权 ΔU、逐对手/两席、地图区间，保留 generic 负例 | 按原门槛决定是否独立确认，避免挑最佳 seed。 |
| 3 | 通过时冻结旧 M 确认面板并执行；未通过时明确归档 | 把当前结果闭环，不以新想法掩盖未确认。 |
| 4 | 下一项新训练只做上述 M-wide 首 seed；通过筛选才第二 seed | 检验更多布局是否比同图复本更有效，保持可解释预算。 |

## 7. 本次复算与复现

- [stats_recompute.py](carbon-zero-mg-review-20261125/stats_recompute.py)：校验 23 个输入逐字节等于固定提交，复算 U/WDL/分差、地图区间、配对、采样预算和布局重叠。
- [statistics.json](carbon-zero-mg-review-20261125/statistics.json)：完整输入 SHA、方法、逐席/逐图结果及敏感性分析。固定 bootstrap RNG 为 `20261124`，100,000 次。
- [advantage_audit.py](carbon-zero-mg-review-20261125/advantage_audit.py)：从固定源码提取实际 full-game GAE helper，用已保存六位小数校准轨迹核验统计、λ=1 恒等式和有限时域展开；不依赖 PyTorch 或模型权重。
- [advantage_audit.json](carbon-zero-mg-review-20261125/advantage_audit.json)：实际运行结果；明确这些轨迹属于既有校准数据，不是当前 M/G 训练数据。

```sh
git fetch origin experiment/carbon-zero-b1-residual
git worktree add --detach ../carbon-mg-evidence be792f64e3b25b392b1cd502e5b55cfbf11e0950
python analysis/carbon-zero-mg-review-20261125/stats_recompute.py ../carbon-mg-evidence /tmp/mg-statistics.json
python analysis/carbon-zero-mg-review-20261125/advantage_audit.py ../carbon-mg-evidence /tmp/mg-advantage-audit.json
```

[report]: https://github.com/Zhaoyi-Yan/carbon/blob/be792f64e3b25b392b1cd502e5b55cfbf11e0950/analysis/carbon-zero-terminal/b1/README.md
[panel]: https://github.com/Zhaoyi-Yan/carbon/tree/be792f64e3b25b392b1cd502e5b55cfbf11e0950/analysis/carbon-zero-terminal/b1/mg/evaluation
[mgcode]: https://github.com/Zhaoyi-Yan/carbon/blob/be792f64e3b25b392b1cd502e5b55cfbf11e0950/experiments/carbon_zero/terminal_mg.py
[gamescode]: https://github.com/Zhaoyi-Yan/carbon/blob/be792f64e3b25b392b1cd502e5b55cfbf11e0950/experiments/carbon_zero/terminal_games.py
[optcode]: https://github.com/Zhaoyi-Yan/carbon/blob/be792f64e3b25b392b1cd502e5b55cfbf11e0950/experiments/carbon_zero/terminal_rloo.py
[calcode]: https://github.com/Zhaoyi-Yan/carbon/blob/be792f64e3b25b392b1cd502e5b55cfbf11e0950/experiments/carbon_zero/terminal_calibration.py
[caldata]: https://github.com/Zhaoyi-Yan/carbon/blob/be792f64e3b25b392b1cd502e5b55cfbf11e0950/analysis/carbon-zero-terminal/b1/calibration/audit-8maps-fixed.json
[gae]: https://arxiv.org/abs/1506.02438
[ppo]: https://arxiv.org/abs/1707.06347
