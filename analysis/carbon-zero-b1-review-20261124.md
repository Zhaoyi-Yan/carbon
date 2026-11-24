# Carbon-Zero B1 复审：先验证学习信号，再扩大更新

> **后续状态：**`d6008ed`已完成固定w/B2修复及B1-fixed两种子50k实验。当前接续入口为[B1-fixed新结果复审与M/G对照](carbon-zero-b1fixed-review-20261124.md)。新结果没有支持继续扩大B1-fixed；优势诊断另有广播错误，已附独立复现。以下保留为上一轮设计，不要重复执行已完成的P0/P2。

2026-11-24。状态：**研究复审及下一轮执行协议；尚无新训练或晋级结果。**

本次核对 `experiment/carbon-zero-b1-residual` 的不可变快照 **`afcfe81712676dcc0d680e47881db6ecc71ea55b`**，代码提交 `66694a0`，其前置完整终局实验为 `868a0abe5d840f84614cfa437452e9b0f921b468`。下文的“已发现”来自该快照的代码、逐局 JSON 和 critic 留出预测；“建议”均尚待 Codex 实施。当前发布仍是 `v11-yan-v8-elite-refit`。

**接续决定：保留 B1 作为研究主线，暂不放宽 anchor、不提高 lr、不直接把 critic 接入旧搜索。先修正固定现金基线与 B2 诊断的实现，补齐混合效用评价和训练过程中的 critic 校准；下一轮从 ZW0 重启受控的 B1-fixed。** 如果确认 GAE 对 bootstrap 误差敏感，再试完整终局回报配状态价值基线。

## 1. 本轮真正的新信息

1. **B1 的初始化有价值，但“止住退化”尚未证实。** 残差 critic 在少量独立地图上比现金基线更准确；策略层面尚未证明超过 ZW0，也未完成事前定义的非劣检验。
2. **主报告遗漏了会改变算法排序的指标。** 原始 JSON 已保存混合效用 U，报告主要比较的是未截断平均分差。两者在多个对照上方向相反。
3. **“固定现金基线”的实现只在预训练阶段固定。** PPO 把现金系数交给了优化器；历史 B1 不能等同于严格固定基线实验。
4. **B2 的完整训练路径还有一个静态可定位的错误。** detach 后价值项对 trunk 没有梯度，诊断代码却会拼接空梯度列表。
5. **最重要的机制缺口是持续校准。** 初始化在 ZW0 上拟合好的 critic，进入 PPO 后改用截断 GAE 目标，而且共享表示和策略都在改变。现有结果没有回答 10k/50k 时它是否仍能正确预测当前策略的终局结果。[原报告][b1] · [训练器][tppo] · [PPO][ppo]

这轮已执行的是原始数据复算，没有重新训练策略、没有新的对局，也没有读取 Codex 机器上未入库的 actor 权重。本环境没有 PyTorch，因此涉及训练执行的代码发现属于静态审查，不能写成“B2 实跑失败”或“修复已验证”。

## 2. 先按实际训练目标重新看结果

正常结束时，当前训练目标是：

\[
U(D)=\tfrac12\operatorname{sign}(D)+\tfrac12\tanh(D/1500).
\]

提前淘汰沿用现有裁判胜负效用。这个 U 是项目既定的本地混合目标，不能冒称为平台未公开软分差公式的精确实现。它包含胜负和有界分差，因此**平均原始分差增加，不蕴含平均 U 增加**。[前轮协议][previous]

### 2.1 B1 相对同图、同席、同随机流的冻结 ZW0

下面不是增加新局后的结果，而是重算已有数据。区间使用地图聚类、100,000 次 bootstrap，固定 RNG 20261124；均为未作多比较校正的探索性 95% 区间，条件于这一个训练种子。

| 候选 | 对手 | 配对 Δ 分差 | 配对 ΔU［95% 区间］ | 胜率变化 |
| --- | --- | ---: | --- | ---: |
| B1-10k | ZW0 | −132.06 | −0.037［−0.357, +0.278］ | 0 pp |
| B1-10k | generic 无搜索 | +254.93 | −0.026［−0.315, +0.273］ | −8.33 pp |
| B1-10k | v6 无搜索 | +205.34 | +0.086［−0.164, +0.334］ | +8.33 pp |
| B1-50k | ZW0 | +42.92 | +0.185［−0.205, +0.560］ | +16.67 pp |
| B1-50k | generic 无搜索 | +40.76 | −0.086［−0.503, +0.299］ | −8.33 pp |
| B1-50k | v6 无搜索 | +230.66 | +0.045［−0.143, +0.257］ | +4.17 pp |

尤其需要保留两个反例：B1-50k 对 generic 的分差 +41、效用却 −0.086；B1-50k 相对 A50 对 v6 的分差 **+503.29**，效用却 **−0.074**，胜率少 8.33 个百分点。A50、C100 对 v6 也各有“分差相对 ZW0 下降、U 上升”的情况。因此不能从原来的分差表推导“其余 A/C 都退化，只有 B1 保住棋力”。[逐局来源][evaldata] · [本轮复算数据](carbon-zero-b1-review-20261124/statistics.json)

B1-50k 相对 A50 对 ZW0 的 ΔU 为 +0.443，探索性区间 [+0.044,+0.814]，是值得保留的积极信号；对 generic/v6 则分别为 −0.188/−0.074，区间跨零。它说明对照结果并不一致，不能推出整体优胜。

**以后以预注册的 ΔU 为主指标，同时报告胜–平–负、分差、淘汰率、两席结果。** 不在看完结果后改用更好看的那一个。跨三个对手的等权平均 U 可以作为下一轮研究主指标，但对手权重须在评测前固定，它也不等同于线上排行榜表现。

### 2.2 结论强度与算力口径

“区间包含零”表示当前证据不能区分差异，不等于证明相等。要声称不退化，须预先给定可接受退化界限 ε，并证明 ΔU 下界高于 −ε；要声称实用增益，还要事前定义需要多大的效果。**本轮可以说 B1 值得继续，不能说已经证明稳定打平。**

原文“相同预算”应读作“相同 actor E”：B1-10k/50k 另用了 65,869 E 的 critic 拟合数据和 19,136 E 的 critic 验证数据，总 E 分别 95,005/135,005。拟合、验证、actor 训练、策略评估的 E 与 wall-clock 分开记录；同一 critic 在多个实验中复用时，同时报告一次性成本和每臂增量成本。[训练协议][b1protocol]

A 与 B1 还同时改变了公开特征、现金基线和 value 架构，所以目前支持的是“B1 整套配置的部分结果更好”，不能单独归因于 MC 预训练。

## 3. 两个优先实现修复

### 3.1 现金权重 w 没有在 PPO 中冻结

代码链明确如下：

- `policy.py` 的 `value_baseline` 是默认可训练的 `nn.Linear`。
- `terminal_critic.py` 仅优化 `actor.value.parameters()`，所以拟合残差时 w 固定。
- `terminal_ppo.py` 使用 `AdamW(learner.parameters(), lr=3e-5)`，现金权重也进入优化器。
- `ppo.update` 的 warmup 明确允许 `value_baseline.*` 训练，结束后又把所有参数设为可训练。[模型][policy] · [critic 拟合][criticcode] · [训练器][tppo] · [PPO][ppo]

这说明“严格固定 w”没有得到实现；本次没有量出已存 10k/50k checkpoint 中 w 的实际漂移，不能据此断言它造成了多少性能变化。

**Codex 修复要求：**用 buffer 或明确的 optimizer 参数排除机制固定 w；warmup 保存/恢复原 trainable mask，不要无条件全部解冻。初始化、warmup 后及每个 checkpoint 保存 w 的 hash、最大绝对差和 optimizer 参数成员审计。新臂命名 `B1-fixed`，保留旧 B1 的原记录与 checkpoint SHA，不能覆盖历史结果。

另外，`V=V_cash+residual` 并不保证留出 MSE 永远优于现金基线。把“不可低于的现金基线”改成“初始化参照及可回退基线”；残差为零时可恢复基线，训练后的任意残差仍可能有害。

### 3.2 B2 的零梯度诊断没有处理好

`terminal_ppo.gradient_audit` 在 `len(updates)==2` 时必执行，当前默认采集对应 E≈3072。B2 detach 后，value loss 对全部 trunk 参数都没有连接；`terminal_rloo.grad_breakdown` 过滤所有 `None`，随后会执行 `torch.cat([])`。这是静态可确定的调用路径问题，不是已完成的 B2 棋力实验。[诊断实现][rloo]

**Codex 修复要求：**按同一参数坐标为 `None` 填零，保持不同 loss 梯度向量等长；范数为零时 cosine 记 `null` 并说明原因。B2 smoke 要跨过首次 audit，并验证 value→trunk 梯度为零。普通有连接的诊断路径也应保留。此修复不要求立即大规模训练 B2。

## 4. 新的主要假设：critic 可能只在起点可靠

B1 的 MC 标签是：

\[
G_t=U_T-\Phi(h_t),\qquad
\Phi(h_t)=\tfrac12\tanh((C_{own}-C_{opp})/1000).
\]

该标签和当前 gamma=1 的 potential shaping 一致，pub 来自合法公开现金和轮次，没有在此次审查中发现 pub 偷用了隐藏状态。进入 PPO 后，value 的监督却是 rollout64、λ=.95 的 `adv+v`，不再是完整终局 MC 标签。[critic 拟合][criticcode] · [环境奖励][env] · [PPO][ppo]

actor 开始训练后，状态访问分布、未来 continuation policy、共享 trunk 表示都会改变。初始化的 ZW0-vs-ZW0 校准不能自动覆盖这些变化。

还需要修正一个解释：**B2 只挡住 value→trunk，挡不住 policy→trunk 改变 critic 的输入表示。** 它不是完全独立或固定的 critic。10k 抽查中 policy/value 原始 trunk 梯度范数约 35.7/0.14，也不足以排除 critic 问题：错误的 V 可经 GAE 进入 policy loss；AdamW 的预条件、动量和全模型 clip 又使原始梯度范数不等于实际参数更新影响。[10k更新日志][b1updates10] · [AdamW 官方说明](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

### 4.1 先做这组小诊断

1. 冻结 ZW0、B1-10k、B1-50k，各在同一组 **8 张诊断地图 × 两席**上，对冻结 ZW0 打完整对局。ZW0 的动作使用原冻结actor，估值检查使用已预训练B1 critic，而不是原来的零价值头。保留每局独立随机流、合法历史、V、pub、奖励和最终 U；这批作为诊断数据，不能拿来宣布棋力晋级。可复用同协议的完整旧轨迹，但必须能恢复合法观察历史和预测。
2. 对每个 checkpoint **自己的 continuation**，计算 `U−Phi_t` 与其 V 的分阶段 MSE、bias、EV，并和现金/常数对照。现金及常数比较器沿用原拟合数据的系数，不在这8张诊断图上重新拟合。别只拿旧 ZW0 轨迹标签评判新策略的真实价值函数。旧固定局面可以另作“预测漂移”诊断。
3. 在完整冻结策略轨迹上重算三种优势：当前64步分块 GAE、同 λ 的全轨迹 GAE、`G_t−V_t` 的 MC 优势。按游戏/地图报告均值、尺度、阶段差异和多批梯度方向。**单条 MC return 不是某个动作的无噪声真值，单批梯度夹角也不是棋力结论。** GAE 的偏差与方差权衡见[原论文](https://arxiv.org/abs/1506.02438)。
4. 从同一个 checkpoint 复制**参数、optimizer 状态和 minibatch 顺序**，比较带/不带 value→trunk 的一次真实混合 loss 更新；测 actor 参数增量与固定局面的动作 KL。这比新建优化器做一次 value-only 更新更能判断是否值得优先跑 B2。
5. 检查已校准 critic 经两次 TD warmup 后是否失准。warmup 后的旧轨迹复验只作漂移检查；真正精度仍用当前策略完整续局。不要直接删除 warmup 并同时改其他超参数。

初始批次只能发现大问题；诊断噪声大时明确保留不确定性，不根据单个阶段、单个 seed 的符号自动改路线。

### 4.2 按诊断决定下一步

| 观测 | 更应优先做什么 | 当前不能推出什么 |
| --- | --- | --- |
| PPO 后 MC 校准明显丢失，GAE 与完整回报方向持续分歧 | 完整终局状态基线对照；必要时单独 MC refresh critic | 不能靠继续压低 TD training loss 证明 V 好 |
| 去掉 value→trunk 明显改变真实更新且改善诊断 | 匹配设置的 B1-fixed/B2 对照 | 小原始 value 梯度不等于无影响 |
| V 保持校准，但策略只对部分对手受益 | 用共享地图交叉评估对手差异，再考虑固定小对手混合 | 当前不同地图上的正负号不证明风格过拟合 |
| B1-fixed 在两种子上出现一致效用增益 | 保持设置延长到100k，单独测训练时长 | 不能同时减半 anchor、提高 lr 再把效果归因于时长 |

anchor 的训练状态 KL 从10k约0.090升到50k约0.249，本身没有显示“策略被束缚得动不了”。它也不是固定参考局面分布上的完整 joint KL。先增加固定局面的阶段性漂移记录，再决定是否值得单独试0.05；本轮不优先这样做。[更新日志][b1updates]

## 5. 一个有明确用途的备选：完整终局回报＋状态价值基线

如果上面的诊断指向 bootstrap 问题，复用已有完整对局 collector，增加一个 `MC-state-baseline` 对照。它是已有 policy-gradient 思路的项目内应用，不是宣称新算法。

\[
A_t^{MC}=U_T-\Phi(h_t)-b_{frozen}(h_t).
\]

这里 b 是预先校准的 shaped-value critic，输入只含合法历史。和 C 的 RLOO 不同：C 给一局中的各步同一个基于同根其他对局的回报基线；此对照使用随状态变化的基线，但不使用 TD bootstrap。

在行为策略点、未做 clipping 的期望 policy gradient 中，当前动作无关的状态基线满足：

\[
\mathbb E_{a\sim\pi(\cdot|h)}[\nabla\log\pi(a|h)b(h)]
=b(h)\nabla\sum_a\pi(a|h)=0.
\]

所以 critic 在这里主要影响估计方差，而不会像有限 λ 的 bootstrap 那样通过错误的后继价值引入同一种优势偏差。**不保证它能降低方差或胜过 PPO。** clipping、多次 minibatch 更新与样本内归一化仍会改变有限样本更新，不应把整个训练器描述为严格无偏。

具体对照协议：

- 从相同 ZW0 actor 起点开始；冻结 ZW0 对手，T=.4、相同 sequential decoder、终局 U、anchor=.1、lr=3e−5 均不变。
- 第一版 b 使用单独冻结的预训练 critic，**包括其 trunk**；只 detach 当前 actor trunk 不是固定 b。b 来自之前数据，不能先用本条轨迹未来回报拟合 b 再给同条轨迹算优势；需要更新 b 时用前一批数据或按游戏交叉拟合。
- 行为策略固定到完整采集批结束，沿用 C 的完整组采集语义及提前淘汰处理。每局相同权重，按固定299步尺度归一，不按各局实际长度归一；保留全体败局和提前淘汰局。
- 保留队伍联合 logprob、实际采样前缀和条件 mask。复用每轮只遍历一次的 clipped team-ratio 实现时，明确它是 surrogate，不是全轨迹重要性比率；记录 optimizer steps 和与 PPO 的更新次数差异。
- 第一版复用现有C的采集和优化链路，只为每步增存Phi及冻结b预测，把`prepare`中的每局RLOO常数优势改为逐状态`U−Phi−b`。沿用`optimize`的policy、entropy、anchor共同`rows_scale`，不新增样本内优势标准化。与B1比较时同时改变了完整对局采集和更新频次等，不能单独证明bootstrap就是根因。
- 不同时加入新搜索、league、新网络、温度变更或 KL 扫描。10k仅做smoke，50k看方向；只有方向可信才到100k。

现金基线和已校准残差基线可先在**同一批冻结轨迹**上比较优势方差与梯度稳定性，不必立刻各跑一套长训练。若需要问清“MC残差知识是否有用”，再增加同架构、同固定现金 w、仅残差不做MC预训练的零残差初始化对照；其后仍按相同PPO训练。持续冻结为纯现金会同时取消价值学习，是另一个实验。研究对照应显式标记用途，不伪造 `qualified=true` 来绕过原验收逻辑。

## 6. 下一轮评估设计：把地图和对手分开

### 6.1 当前的“12图”到底是什么

`terminal_eval.paired_map_panel` 把 rival 放入 layout seed，所以当前是**每对手12图，总36个布局**。同对手内的候选配对有效；跨对手比较仍混有有限地图抽样差异。[评估实现][teval]

下一轮用一个不含 rival 的 layout key 生成统一地图清单；所有对手完全交叉这份地图及两席。learner/opponent RNG 仍按地图、对手、座位、复本分开，在候选间配对复用。按同图比较如：

\[
I_m=(\Delta U)_{m,generic}-(\Delta U)_{m,ZW0}.
\]

地图是重采样簇，同一地图的两席、随机复本及对手结果要一起保留。两个训练种子分别报告，另给同图种子均值；不能把增加一个训练种子当成增加一批独立地图。确认面板的区间仍条件于这两个固定训练种子，两种子只提供初步稳定性检查，不构成覆盖所有训练随机性的算法级置信保证。

### 6.2 固定预算与结论规则

- 先提交16张共同开发图及40张共同确认图的清单/生成规则、hash和去重审计，排除已知训练、critic和旧评估图。确认图留给冻结候选，不参与调整 anchor/lr 或选 checkpoint。
- 主候选预定50k，保留10k作为轨迹诊断；比较冻结 ZW0。仅在需要区分机制时增加一个对照臂，不让所有历史 A/C checkpoint 都消耗新的确认集。
- 第一训练种子24000000，第二24010000。主线 `B1-fixed` 从同一个校准 critic 重启；先首种子50k，再相同配置复验第二种子50k。不要从旧50k接着跑并混称重启对照。
- 主指标为事前固定的三个对手等权 ΔU，各对手 ΔU 和两席单列。可用 ε=0.05 作为“非劣”声明的严格研究容忍度，但这是提议的决策标准，不是平台规定；实施前写入 protocol，之后不得按结果调整。未达到该统计标准不自动否定研究价值。
- 声称增益需要确认面板的主要效用比较支持正效应，并呈现两个种子的方向与各对手退步；声称对每个对手不退化，要对各对手分别满足非劣条件。若要同时作多个正式声明，须在协议里处理多重比较，不能直接把探索性区间当联合保证。
- 固定40图终点后如仍不确定，诚实归档为“不确定”，不反复加图直到首次显著。下一轮算力用于能改变决策的机制对照，而非无限扩面板。

40图并非万能：用当前 B1-50k 的地图级效用方差估算，40图95%半宽约为 ZW0 **0.220**、generic **0.231**、v6 **0.115**，仍可能辨不清0.05量级的差异。这是同方差近似规划值，不是未来误差条或功效保证。[复算数据](carbon-zero-b1-review-20261124/statistics.json)

## 7. critic 留出规模与末盘处理

本次 critic 拟合为28地图/224局，留出是**8地图/64局/19,136步**，不是64个独立布局。每图两席、每席4个随机复本。本轮按真实地图聚类复算如下：[critic 原始记录][criticdata]

| 阶段 | 现金 MSE − 残差 MSE | 95% 地图区间 |
| --- | ---: | --- |
| 全盘 | +0.014315 | [+0.008537,+0.020109] |
| 开局 | +0.044397 | [+0.012942,+0.073148] |
| 中盘 | +0.008891 | [+0.001466,+0.017133] |
| 末盘 | +0.001006 | [−0.005133,+0.007351] |

方向与原报告基本一致：中前盘有积极信号，末盘尚不确定。原报告未附其 critic bootstrap 实现，本次区间与其表格不完全相同，不能仅据此断言原实现有错；今后同时提交重采样脚本、地图簇定义和点估计口径。8个独立地图也不足以覆盖广泛的布局泛化。

**末盘先保留残差，不在230步硬切现金基线。** 区间跨零不是残差有害的证据；硬切还可能人为制造价值跳变，影响跨边界 TD/GAE。若扩大数据后稳定发现末盘残差有害，再单变量比较连续收缩或门控，并同时检查最终策略效用。点估计比现金好0.001也不应成为一条过度机械的“校准成功”判据：新校准协议应强调独立地图、误差幅度和不确定性。

## 8. 对5个待评审问题的直接答复

| 问题 | 决定 |
| --- | --- |
| 12图是否足以支持B1不退化？ | 不足以证明非劣；足以把B1列为优先候选。25–40图及第二种子有帮助，仍须看区间和预定容忍度。 |
| 打平算进展吗？优先100k放宽anchor还是接搜索？ | 这是有价值的研发进展，棋力增益尚未成立。先按上述P0/P1补证据、受控50k复验；有增益再原设置100k。当前两项激进改动都不优先。 |
| 50k混合结果是地图还是对手风格？ | 当前分不开；先用同一地图交叉对手，而且按U重新判读。跨图、跨训练种子复现的interaction才支持风格解释。 |
| 末盘是否硬退化为现金基线？ | 暂不。先保留，若确认有害再单独做连续收缩/门控。 |
| 能否裁掉critic导出？ | 可以。动作路径不依赖value/pub，现有导出核心已经只序列化actor；修兼容loader和parity入口即可，不能宣称尚未执行的部署验收已通过。 |

### 导出与搜索的具体边界

`export.build` 已只提取 stem、blocks、style、worker/base heads。当前真正的断点是 `terminal_export.py`/`checkpoints.py` 仍按非残差 Actor 加载，以及 parity 调用了需要 pub 的完整 forward。使用 `actor_from_checkpoint`，让动作路径显式走 `spatial → heads → decoder`，不要给动作推理伪造 pub。[导出封装][texport] · [parity][checkpoints] · [导出核心][export]

必要验收：正确checkpoint SHA；无pub的actor-only前向；合法完整历史上的logits、条件mask、同随机流动作一致；双方座位、开中末盘、招募/工人消失、终局和重置；实际导出文件在新进程完整对局；≤10MB及目标运行环境单步≤1s。**本次没有新的B1导出文件或上述验收结果。**

critic 若用于搜索，还需证明候选行动后的叶子价值排序有效。状态级MSE降低并不提供这个保证；其当前价值又绑定训练对手与continuation policy。若拿它预测终局utility，应还原 `V_U=V_shaped+Phi`；如果搜索已经累加shaping reward并用shaped value bootstrap，则不要重复加Phi。utility量纲也不能直接混入旧cash score。先做留出候选排序检查，再考虑完整搜索对局。

## 9. 给Codex的执行清单

按以下顺序工作，不需要重新询问是否保存研究建议；实现与结果继续留在实验分支。

| 顺序 | 工作与输出 | 进入下一步的条件 |
| --- | --- | --- |
| P0 | 补U报告；修固定w/trainable mask；修B2空梯度诊断；修actor-only导出加载/校验。保存必要smoke、参数hash和旧结果说明。 | 实现语义可复现，B2 smoke跨过首次audit；不要求因此有棋力改善。 |
| P1 | 用已有checkpoint完成第4节的完整终局校准、GAE/MC诊断、实际optimizer配对检查；提交新的共同地图协议。 | 确定是否存在明显校准丢失或梯度干扰；不确定就明确记录。 |
| P2 | 同critic、同0.1 anchor、同lr重启B1-fixed，首种子50k，再第二种子50k；保留0/2k/10k/50k预测漂移与当前策略MC诊断。 | 以U报告方向和不确定性；有希望的冻结候选进入40图确认。 |
| P3（条件分支） | 若bootstrap诊断持续不良，运行第5节MC-state-baseline；若真实更新显示value→trunk干扰明显，先做B2；若B1已有可重复增益，先原配置延长到100k。 | 每次只改变能解释的一个机制，不同时做三条长跑。 |

如果要定位残差预训练的独立贡献，优先使用第5节末尾的同架构零残差初始化对照；不要把A-vs-B1当作已完成这个消融。对手混合、末盘门控、学习率/anchor放松和critic接搜索均不列入本轮默认并行矩阵。

每轮产出：协议与代码commit、actor/critic/hash、按图按席逐局U/WDL/分差、地图聚类区间、训练和评估成本、负例、明确的“支持/否定/不确定”结论。研发候选与线上v8的晋级仍按仓库AGENTS执行，排行榜上传由维护者决定。

## 10. 本轮复算的复现方法与来源

[recompute.py](carbon-zero-b1-review-20261124/recompute.py) 只读实验快照，不训练模型，并核对10个输入文件与固定Git提交逐字节一致；[statistics.json](carbon-zero-b1-review-20261124/statistics.json) 包含输入SHA256、固定随机种子、抽样单位、版本和所有配对结果。需要Git与NumPy。先取得实验快照，例如：

```sh
git fetch origin experiment/carbon-zero-b1-residual
git worktree add --detach ../carbon-b1-evidence afcfe81712676dcc0d680e47881db6ecc71ea55b
python analysis/carbon-zero-b1-review-20261124/recompute.py ../carbon-b1-evidence /tmp/carbon-b1-statistics.json
```

策略评估表的区间与原报告4000次bootstrap会有Monte Carlo数值差异，不要据微小数值差异构造新结论。critic区间另按恢复出的地图簇重算；其原报告使用2000次且实现未提供，不能把差异全部归因于抽样次数。附带的按游戏重采样区间仅用于审计口径，正式解释采用地图簇。

[b1]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/analysis/carbon-zero-terminal/b1/README.md
[previous]: https://github.com/Zhaoyi-Yan/carbon/blob/868a0abe5d840f84614cfa437452e9b0f921b468/analysis/carbon-zero-terminal/README.md
[evaldata]: https://github.com/Zhaoyi-Yan/carbon/tree/afcfe81712676dcc0d680e47881db6ecc71ea55b/analysis/carbon-zero-terminal/b1/evaluation
[b1protocol]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/analysis/carbon-zero-terminal/b1/b1-50k/protocol.json
[b1updates]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/analysis/carbon-zero-terminal/b1/b1-50k/updates.json
[b1updates10]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/analysis/carbon-zero-terminal/b1/b1-10k/updates.json
[criticdata]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/analysis/carbon-zero-terminal/b1/critic/critic.json
[policy]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/experiments/carbon_zero/policy.py
[ppo]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/experiments/carbon_zero/ppo.py
[tppo]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/experiments/carbon_zero/terminal_ppo.py
[criticcode]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/experiments/carbon_zero/terminal_critic.py
[env]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/experiments/carbon_zero/env.py
[rloo]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/experiments/carbon_zero/terminal_rloo.py
[teval]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/experiments/carbon_zero/terminal_eval.py
[texport]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/experiments/carbon_zero/terminal_export.py
[checkpoints]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/experiments/carbon_zero/checkpoints.py
[export]: https://github.com/Zhaoyi-Yan/carbon/blob/afcfe81712676dcc0d680e47881db6ecc71ea55b/experiments/carbon_zero/export.py
