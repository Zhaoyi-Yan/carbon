# Carbon-Zero B1-fixed 新结果复审与 M/G 对照

> 后续状态（2026-11-25）：M/G 首种子结果已提交并完成复审，见 [M/G 复审与 M-wide 下一轮协议](carbon-zero-mg-review-20261125.md)。本文保留 B1-fixed 证据及原 M/G 预定规则；当前执行入口以新文档为准。

2026-11-24。核对快照：**`d6008ed2e8bc42df056ab08ac48711d19c4249c8`**，分支 `experiment/carbon-zero-b1-residual`。前置：[B1 复审与执行协议](carbon-zero-b1-review-20261124.md)。

**决定：B1-fixed 暂停扩大训练，也不为它追加40图确认。下一轮先修正优势诊断，再做两个各约50k E的受控实验：M为完整MC状态基线，G为同一完整冻结critic的整局GAE。两者共用采集和优化设置，只改变优势估计器。**

本轮新增的是代码审查、已有数据复算、一个实际执行的NumPy形状复现及数学反例；没有新增Carbon训练或对局。固定w/B2 smoke、两种子训练和导出记录来自Codex提交。后续M/G尚未执行。

## 1. 哪些新结果已经成立

### 1.1 实现修复有证据，但没有带来稳定棋力增益

- 两个B1-fixed训练种子的全部50次更新都记录 `baseline_max_abs_diff=0.0`、`baseline_in_optimizer=False`，固定现金系数的要求已实现。
- B2已处理缺失梯度，3072 E smoke跨过首次审计，value→trunk梯度为0、cosine为null。这个阻断已解除，不必重复把它列为待修。
- 新的actor-only导出入口已支持残差checkpoint。已有B1-10k导出为5,539,172 B，校验了64个动作；这是开头32步×两席的检查，还不是完整部署验收。[实验报告][report] · [修复及运行记录][runs] · [导出记录][exportresult]

对冻结ZW0的12图换边面板，本次用固定RNG 20261124、100,000次地图bootstrap重算：

| 候选相对ZW0 | 配对ΔU | 探索性95%区间 |
| --- | ---: | --- |
| A50 | −0.0196 | [−0.3555,+0.2819] |
| C50 | −0.0663 | [−0.3134,+0.2022] |
| B1-fixed seed 24000000 | −0.0195 | [−0.3417,+0.2942] |
| B1-fixed seed 24010000 | **−0.3708** | **[−0.6465,−0.1071]** |
| 两种子逐地图等权平均 | −0.1952 | [−0.4205,+0.0405] |

区间未作多重比较校正，条件于这些固定checkpoint。两种子均值仍只有12个地图簇，不把种子当新增地图。seed 2相对seed 1的ΔU为−0.3513，区间[−0.7194,+0.0091]；不能因为一个种子对ZW0显著、另一个不显著，就自动断言两种子之间的差异也显著。[原始面板][shared] · [本次复算](carbon-zero-b1fixed-review-20261124/statistics.json)

可支持的结论是：**当前配置缺少继续扩大投入的积极信号，第二种子出现了明确的探索性退步。** 这足以决定暂停，不需要先把“不值得继续这组配置”升级为“已经证明整个算法无效”。

### 1.2 两处报告口径要纠正

**共享面板目前只评了ZW0。** `protocol.json`及五个候选文件的对手字段均只有ZW0。支持共享地图的代码已写好，但generic/v6的跨对手实验还没有执行。新旧面板同时换了地图和随机流，结果改变只能说明面板敏感性，不能单独归因于地图，更不能宣称已经验证对手风格差异。[面板协议][sharedprotocol]

**训练局数漏记了平局。** 两个种子实际各完成162局：seed 1为**78–1–83**，seed 2为**63–2–97**；原表的78–0–83与63–0–97遗漏了双灭平局。平均分差+10.177/−186.672则已包含全部162局。应恢复完整WDL，保留提前淘汰样本。[本次复算](carbon-zero-b1fixed-review-20261124/statistics.json)

## 2. 新发现：A/ZW0的优势诊断存在广播错误

`terminal_calibration.predict`两条路径返回形状不一致：

```python
# residual critic -> (T,)
actor.predict_value(features, pub)
# non-residual critic -> (T, 1)
actor.value(features)
```

随后 `mc = utility - phi - value` 用 `(T,)` 减 `(T,1)`，广播为 `(T,T)`，不再是逐时间步优势。`gae`仍为`(T,1)`，相关函数因为展平长度不相等而返回`None`。所以**“A/ZW0的GAE–MC相关无定义，是因为V接近常数”这个解释错误**。[诊断源码][calibrationcode]

影响范围必须分清：

- ZW0/A10/A50的GAE–MC诊断需要重算，尤其相关性、绝对差及阶段统计；不同长度对局还会在拼接时出错。
- value自身的MSE/bias/EV在`metrics`中先展平，因此不受这个具体错误影响。
- B1及B1-fixed使用一维residual路径，不受这个广播错误影响。
- 这个问题在新诊断脚本里，**不是B1-fixed训练退化的原因，也不使已保存的对局U/WDL失效**。

### 已执行的最小复现

本次从固定源码提取实际`predict/analyse/metrics/corr`，仅用NumPy模拟张量转换和零critic，没有加载神经网络或运行游戏。同一四步轨迹：

| 输出形式 | MC形状 | GAE–MC相关 | 平均绝对差 |
| --- | --- | ---: | ---: |
| 当前`(T,1)` | `(4,4)` | null | 0.107919 |
| 修为`(T,)` | `(4,)` | 0.905565 | 0.051232 |

value MSE两者均为0.556547；四步/五步混合轨迹在原分析路径报错，修正后可分析全部9步。[已执行脚本](carbon-zero-b1fixed-review-20261124/shape_probe.py) · [复现结果](carbon-zero-b1fixed-review-20261124/shape_probe.json)

**Codex最小修复：**统一调用一维`predict_value`或在出口明确`reshape(-1)`，并断言`value.shape == phi.shape == reward.shape == (T,)`。相关函数应把长度不等视为错误，不能静默返回null；真正的零方差才标记相关未定义。补充不同对局长度、提前终局和下节λ=1恒等式检查。修复后的真实A/ZW0数值目前未知，不能用本次toy数字替代。

## 3. 即使没有代码错误，GAE与MC差得多也不证明GAE错

这里应修正我们接续实验时的判断标准。两种优势估计器包含不同的后续随机性，不能把MC当作逐样本无噪声真值，或把相关性小于0.5当作故障阈值。

一个可精确计算的反例：299步、Phi=0，只有终局出现独立公平的±1奖励，奖励与所有状态、动作独立。此时**完美critic就是V=0**，但：

\[
A_t^{MC}=U,\qquad A_t^{GAE}=0.95^{298-t}U.
\]

将一条+1和一条−1的完整轨迹各步等权池化，相关性为**0.361157**，平均绝对差为**0.933110**。critic完全正确，真实期望策略梯度为零；MC携带了更多无法预测的终局随机性。数值由仓库的实际`advantages`递推函数计算，λ=1时逐元素等于MC。[反例输出](carbon-zero-b1fixed-review-20261124/shape_probe.json)

**这是合成反例，不是Carbon成绩。** 它否定的是“差距大所以GAE有问题”的推断，不是否定MC路线。下一轮应比较两种估计器最终能否产生更好的策略更新。

当前校准脚本算的还是**整局GAE**，并非训练中64步及checkpoint边界短窗口的实际GAE；二者也不能混称。我们此前提出的分块/整局对照、真实optimizer配对更新等尚未完成，不把本轮P1称为全部完成。[诊断源码][calibrationcode] · [训练GAE][ppocode]

## 4. critic的绝对EV变化也不能直接当作修复效果

报告用旧B1-50k的EV≈0.042和fixed seed 1的≈0.182说明critic恢复，但现金比较器在两组轨迹上的EV也同时变化：

| 模型及其自己的轨迹 | value全盘EV | cash全盘EV | value−cash EV | value/cash MSE |
| --- | ---: | ---: | ---: | --- |
| 旧B1-50k | 0.04186 | 0.02740 | +0.01446 | 0.30029 / 0.30790 |
| fixed seed 1-50k | 0.18190 | 0.17018 | +0.01172 | 0.26774 / 0.28121 |
| fixed seed 2-50k | 0.05778 | 0.03074 | +0.02704 | 0.33136 / 0.37126 |

这是不同actor、不同状态分布和不同continuation产生的标签，不能用绝对EV上升证明冻结w造成了预测改善。fixed seed 1的中盘value EV甚至低于现金（0.16089对0.16942），但MSE仍较低，因为偏差也变小。全盘相对现金MSE收益约0.01347，其中约0.00974来自bias²减少。[校准记录][calibrations] · [复算分解](carbon-zero-b1fixed-review-20261124/statistics.json)

本轮依然有有用的发现：这些residual critic在各自采样轨迹上MSE比现金低，A的value本身预测能力较弱。但现有聚合JSON没有每图预测误差，无法给这些差异补地图置信区间，也无法证明网络在训练过程中“遗忘了多少信息”。

最低限度补齐：

1. 零步基准使用**ZW0 actor＋已预训练B1 critic**。当前脚本的ZW0分支用了原始零价值头，未测到我们要的起点。
2. 保存每步`map/seat/turn/V/Phi/U/terminal`及明确的actor、完整critic、代码SHA。当前`cash_baseline_sha256`写的是文件名`critic.pt`，应改为实际内容hash。大轨迹可留本机，入库保存压缩预测数组或按图误差和hash，以便离线重算。
3. 若在同一批轨迹上比较初始与50k critic，明确标签对应哪一个continuation policy；跨策略旧模型的误差是迁移/预测漂移检查，不能自动当作其自身价值函数的校准误差。

先完成这些必要项即可，不把所有诊断补齐变成开启下一实验的无限前置条件。

## 5. 下一轮主线：M/G两臂，只改优势估计器

### 5.1 为什么增加G对照

直接做MC-state-baseline再和历史B1比较，会同时改变采集长度、critic更新、优势标准化和优化次数。为了让下一次结果更有解释力，新增一个共享设置的G臂。历史A/C/B1保留为参考，本轮不重新长跑全部历史臂。

共同设置：同ZW0 actor起点、冻结ZW0对手、同一个**完整冻结的预训练B1 critic（含trunk）**、T=.4、原sequential decoder、相同终局U/Phi、原C完整对局collector、原C优化器及loss尺度。value不参与反向传播或在线拟合，不增加warmup。critic文件使用已有`fceedd2e7944394f2119377ae30997bc86adb8db00272f0dcd14e3548540ec47`并校验完整来源。[原critic与协议][fixedprotocol] · [完整对局训练器][rloocode]

该`critic.pt`只包含value/value_baseline，不包含trunk。须用已冻结ft3源码SHA `920f656a31277751651883099d3bc2891a68412a69533cce4af3e3c801ba9446`、style 122重建独立trunk，再加载上述head/w；冻结整个模块，记录组合来源与参数hash，禁止与正在更新的actor共享参数。

| 臂 | 每步优势 |
| --- | --- |
| **M：MC状态基线** | `U_T − Phi_t − b_t` |
| **G：整局GAE** | `delta_t + .95 delta_(t+1) + ...`，其中`delta_t = r'_t + b_(t+1) − b_t`，真实terminal后`b=0` |

两臂都用同一个b。gamma=1，非终局`r'=Phi_(t+1)−Phi_t`，终局`r'=U_T−Phi_t`。对任意一维完整轨迹，λ=1的G必须与M逐元素一致；把这个恒等式作为实现检查，而不是用低相关性判断训练有效性。

### 5.2 实施细节与预算

- 复用C每批两地图、两席、每组四局的采集约定。一次完整批内冻结行为策略；保留所有败局和提前终局，先完成整组再过预算标记，报告实际E而非把它强写成50,000。
- 第一批两臂参数相同，可以共用同一批轨迹验证优势构造。**策略分开更新之后各自重新on-policy采样**；相同地图/随机流不意味着可以把M轨迹直接当作G的on-policy数据。
- 保留`C.optimize`的AdamW lr=3e−5、anchor=.1、entropy=.01、clip epsilon=.2、每批单次遍历、梯度clip=.5。policy/entropy/anchor共同使用原`rows_scale`和固定299步尺度。不新增样本内优势标准化，不按实际局长重加权。[优化实现][rloocode]
- 第一训练seed 24000000，两臂各做10k smoke和预定约50k终点；保存每批优势分布、policy/anchor梯度、实际KL、优化步数、时间和完整组信息。两臂优势尺度本来会不同，这属于估计器整体效果，不能把最终差异全归因于“bootstrap偏差”。
- 探索性预算筛选预定为：若任一候选相对ZW0的三对手等权ΔU点估计>0，且没有单对手95%区间上界<0，则用24010000匹配复验两臂。这不是统计晋级标准；首轮区间宽也只能称筛选信号，不依靠单局或训练胜率选臂。
- 若两臂在开发面板上均无积极效用信号，本轮到50k停止，不自动扩大100k、不继续调小anchor。若某臂有跨种子一致改善，再按旧方案进入独立确认图。

这里比较的是**优势估计器的偏差、方差与尺度的整体权衡**。冻结b会随策略变化而变旧；M如果更好，只支持它在这套冻结基线和预算下更有效，不能推导一般GAE/PPO无效，也不能证明历史B1退化必由GAE造成。

### 5.3 评估只给新候选投入

沿用当前12张共同开发地图，预先固定generic/v6的随机流，评估ZW0、M、G三个候选对ZW0/generic/v6的完整交叉。保留当前ZW0-vs-ZW0已有结果，不重跑历史A/B/C全表。主指标预定三个对手等权ΔU，逐对手、逐席及WDL/分差/淘汰率同时报告；各对手结果在同地图簇内一起重采样。

注意现有`terminal_eval`不允许覆盖不同protocol：当前目录只有ZW0。如果直接以默认三个对手复用同一输出目录，会触发protocol不一致。Codex应新建三对手评估目录，冻结完整manifest；仅在版本、地图、席位、温度和随机流完全相同时导入已有ZW0条目，另两对手补跑。先把结果记录完整，再作任何跨对手结论。[评估实现][evalcode]

当前12图已是开发数据。进入确认前，要求同一臂两个训练种子的三对手宏平均ΔU方向都为正，固定使用约50k checkpoint并分别呈现两种子，不挑最佳seed；两臂都满足时，预定选择两种子等权宏平均较高者进入独立确认。完整规则在首轮运行前写入protocol；它只决定下一笔评估预算，不是已经证明增益。没有必要给缺少积极信号的B1-fixed补40图。

## 6. 给Codex的下一次交付

| 顺序 | 具体任务 | 需要保留的证据 |
| --- | --- | --- |
| 1 | 修校准输出形状和metadata，修README中的漏平局、对手覆盖及过强因果解释。 | 一维/同长度断言，λ=1恒等式、混合长度检查；保留旧诊断并标记受影响字段。 |
| 2 | 用已有合法轨迹重算受影响诊断；没有可重放轨迹时再采集必要对局。补ZW0＋预训练critic零步基准。 | 每图可重算的V/Phi/U与真实SHA；不必重跑已完成B1训练。 |
| 3 | 实现并运行上述M/G首seed各约50k；不同时修改网络、奖励、对手、decoder或KL。 | 初始参数一致、完整critic冻结、实际采样前缀/logprob、分组/预算/优化记录。 |
| 4 | 新三对手共同开发面板；有希望才两臂第二seed复验。 | ΔU主指标和地图区间；负例、种子分别结果；无增益则停止本轮扩量。 |

原有B1-fixed记录保持不变，不更新发布策略或上传排行榜。`--no-freeze-value-baseline`目前也不严格复现旧B1：新warmup固定现金基线，旧warmup会训练它。需要历史复现时使用旧commit或显式恢复该差异，别把此开关当作已验证的完全等价复现。

## 7. 本次复算附件

- [recompute.py](carbon-zero-b1fixed-review-20261124/recompute.py)：验证16个输入与固定Git提交逐字节一致，重算逐局、按地图的ΔU和校准误差分解。
- [statistics.json](carbon-zero-b1fixed-review-20261124/statistics.json)：输入SHA、固定RNG、100,000次重采样、种子/地图口径与限制。
- [shape_probe.py](carbon-zero-b1fixed-review-20261124/shape_probe.py)：提取固定提交中的实际分析函数，复现广播错误和完美critic反例；只用NumPy，不依赖模型权重或PyTorch。
- [shape_probe.json](carbon-zero-b1fixed-review-20261124/shape_probe.json)：实际执行结果，明确区分合成例子与Carbon数据。

```sh
git fetch origin experiment/carbon-zero-b1-residual
git worktree add --detach ../carbon-b1fixed-evidence d6008ed2e8bc42df056ab08ac48711d19c4249c8
python analysis/carbon-zero-b1fixed-review-20261124/recompute.py ../carbon-b1fixed-evidence /tmp/b1fixed-statistics.json
python analysis/carbon-zero-b1fixed-review-20261124/shape_probe.py ../carbon-b1fixed-evidence /tmp/b1fixed-shape-probe.json
```

[report]: https://github.com/Zhaoyi-Yan/carbon/blob/d6008ed2e8bc42df056ab08ac48711d19c4249c8/analysis/carbon-zero-terminal/b1/README.md
[runs]: https://github.com/Zhaoyi-Yan/carbon/tree/d6008ed2e8bc42df056ab08ac48711d19c4249c8/analysis/carbon-zero-terminal/b1/b1-fixed
[exportresult]: https://github.com/Zhaoyi-Yan/carbon/blob/d6008ed2e8bc42df056ab08ac48711d19c4249c8/analysis/carbon-zero-terminal/b1/export-b1-10k.json
[shared]: https://github.com/Zhaoyi-Yan/carbon/tree/d6008ed2e8bc42df056ab08ac48711d19c4249c8/analysis/carbon-zero-terminal/b1/evaluation-shared
[sharedprotocol]: https://github.com/Zhaoyi-Yan/carbon/blob/d6008ed2e8bc42df056ab08ac48711d19c4249c8/analysis/carbon-zero-terminal/b1/evaluation-shared/protocol.json
[calibrationcode]: https://github.com/Zhaoyi-Yan/carbon/blob/d6008ed2e8bc42df056ab08ac48711d19c4249c8/experiments/carbon_zero/terminal_calibration.py
[ppocode]: https://github.com/Zhaoyi-Yan/carbon/blob/d6008ed2e8bc42df056ab08ac48711d19c4249c8/experiments/carbon_zero/ppo.py
[calibrations]: https://github.com/Zhaoyi-Yan/carbon/blob/d6008ed2e8bc42df056ab08ac48711d19c4249c8/analysis/carbon-zero-terminal/b1/b1-fixed/calibration-s1.json
[fixedprotocol]: https://github.com/Zhaoyi-Yan/carbon/blob/d6008ed2e8bc42df056ab08ac48711d19c4249c8/analysis/carbon-zero-terminal/b1/b1-fixed/s1-50k/protocol.json
[rloocode]: https://github.com/Zhaoyi-Yan/carbon/blob/d6008ed2e8bc42df056ab08ac48711d19c4249c8/experiments/carbon_zero/terminal_rloo.py
[evalcode]: https://github.com/Zhaoyi-Yan/carbon/blob/d6008ed2e8bc42df056ab08ac48711d19c4249c8/experiments/carbon_zero/terminal_eval.py
