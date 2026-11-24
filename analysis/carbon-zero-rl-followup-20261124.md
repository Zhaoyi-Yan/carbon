# Carbon-Zero：首轮 PPO 退化复盘与完整终局 RL 实验

日期：2026-11-24  
状态：研究交接 / 待实施。本文新增的是诊断、假说与实验协议，未新增训练或对局成绩。  
证据冻结点：[experiment/carbon-zero @ c73c283](https://github.com/Zhaoyi-Yan/carbon/commit/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72)。  
前置设计：[Carbon-Zero 研究方案 v2](carbon-zero-research-plan-20261124.md)。首轮已经完成；接下来以本文件为实验优先级，不重复 200k 主跑。

## 0. 下一轮最重要的事情

**保持 Carbon-Zero 路线，但先停止无约束 PPO 的规模放大。优先实现完整对局 RLOO（Carbon-GRPO 的无 critic 基线），并与充分预训练 critic 的 PPO 做小规模受控对照。**

当前模型连固定训练对手都越来越打不过，尚不能把问题归结为 League 泛化、搜索迁移、网络容量或缺少新规则。现有训练速度足够支持快速实验，不应退回手工策略补丁。

核心问题是：从已经很强的 BC actor 出发，能否通过可靠的终局反馈改善同一策略？先排查梯度与信用分配，再决定是否扩大训练分布。以下原因均为待检验假说，不把相关性认定为因果。

## 1. 首轮已经知道什么

所有记录来自固定提交 c73c283 下的 analysis/carbon-zero 与 experiments/carbon_zero。[1–5]

| 检查点 | 对冻结 ZW0 W–L | 平均终局分差 | 混合效用 U | 相对初始策略的平均 team KL |
| --- | ---: | ---: | ---: | ---: |
| 普通 PPO 10k E | 1–7 | −445.49 | −0.497 | 0.246 |
| 普通 PPO 100k E | 3–5 | −701 | −0.296 | 2.512 |
| 普通 PPO 200k E | 0–8 | −2362 | −0.954 | 4.576 |
| 锚定 PPO 10k E（ref KL 权重 0.1） | 5–3 | +42.69 | +0.140 | 0.100 |

每个检查点均为同一组四张地图换边，仅属于开发检查。锚定组对两个外部纯策略的配对终局分差仍退步：generic −121、无搜索 v6 −750，不能宣布有效。无锚定 200k 外部 pure 配对变化分别为 −2836、−1653；旧搜索对 v8 的配对变化为 −1322（仅一张图换边）。搜索评估只用于接口与退化诊断，不是晋级。

从 episodes.json 重算：主跑累计 668 场，93 胜、5 平、570 负；前 100 场 35 胜、1 平、64 负，末 68 场 0 胜、68 负，平均分差 −1959.68。训练对手始终冻结为 ZW0。训练中每场可能跨多轮策略更新，所以该曲线不是固定 checkpoint 的独立竞技场成绩；但同时有冻结 checkpoint 的退化证据。

值得保留的工程结果：

- ft3 Apollo 精确源码 SHA-256 为 920f656a31277751651883099d3bc2891a68412a69533cce4af3e3c801ba9446，风格行 122；54 通道、96 宽、6 residual blocks。温度 0.4 对温度 1.0 打出 8–0，固定它作为当前对照。
- 同权重下，independent decoder 对原生 W-raw 0–6（−1672），sequential decoder 1–5（−680）；另一两图复查为 1–3（−222）。有适配损失，但 ZW0 与 ZW1 都使用相同 sequential decoder，不能用适配损失解释 RL 后续退化。
- 初始网络数值、部分合法观测隔离、old logprob 重算、恢复及纯策略导出动作已通过现有测试。快引擎逐步比对通过 19,397 随机步与 7,176 历史步，但独立视野/状态接口与手工边界仍待核验；本轮训练仍用参考裁判。
- 主跑约 25.9 分钟、129 E/s，后续实际速度需重新测量。首轮本地旧搜索导出约 11.18 MB，高于文档记录的 10 MB 平台限制，不能当作可上传成品。

## 2. 优先假说与需要新增的诊断

### H1：冷启动 critic 和 bootstrap 导致错误的长期优势

policy.py 中 Actor 新增 value head，最后一层为零。pilot.py 仅前两个 update 冻结 actor（2,048 E），之后开始更新共享 trunk 与动作头。updates.json 前五个窗口：

| 累计 E | 已完成局数 | Actor 更新 |
| ---: | ---: | --- |
| 1024 | 1 | 否 |
| 2048 | 1 | 否 |
| 3072 | 1 | 是 |
| 4096 | 1 | 是 |
| 5120 | 17 | 是 |

唯一早期结束的局是 seed 24000009、第 4 回合双方淘汰平局，裁判分均为 −297。即第一次 actor update 时连一场正常终局的胜负监督都没有。此时 value 主要由现金势函数增量和初始 bootstrap 学习，可能使优化方向失真。当前 critic 的拟合 loss 不是对真实终局预测的独立验证。

GAE 当前 gamma=1、lambda=.95，直接 TD residual 的远期权重为 lambda^l；100 步约 0.00592，200 步约 0.0000351。成熟 critic 可通过 bootstrap 传递长期价值，这不是 gamma 变小或数学目标变短；**冷 critic + 有限窗口下**，潜在问题是优势估计偏差。势函数塑形在完整 return 上不改变目标，但不保证有限样本、近似 value 下的梯度同样可靠。

### H2：共享表示被 value/actor 梯度共同破坏

value head 从共享 CNN 特征计算，正式 PPO 后 value loss 与 policy loss 均更新 trunk。旧日志的 gradient_norm（clip 前总梯度）在非 warmup 更新约 10.7–19.1，clip 设 0.5，不能据此认定是 critic 单独造成，但应增加分项日志：policy、value、entropy、reference KL 的梯度范数、共享 trunk 的 cosine、clip 前后梯度与实际参数/输出漂移。先测同一小批数据，无需每步都做额外 backward。

做一次 value-only 梯度对照：相同真实终局标签、只更新 value head 时 actor logits 必须保持；若允许 value 更新 trunk，记录 actor 输出变化。这是机制诊断，不能作为改进棋力的独立证据。

### H3：PPO 的局部 KL 控制不等价于累计 reference 约束

每次 minibatch 相对旧行为策略的近似 KL 约 0.005–0.007，但累计参考漂移达 4.58；工人最可能动作翻转率约 5%、16%、21%。10k 增加 reference KL 后漂移减小，但外部成绩仍退步。

注意 diagnose.py 的 112 个根来自两种布局的八场历史对局、每根一个冻结策略前缀；这只是条件动作分布的漂移估计，不是整个状态分布上的精确 KL。后续需要更广的独立地图和多个 reference prefix，并区分 learner-prefix 和 ref-prefix 的 KL。

不优先动 reward、网络宽度、温度、decoder 或对手池。先控制这些变量，确认是否存在真实学习增益。

## 3. 固定条件与工程入口

从 c73c283 的代码继续开发，不从 main 的旧策略目录重新实现。main 仅承载研究文档，实验代码在 experiment/carbon-zero。已有工作树/分支则先检查增量，勿覆盖：

~~~sh
git fetch origin main experiment/carbon-zero
# 从当前实验工作树查看新文档；若已有更新，先核对增量。
git show origin/main:analysis/carbon-zero-rl-followup-20261124.md
~~~

如需独立 worktree，可从冻结 c73c283 创建 experiment/carbon-zero-terminal-rl（建议名），不要改变发布文件。

固定 ft3、style=122、T=.4、sequential decoder、参考裁判、训练对手 ZW0。维持原有目标：正常局 U=0.5 sign(D_T)+0.5 tanh(D_T/1500)，提前正常淘汰按裁判赢家 ±1，平局 0。soft score 为平台近似目标，不能宣称精确复现平台函数。

沿用 Game、Encoder、Actor、slots、logits、coordinated_sample、commands、evaluate、export。所有新训练臂从冻结 ZW0 出发，而不是从已经退化的 200k checkpoint 继续。原始 ft3 及权重在外部本地目录，必须核验 provenance.json；不要伪造或猜测缺失权重。新增代码、协议、日志和可恢复权重 manifest 继续提交实验分支。完整 v8 仍为发布基线，线上提交由维护者决定。

## 4. Carbon-RLOO：不依赖 critic 的完整终局策略梯度

### 4.1 采样单位与奖励

对一个任务组 g=(layout, seat, frozen opponent snapshot)，固定合法初始状态，使用同一冻结行为策略版本生成 G 条完整对局。建议 G=4（必要时 8）；组内 learner 随机流相互独立，对手也用独立且可记录的随机流。两席分别成组，按地图均衡；每局对手均在实际新状态上重新决策，不重放历史未来指令。不同组可批量推理。两方每回合先各自生成完整命令，再同时推进真实裁判。

使用已有终局混合 U；RLOO 不需要中间 shaping，因为完整轨迹从同一个起始状态出发时 shaped return 与 U 仅差同一个常数。在现有 gamma=1 下，其完整 return 的组内差分与原 U 一样。不要用 64 回合切片截断轨迹再当成终局。

RLOO 优势：

$$
A_{g,i}=U_{g,i}-\frac{1}{G-1}\sum_{j\ne i}U_{g,j}.
$$

同组其它独立轨迹的 return 是当前轨迹的 baseline，在固定组条件下不依赖当前轨迹采样动作。等于 G/(G−1) 乘以组内去均值后的 U。第一轮不除以组内 std、不按奖励或难度挑组、不丢掉败局或同分组；相同奖励组的 A=0，是正常无信号组，应计入预算、统计。部分可观测历史只来自本方，实际地图种子不输入 actor。

这是 RLOO/REINFORCE 风格的基线，不等同于某一论文的完整 GRPO 实现。第一轮用完整对局共享一个优势，信用分配较粗、方差可能高；其作用是隔离 critic，并非承诺最终最强。

### 4.2 正确的联合动作概率与损失

对第 i 局第 t 个队伍回合，按现有 sequential decoder 记录实际有效 mask、UID/slot、已选前缀、最终指令、温度、old joint logprob。固定序贯顺序与 prefix；重算新 logprob 时使用原先采样的相同前缀：

$$
\ell_{it}=\sum_{j\in\mathrm{active}}\log\pi_\theta(a_{itj}\mid h_{it},a_{it,<j}).
$$

最简单次 on-policy 更新为：

$$
L_{\rm PG}=-\frac{1}{NGT_{\max}}\sum_{g=1}^{N}\sum_{i=1}^{G}
A_{g,i}\sum_{t<T_{g,i}}\ell_{git}.
$$

T_max=299 仅为固定尺度，不按每局实际长度单独归一化，不把提前淘汰的短局相对长局额外放大。对 minibatch 实现，可用每回合加权求和后按整个采样批次固定尺度归一。不要把工人当独立 RL 样本。单次梯度在行为参数处等价于轨迹 REINFORCE；重复利用数据则需要明确近似目标。

为与 PPO 工程复用，允许采用按队伍回合的 clipped ratio：

$$
\rho_{it}=\exp(\ell_{it}-\ell_{it}^{old}),\quad
L_{\rm RLOOclip}=-\frac{1}{NGT_{\max}}\sum_{g,i,t}
\min(\rho_{it}A_{g,i},\operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)A_{g,i}).
$$

这里每回合 ratio 使用联合 logprob 之和，不是平均，不跨全 299 步做一个巨大乘积。多 epoch clipping 是近似 surrogate，不声称精确修正轨迹分布偏移。优先 1 epoch、clip=.2、lr=3e−5，控制每批 old-policy KL；必要时再增加 epoch。不要沿用 PPO 的 value loss 或 GAE；RLOO 也不需要随机 value head 更新共享 trunk。第一轮保持与 PPO 对照相同的 per-active-slot entropy 定义和系数，完整记录，必要时独立消融。若全组优势为零，纯 reward gradient 应为零，entropy/anchor 的单独作用仍须记录。

### 4.3 可执行采样预算

建议每次冻结行为模型收集 4 个组 × 4 条完整局（两席平衡），约 4,784 E/批，允许环境提前结束后空闲或启动同协议新完整组。所有被收集到的有效完整组都纳入；达到目标预算时完成最后整组，报告实际 E，不能丢弃“恰好慢/输”的轨迹。不得同一组在 actor 更新后才生成剩余轨迹。

10k E 仅约两个这种批次，适合工程 smoke，不够证明 RLOO 无效。50k、100k E 约十、二十批，作为第一轮学习曲线；报告有效组数、独立布局数、终局效用方差、零优势组比例、正负优势分布、每批更新数。对比 PPO 时 E/L 为主预算，同时列实际优化次数与墙钟，不把同 E 误说成同梯度步数。

### 4.4 单测与验证

- 人工小组已知 U，验证 leave-one-out 计算、组内优势和为零、组内相同 U 的 reward gradient 为零、加入常数不改变优势。
- 用可枚举两行动 toy game 验证期望策略梯度朝正确方向；固定旧模型时 ratio=1。
- 检查完整终局与提前结束、实际发送指令、decoder mask、独立 RNG、策略版本、无隐藏信息输入。
- 单局/多局并行、checkpoint 恢复和导出与训练一致；RLOO 不应出现 value 梯度。
- 在未查看地图的 ZW0 与多风格外部对手上比较实际棋力。不要因训练时组均值优势恒为零就说训练没有效果，也不要把训练 surrogate loss 当胜率。

## 5. PPO 另一条路线：先学会估值，再更新策略

### 5.1 B1：完整轨迹 critic 预训练

冻结 actor 全部参数；用 ZW0 对同一冻结对手、两席均衡生成完整对局。起步建议 100–200 场（约 30k–60k E），另保留未用于拟合的地图和轨迹。按实际可用预算定，不把这部分免费忽略。每个合法历史 h_t 的监督标签为：

$$
y_t=U_T,\quad
V_\mathrm{terminal}(h_t)\approx E_{\pi_0,\rho_0}[U_T\mid h_t].
$$

也可显式构造用于原 shaping 的 value：

$$
V_\mathrm{shaped}(h_t)=V_\mathrm{terminal}(h_t)-\Phi(s_t).
$$

因为 gamma=1 且 Phi(terminal)=0，完整未来塑形回报恰为 U_T−Phi(s_t)。应先证明这个实现数值正确。原现金势函数必须用当时公开现金，不从未来推断。可以直接拟合 U−Phi 以维持原 value 接口；显式分解作为单变量对照，不同时改 reward 与目标时域。

第一版只训练新 value head，冻结特征，使用完整 Monte Carlo 标签而非随机 critic 的 64-step bootstrap。训练/验证按地图，报告开局 0–60、中盘、残局的 held-out MSE、均值偏差、explained variance、终局效用相关性/校准，比较常数预测、当前公开现金的简单估计和原冷 critic。不能要求开局完美预测结果；按阶段、领先/落后分层，避免末局样本掩盖中盘失配。若不优于简单基线，先扩大有标签数据或改 critic 容量，不急着更新 actor。

预训练后从完全相同的 actor 出发，保持原 PPO 64-step collector、lambda=.95、lr、decoder 和 anchor，验证是否改善。B1 的首要变化是**critic 初始化**；共享 trunk 是否受 value 梯度影响、actor 解冻范围、GAE lambda 是不同变量。

### 5.2 B2：隔离 value 梯度（后续单变量）

若 B1 仍退化或 value-only 诊断显示明显表征漂移，比较：
- 原共享 trunk 正常 policy/value 更新；
- 对 value 输入做 stop-gradient，或独立冻结特征/独立 critic；actor 仍通过 policy loss 学习。

前者和后者保持相同初始化、标签、采样和预算。独立 critic 增加计算需报告；仅冻结 value head 不等于隔离它对 trunk 的影响。之后才比较 lambda=.99、完整 Monte Carlo/长 rollout、仅更新动作头或最后 block。不得一口气改完并将收益归因于 critic 预训练。

## 6. Reference KL：先复现固定系数，再做自适应

主对照使用现有 reference KL 系数 0.1，与 anchored-10000 一致：

$$
L=L_\mathrm{RL}+\eta\,E_h\sum_j D_{KL}(\pi_\mathrm{ZW0}(\cdot\mid h,a_{<j})\Vert
\pi_\theta(\cdot\mid h,a_{<j})).
$$

按同一有效 mask 计算，排除 padding。训练 on-policy 历史/learner prefix 上的条件 KL 与独立面板上 reference prefix 的 KL 不是同一统计量，分别记录。冻结完整参考参数、不随 learner 更新，PPO old-policy KL 仍单列。RLOO loss 按固定队伍回合尺度归一，anchor 与 entropy 的尺度要对应；相同 eta 是受控起始值，不保证不同梯度的有效约束相等。

待固定系数对照完成后，可以试自适应 eta：以独立开发观测面板上的平均 team KL 0.1 为探索性中心，低于 0.05 缓慢减小、高于 0.15 增大，明确更新频率和上下界（例乘/除 1.5，范围 0.01–1），并记录每次调整。此阈值是初始研究假设，不是理论最优或硬保证。也报告 p90/p99、各阶段、base/worker 与贪心动作变化率。若小 KL 仍严重退化，先查优势方向，而非无限提高惩罚；若有稳定收益并撞到预算，才试单变量放宽。开发面板不能使用 locked final test。

## 7. 下一轮实验矩阵与验收

| ID | 起点与策略 | 优先要回答的问题 |
| --- | --- | --- |
| ZW0 | 冻结 ft3 Apollo、T=.4、sequential | 不训练的对照 |
| A | 原 PPO + ref KL=.1 | 延伸现有 10k 锚定对照 |
| B1 | 完整局 MC 预训练 critic + 同 PPO/KL | 是否冷 critic 导致退化 |
| C | 完整局 RLOO + 同 ref KL，无 critic | 去掉不成熟 critic 后能否学习 |
| D | C，但仅更新 worker/base 动作头 | 是否共享表示变动是重要因素 |

优先 C 与 B1；A 使用已经归档的 10k 对照，若可恢复其权重与训练状态则从该文件延续（核验 hash），否则从同种子重跑。D 在 C 可正常训练后执行。第一轮仍固定 beta=.5,K=1500、T=.4、decoder、opponent ZW0，不做动态 League、全量 BC、搜索重训、奖励网格。

预算：每臂 10k smoke（B1 的 critic 预训练另单列），50k、100k actor 训练检查点，若有可解释信号用第二 seed 与新图验证。B1 同时报总 E=critic 数据 E+actor E；既比较相同 actor E，也比较相同总 E 时的曲线，不能把额外 critic 数据成本隐藏起来。若需要缩减算力，优先 A/C 同预算、B1 价值预训练与一次可行更新，保留所有实际结果。不要因为 10k RLOO 仅两次更新的噪声立即否定它，也不要因为一个 5–3 的面板就自动扩大。

评估：
1. 固定一个新的开发地图清单，ZW0 对自身 A/A（两席与独立评估随机流）以估计基线波动，比较同图、同座位、相同对手版本的配对效用。原四图可作为已查看回归图，不能再叫独立测试。
2. 先对 ZW0 及 generic、无搜索 v6 等冻结外部策略进行纯网络评估。使用固定独立 per-game learner/opponent RNG，记录随机流；外部对手在真实状态闭环行动。报告胜负、混合 U、现金差、提前淘汰、每席、每图、耗时，地图聚类 bootstrap；有趋势后扩大到至少 8–16 张新图换边并第二训练 seed。
3. 对纯策略已有可信正趋势的 checkpoint，再做冻结旧搜索 root-only 迁移（v2 §11），避免每个明显退化 checkpoint 都花大量墙钟跑 v8。旧搜索 value、opponent、rollout 全冻结，固定搜索形状诊断与真实限时结果分开；不把超平台大小的文件当发布候选。

决策：若 C 稳定胜 A/B1 且外部改善，优先沿无 critic 终局训练扩大，后续可研究更细粒度的 baseline；若 B1 明显优于 A，支持 critic 初始化机制，继续做独立 value 梯度/GAE 对照；若 D 胜 C，优先低风险局部参数更新，再渐进解冻。若都无提升，先看优势信噪比、梯度方向、探索温度与训练对手难度，考虑完整轨迹的高方差、表示不足和换任务分布，不直接返回 v8 规则补丁或无条件放大。

## 8. Codex 第一批任务

- [ ] 从 c73c283 或已存在的后续实验工作树开始，核验 ft3 与冻结 ZW0，保存新协议。先阅读已完成的增量，避免重复实验。
- [ ] 实现 full-episode group collector、RLOO loss、reference KL 及单测；复用现有合法观测、decoder 和参考裁判。输出组级统计和 E/L。
- [ ] 从冻结 ZW0 建立 C 的 10k smoke，并准备固定新开发面板和 A/A；先跑 50k/100k 可行学习曲线。并行采集 B1 critic 完整终局训练/独立验证数据；先做 critic 校准报告再开启 actor update。
- [ ] 记录分项梯度与累计 reference KL，继续 A 对照；有资源再做 D 和第二训练 seed。保持所有控制变量、原始记录及失败结果。
- [ ] 将训练代码、协议、日志、评估、权重 hash/可获取位置与 README 提交推送实验分支。结论明确区分已观测现象、假说和因果验证。按 AGENTS.md 保留当前发布策略，不自动上传平台。

这次的目标不是造一个新的规则版本，而是找到能够持续提升强初始化策略的 RL 信号。**先证明学会，再讨论学得更快或更强。**

## 证据入口（固定版本）

[1] [首轮报告](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/analysis/carbon-zero/README.md)；[训练对局](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/analysis/carbon-zero/episodes.json)；[优化日志](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/analysis/carbon-zero/updates.json)。  
[2] [PPO 与 GAE](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/experiments/carbon_zero/ppo.py)；[Actor/value/decoder](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/experiments/carbon_zero/policy.py)；[训练入口](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/experiments/carbon_zero/pilot.py)。  
[3] [原 reference 漂移](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/analysis/carbon-zero/actor-drift.json)；[锚定 10k](https://github.com/Zhaoyi-Yan/carbon/tree/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/analysis/carbon-zero/anchored-10000)。  
[4] [来源和版本](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/analysis/carbon-zero/provenance.json)；[评估汇总](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/analysis/carbon-zero/summary.json)。  
[5] [实现复现说明](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/experiments/carbon_zero/README.md)；[reward/环境](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/experiments/carbon_zero/env.py)；[现有评估器](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/experiments/carbon_zero/evaluate.py)。

