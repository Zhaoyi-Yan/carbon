# Carbon-Zero：从终局目标出发的独立策略学习方案

日期：2026-11-24  
状态：设计与实验交接（尚未实现、尚无 Zero 棋力数据）。  
目的：不再以修复 yan-v8 某个失败案例为主要研发范式，建立一个可持续产生更强策略的学习闭环。本文为新的独立研究主线，不要求完成旧评分审计后才开始。保留旧路线和所有负结果作为对照。

## 0. 核心决策：重建学习机制，而非重写更多规则

此前多轮实验中，局部经济指标改善并未稳定转化为终局收益。持续树任务相对原 v8 的终局差为 −211.76；回本实验中，成功种下的 11 棵树均回本，但对应分支终局均值 −206.30（这是事后成功子组，不是全体策略效应）；最新评分审计的独立回复检查补到一个真实碰撞，却没有改变选招。[1][2]

此前 v7 的 own-search 是拟合历史搜索动作，selfplay-value 是拟合历史中期现金标签；它们不是通过当前策略的终局奖励反复更新整套策略的 RL。已有自对战竞技场也不等于 RL 训练器。[3]

这不是证明 v8 已经到理论上限，或 PPO 必然更强。它说明继续投入大量时间于局部规则，难以建立可靠的能力增长机制。新路线的目标是：

- 训练时直接用真实游戏的最终胜负反馈更新**全队策略**，允许它发现我们没有预先想到的经济与对抗策略。
- 推理时先只用网络，不要一开始就接入 v8 的候选生成、资产评分、tail、对手路由或战术后处理。
- 充分利用已有回放作为初始化，而不是把模仿准确率当成目标；通过在线环境学习超越初始化。
- 先证明低成本闭环有效，再考虑联合动作的更强表示、联赛、搜索；不一次实现 AlphaZero/AlphaStar 全套系统。

**保留的资产**：原版裁判、回放解析和合法观测构建、可信快速模拟器中经核验的规则部分、竞技场、历史版本、前六/前十数据、旧实验的诊断方法。  
**不继承的策略机制**：人工决定种植员比例、树收益加权、临时回库/护树补丁、旧 style 路由、短续局价值融合、搜索后的强行改招。若以后确实需要安全机制或搜索，必须作为独立可消融候选进入。  
**首要假说**：相同数据、相同模型和相同评估条件下，BC 初始化 + on-policy RL 能比仅 BC 的策略更强，并在固定多风格对手池上持续提升。先检验这一条，不承诺具体天梯名次。

## 1. 固定基线、代码组织与首轮范围

设计时 main 为 [f2d1f6a](https://github.com/Zhaoyi-Yan/carbon/commit/f2d1f6a5cff5e08dd2f24ebcfa6e73ad0618019d)，已发布 v8 源码提交为 155dcc5275b8a23a223769346cc7c00661e36155。最新版评分审计位于 6ade8e72b2efbc2b345f76a1bd45000aede20ce3。旧分支保留，不合并其策略代码进 Zero。

冻结 v8：标签 v11-yan-v8-elite-refit，submission.py SHA-256：

    a87f8e6b83656c771cb5a25d6b6f383b245127e7c782e4c6c7c4a0a2cc15d396

参照 main 的 AGENTS.md、ARENA.md 及 scripts/selfplay.py；主力、标签和比赛平台上传均保持不变。[4]

建议在独立 worktree 开发。若分支/目录已存在，先读现有增量再复用，不覆盖：

~~~sh
git fetch origin main --tags
git worktree add -b experiment/carbon-zero ../carbon-zero origin/main
cd ../carbon-zero
~~~

建议新增路径（均为**待实现**，不是现有命令）：

    experiments/carbon_zero/
      env.py             # 参考环境适配器
      fast_env.py        # 可选：仅在 parity 通过后替换参考环境
      obs.py             # 统一合法观测与历史编码
      policy.py          # 独立 policy/value
      replay_data.py     # 复用归档解析并输出 BC 样本
      bc.py
      rollout.py
      ppo.py
      league.py         # 第二阶段；首轮先用固定 pool
      evaluate.py
      export.py
      configs/
      tests/
    analysis/carbon-zero/
      protocol.json
      README.md
      ...                # 指标、manifest、版本、评测记录

第一轮只做环境契约、吞吐、独立 BC、固定对手训练及短 PPO 验证。不要先实现动态联赛、复杂任务规划、全量搜索、分布式异步训练，也不要将全部研发工作绑在新架构上。

## 2. 数学问题：一个队伍是一个 agent

Carbon 是部分可观测的双人同时行动有限时域博弈。以裁判状态 s、各自合法观测历史 h^i、联合动作 a^i 表示：

$$
s_{t+1}=F(s_t,a_t^1,a_t^2), \qquad
a_t^i \sim \pi_\theta^i(\cdot\mid h_t^i).
$$

同一回合中，双方先各自决定完整动作，再统一结算；不能让任意一方看到另一方尚未执行的动作。自身控制的所有 worker 和基地可共享信息，因此第一版把整队视作一个策略 agent，而不是十个独立 RL agent。游戏中的队内冲突、空载压制、树与运输的机会成本，都应成为策略学习的对象。

标准配置为 15×15 环面、每方两个基地、最多十名工人、树寿命 50；episodeSteps=300 对应最多 299 次正常动作推进。当前 vendored helpers 的实际招募只检查工人总量与现金，角色上限检查被注释掉，不能直接根据 JSON 中的 planterLimit/collectorLimit 进行错误 mask。平台配置若不同，按实际裁判和配置处理。[5]

裁判状态包含隐藏的 full_carbon。actor 只能读取玩家可见的 carbon、公开双方资产、合法历史；未观察格的零值不能等价于已知零碳。视野的构造以裁判实现为准。禁止用地图 seed 反推资源图，禁止借另一席观测或未来回放帧填补自己的隐藏信息。

## 3. 奖励：以比赛胜负为本，而不是重新堆手工经济分

### 3.1 先核对真实结算

正常满局以裁判最终分数判断胜负；正常情况下等于现金。提前淘汰时，裁判对被淘汰方给负分，比赛可能提前停止；不能假设所有轨迹都固定 299 步，更不能只用剩余现金比较淘汰局。直接复现原裁判的 reward/status/termination 以及 arena 的 winner 口径，并核对官方回放。软件错误、非法指令、超时等单独统计，不能训练或筛选掉坏局来虚增胜率。

令 S_T^i 为真实结算分数，终局效用为：

$$
U_1=\operatorname{sign}(S_T^1-S_T^2), \qquad U_2=-U_1.
$$

胜/平/负为 +1/0/−1。优先最大化对给定对手混合的期望 U，而不是平均现金；后者仅是补充指标。真实博弈的目标接近 max-min，但有限对手池上的 PPO 不保证全局 minimax 或 Nash。

### 3.2 改进上一版现金差密集奖励

简单使用每步现金差增量，gamma=1 时确实累计为终局现金差减初值，但**现金差与胜率是不同目标**；若再取 gamma<1，短期变现甚至会被额外奖励。这很可能把我们旧路线中的短视重新带入 RL。

建议第一轮用终局胜负奖励 + potential-based shaping。定义非终局势函数：

$$
\Phi_1(s_t)=\alpha\tanh((C_t^1-C_t^2)/K),\quad \Phi_2=-\Phi_1,
$$

终局（包括淘汰）强制 Phi(s_T)=0。原始 r_t^0 在非终局为 0，最后一步为 U。训练奖励：

$$
r_t=r_t^0+\gamma\Phi(s_{t+1})-\Phi(s_t).
$$

则折现累计的塑形项严格 telescope：

$$
\sum_{t=0}^{T-1}\gamma^t r_t
=\sum_{t=0}^{T-1}\gamma^t r_t^0-\Phi(s_0).
$$

在 gamma=1、初始状态相同的情况下，该回报与终局胜负仅相差动作无关常数。势函数只提供更密集的信用分配线索，不给中期现金永久的额外价值。注意最后一步的势函数抵消项不能遗漏；训练切片不是环境 terminal，不得把切片末端 Phi 人为清零。若以后比较 gamma<1，需要明确它对不同终局时间产生额外偏好，不能声称完全保留原始不折现胜率目标。[6]

首轮建议 gamma=1，GAE lambda=0.95（长时域若信号衰减严重，单独比较 0.98/0.99）；K=1000、alpha=0.5 为起始超参，非实测最佳值。并做相同条件的纯终局 U 对照（alpha=0）。先不加树数、存货量、死亡数、采矿量或中间胜率等独立奖励。势函数依赖公共现金，双边 reward 反对称。对实际终局现金差，可作为 value 的辅助预测目标；如果使用 margin 混合效用作为早期课程，明确那是不同训练目标，最终回到胜负效用并单独消融。

必须写 reward 单测：满局、提前淘汰、平局、随机轨迹、分片 bootstrap、两席反对称；验证塑形总回报等式。

## 4. 环境：先证明能大规模正确生成经验

### 4.1 E0 参考环境

先封装现有 benchmark.py 的 configure_engine/new_game/observe 和 vendored carbon.compute_next_board，不另写结算逻辑。训练 API 建议 reset(seed, config) -> obs_both；step(action_both) -> obs_both, reward_both, terminated, info；两个 actor 从同一旧状态各自获得合法视野。内部全量状态与 actor 输入明确分离。

现有 arena 每席独立进程、计入通信与墙钟，用于可靠实战评估，但不是高吞吐 RL collector。训练可以在一个 worker 中运行多个独立环境，或批处理多个棋局。每个 episode 自己持有状态、单位 ID 映射、观测历史和随机数。先确保不会跨局共享 v8 的模块全局状态、复用 belief 或混淆棋手随机数。

需要核对：隐藏资源、初始状态、环面坐标、ID 生成/顺序、招聘顺序、现金消耗、人数限制、两人同时移动、碰撞赢家/平局、碰撞货物转移、在任意基地发生的归仓、种/夺树、树龄、碳再生、结束与计分。尤其同类动作之间的判定顺序、浮点相等导致的碰撞分支、树的市场价格不能被粗略近似。原版实现可能存在特殊边界，训练环境要忠实于裁判，不自行修规则。

### 4.2 高速模拟器的验收

先用参考环境测速度，拆开 step-only / encode / batch inference / rollout end-to-end / optimizer。记录 CPU 核数、线程、设备、内存、batch、每秒双边环境推进数和每秒学习方 team-transitions 数。

如果原版太慢，优先复用旧快速模拟中与真实规则一致的部分，或用紧凑 NumPy/Numba 等实现独立 batched engine；不用旧 search 内部近似模拟直接当训练环境。先做最小 parity，包含：

- 人工构造的边界状态（至少覆盖上述事件每一种以及多个同时发生）；
- 固定初始布局上的随机合法联合动作；
- 高手真实回放及 BC/历史策略生成的高密度状态；
- 至少 10,000 个逐步对照，另留至少 100 个完整/提前结束的独立轨迹做连续一致性检查，数量为工程起始门槛，不是正确性证明。

比较每一步双方现金、货物、位置、角色、树及树龄、隐藏碳图、每席观测、status/reward、下一步动作合法集合；ID 可做一致映射，不允许把不同单位混同。离散事件必须一致，浮点误差的容许值事先明确，并验证不会改变相等/大小比较、合法性、赢家。首个 mismatch 保存可复现状态与动作，修复并加 regression。parity 失败就回退参考环境。

### 4.3 预算与吞吐

统计 E = 双边环境推进次数（不是单个 worker 动作数），L = 参与训练的 team-transitions。一次完整对局约 299 E；1M E 约对应 3344 场满局，提前结束会增加实际场数。双方均参与训练时 L 约等于 2E，与冻结对手训练时通常 L 约等于 E。报告真实数据而非仅报 epoch。

以实测端到端吞吐 q（E/s）计算 rollout 时间 E/q，另加优化、评测、保存成本。Mac CPU/MPS、Linux CPU 或有 GPU 的环境均可用；先 benchmark，GPU 不一定比小 CNN 的 CPU 批量推理快。避免每一步创建进程/重复加载权重；限制 BLAS 线程，减少 Python 对象与跨进程序列化。不要在每个 rollout step 调用带完整搜索的 v8。

分阶段预算建议：10k E smoke；100k E 早期可学习性检查；0.5M、1M E 保存学习曲线；有稳定进展才申请 3M–5M E。预算是待验证的研究档位，不是承诺达到某个棋力或时间。若达到 100k E 尚无信号，先检查实现、BC 初始化、对手难度与奖励；若到 1M 仍无相对 BC 的可复现提升，暂停放大。先测算当前机器所需成本，允许因资源降低预算但必须记录并保留同预算对照。

## 5. 策略：先简单强健，再学联合协同

### 5.1 观测表示

第一版使用紧凑 CNN，不直接继承 v8 的 54 通道及手工风格特征。建议输入：

- 15×15 空间张量：可见碳量、当前可见 mask、上次观测的碳量、观测年龄；双方基地、树及年龄、collector/planter 位置及载货。
- 队伍向量：双方现金、剩余回合、人数、合法配置常量。
- 可选历史：上一回合公开单位位置/载货变化、己方历史动作。先用可复现的 last-seen memory + 短历史；GRU/ConvGRU 是后续单独消融，不要先引入复杂 belief model。

双方特征以本方/敌方统一编码，采用 circular padding 反映环面。可测试旋转、镜像、双方视角变换和整体平移的数据增强；仅当同步变换资源、基地、单位、方向标签、历史与裁判规则均正确时启用。尤其需验证方向映射和初始布局分布，不能仅转图不转动作或跨集合泄漏。

### 5.2 Zero-P0：第一版 CNN actor-critic

建议宽度 48–64，4 个 residual block，参数量在导出后实测。空间特征在每个当前己方单位/基地的位置 gather，结合职业、载货和全局池化向量得到 logits。worker 5 动作（STAY/UP/DOWN/LEFT/RIGHT）；base 3 动作（NOOP/RECPLANTER/RECCOLLECTOR）。无单位/基地的 padding token 不参与计算损失。

整队 value 输出标量，使用可部署观测。初期共享 trunk + actor/value head 便于批量推理；如 value 梯度干扰 policy，再试独立 critic。后续可用训练期独立全状态 critic（centralized training/decentralized execution），它不进入 actor、归一化统计或部署输入，不向 actor泄漏未来/隐藏信息；它只是方差降低的可选消融，不是首轮必要依赖。

### 5.3 动作的合法性与解码

基础版本采用 factorized joint policy：

$$
\pi_\theta(\mathbf a\mid h)=\prod_{j\in U_t\cup B_t}\pi_\theta(a_j\mid h,j).
$$

它并不表示网络不知道队友位置（共享全队观测），但条件独立采样会限制同一步随机行动相关性。先建立这个最简可学习基线。仅 mask 真实不存在/非法的指令；例如按原裁判的基地处理顺序维护可用现金和名额时，应在概率模型中一致体现。第一版可让多个基地同时发出语法有效但预算不足的招聘尝试，由裁判真实决定哪些成功；若改成条件合法 mask，则记录每步同样的 mask 与 base 顺序。两种实现都允许，不能训练时一种、导出时另一种。

STAY/NOOP 通过不下发对应 key 表达。不要把可能失败的种树、故意碰撞、两人同格等合法战略行为全部 mask；否则训练永远学不到这些行为的真实收益。禁止先采样再由旧 v8 修正指令、随后却用原采样 logprob 更新。最终送入裁判的是本次采样所对应的联合指令；实际招募失败、死亡等是环境结果，不是重新标注动作。

### 5.4 Zero-P1/P2：有证据再增加协同表示

P1：在相同 CNN 上加入小型 Unit Attention（2 层、64–96 dim、4 heads 为建议起点），token 包括双方单位、基地及全局信息；相对位置使用环面距离/方向，padding mask 明确。这样一个 worker 可以参考远处队友，而不只靠 CNN 感受野。先做同参数量附近、同训练预算对照。

P2：同一回合内自回归生成全队动作：

$$
\pi(\mathbf a\mid h)=\prod_j \pi(a_j\mid h,a_{<j}).
$$

先基地后 worker，基地顺序匹配裁判招聘处理；worker 使用确定且可重放的顺序或事先采样并记录的顺序，不得训练和部署不一致。可考虑随机顺序消除固定优先级，但要用同一顺序重算 logprob。decoder 接收已分配的队友落点/行动、预算、名额、局部冲突特征，以学习互相让路、掩护和协同。前缀仅包含同一队的动作，不能看敌方当步指令。监督训练用 teacher forcing，RL rollout 和部署都按模型生成的前缀；PPO 更新时在实际采样前缀上重算条件概率。

**第一版不同时引入 P1、P2。** 若 P0 已能稳定学习但出现显著队内协同瓶颈，优先验证 P1 或 P2 各一个最小变体，再组合。真正需要复杂目标分层/宏任务时，也用 RL 端到端比较，避免重新回到一堆手工优先级。

## 6. 初始化：高手示范是起点，不是最终目标

### 6.1 数据选择与提取

收集公开前六精确策略 ID 的全部可获取对局，尤其互战、对我们和其他对手、胜局和败局。前十中的历史自身版本、旧 Sponge 等作为单独补充或对手池；不按榜单名次把自己/同作者旧版本当作新强教师。记录时间、来源、对手、完整轨迹、地图 hash、版本、校验、分页覆盖和缺失。已有 elite-refit 数据及脚本在实验分支 5bda5e5...，不是当前 main 的训练目录；可从该固定提交提取/参考数据解析，不依赖整分支合并。[7]

每场每席重建当时合法观测和历史，动作标签取实际下发指令（若来源只有状态推断，必须标注不确定动作并 mask 对应监督；尤其碰撞、出生/死亡、未成功招聘和 STAY）。完整回放可用于核验、训练标签、critic 的可选全状态输入，不可填 actor 隐藏资源。按地图/对局分组划分 train/dev/locked test；同图双方、换边、旋转等相关布局均留在同一组，训练冻结后新回放单列。

### 6.2 BC 训练

采用当前部署的 generic policy，从随机初始化拟合不同老师实际动作，不要求推理时知道老师身份。按教师及对局均衡采样，整队损失按有效单位/基地动作归一，分别记录工人、基地、种夺树/回库/碰撞等关键切片；避免多数 STAY 或普通移动淹没罕见决策。关键事件切片只能使用训练可知的标签，不作为未来输入。不要把每个胜局的所有动作都当成完美示范。

$$
L_{\rm BC}=-\mathbb E[(\sum_{j\in U}\log p(a_j)+w_B\sum_{j\in B}\log p(a_j))/(|U|+w_B|B|)].
$$

w_B=1 先试，再根据角色损失单独调整，避免招聘信号被单位数淹没。也可给 value 用真实终局胜负做初始化（按实际 actor 观测），但与 RL reward/critic 尺度对齐，若做势函数塑形，value 预训练标签也需转换为 U−Phi(s_t)（gamma=1）。不在第一轮额外引入多教师 embedding/router；后续若验证 generic 学习互相冲突，可独立训练多个专属 BC 策略作为快速联赛对手，不强制线上动态选 style。

首轮 BC 仅以 dev 选 checkpoint，报告同冻结策略实际出招的合法性、分项 NLL 和**完整闭环对局胜率**。teacher-forced NLL 与纯策略棋力都要测，不能只靠拟合误差。

## 7. PPO：让网络从当前自己犯的错误中学习

### 7.1 算法与实现细节

用同步 PPO 先建立正确 baseline。一次 rollout 固定行为策略版本，固定各局对手快照。存储 h_t（或可无损重算的输入）、实际指令、有效 action mask、解码顺序、旧 joint logprob、value、reward、terminal、seat、opponent_id、policy_version、布局。重放前缀时必须精确复现 logprob；在 optimizer step 前随机抽样核对 new/old ratio ≈1。

每个队伍回合是一个 sample，不把十个单位当十条独立环境 transition。团队 reward 产生一个优势 A_t。对于实际采样的全队指令：

$$
\ell_t=\sum_j\log\pi_\theta(a_{t,j}\mid h_t,a_{t,<j}),\quad
\rho_t=\exp(\ell_t-\ell_t^{\rm old}).
$$

factorized P0 去掉条件前缀；PPO 优化 clipped surrogate：

$$
L_{\rm PPO}=-\mathbb E[\min(\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)]
+c_V L_V-c_H H.
$$

GAE 使用同一 reward/discount 和真实 terminal mask；收集窗口结束但未完赛，用同一行为策略 value bootstrap，不能按死亡/淘汰处理。value 使用回报或 GAE target；优势归一化仅跨有效 team-transitions，entropy 可按有效动作数归一，避免单位多的队伍贡献无界。actor 的联合 ratio 用 logprob 的**和**而不是均值；联合 ratio 随单位数变得更尖锐是正常的工程挑战，必须记录按人数的 clip fraction、joint KL 和 per-action KL。若梯度退化，先降低 lr/epoch、扩大 batch 或 target-KL early stop；逐单位 ratio clipping 等替代目标需独立实验，不能称为等价 joint PPO。

建议初始设置（都不是实测最优）：

| 项 | 建议 |
|---|---|
| 环境并行 | 16–64，依据 E0 吞吐 |
| rollout length | 128–256 个环境推进；可跨真实 episode，正确 bootstrap |
| minibatch | 256–512 个 team-transitions |
| PPO epochs | 2–4 |
| Adam learning rate | BC 后 1e-4；随机初始化 3e-4 作为起点 |
| clip epsilon | 0.1–0.2（先 0.2） |
| gamma / lambda | 1.0 / 0.95 |
| value coefficient | 0.5 |
| entropy coefficient | 0.01，按实际 entropy/探索诊断 |
| grad clipping | global norm 0.5 |
| target KL | 先 0.02（明确为 joint KL/turn），按实际 clip 比例复核 |

奖励尺度是有界的，先不使用会改变目标的 reward clipping。若 early elimination 过多，先排查语义、招聘/动作概率、探索和对手课程，不增加“活着就给分”之类新的永久奖励。

初期 BC+RL 可小步更新；若示范能力严重退化，再试逐步减弱的 BC loss 或 KL 到冻结 BC reference，必须对照无 anchor 的 RL。anchor 是优化正则，不是长期必须保留的规则；不能用它把策略锁死在老师分布。禁止把旧策略、冻结对手或专家的轨迹当当前 PPO on-policy 样本。它们可另做 BC，或者训练一个独立 exploiter。

### 7.2 小步先验收再放大

在投入百万步前做以下 smoke：
- 手工短 episode 的 reward、GAE、terminal bootstrap 核验；小型可计算的 toy simultaneous game 或固定简单对手可学习性测试。
- 固定 seed 的 rollout 可复现；logprob 对齐、网络更新有非零梯度、KL/entropy/value loss 有合理范围。
- 观察随机和 BC 策略的合法出招、招聘发生率、是否长期空动作、提前淘汰率。
- 至少几次 PPO update 后，checkpoint 能恢复训练、导出与原模型输出一致。
- 在真实原裁判（不是同一个快速 engine）上跑完整对局，包含未训练 seed。未过这些检查不扩训练。

## 8. 多对手训练：从第一天就避免“只会打上一代”

### 8.1 首轮先固定、分层的对手池

不用等到动态 League 才引入多样性。训练池先包含：
1. 可快速运行的合法简单策略与弱版本（保证训练早期有可学习信号）；
2. 冻结 BC 快照（generic 和若干真实教师风格克隆，前提是已有足够数据且实战验证其风格差异）；
3. 冻结 Zero 最近快照/初始 BC，对手每局固定；
4. 少量历史 v6/v7/v8 或它们明确标记的快速无搜索版本。完整 v8 推理昂贵，先只用于评估或极少量独立训练，不要把无搜索代理成绩当作完整 v8 成绩。

训练双方可来自同一个冻结行为版本，也可为当前 learner vs frozen opponent；只有符合当前 on-policy 条件的一侧/双方进入 PPO。采样先按类别均衡，保留随机对手和不同难度。评估池使用固定 hash 的不同版本、公开强策略、未加入训练的代理家族及最新真实回放诊断，不用评估对局做 PPO、选样本或调整采样。旧 v8 可以作为训练锚点或 locked eval，若训练使用了 v8，就另留未训练的强对手来测外推，不能把该项称为未见对手泛化。

### 8.2 稳定后再建立动态 League

为每个 frozen checkpoint 保存模型、编码、配置、文件 hash、胜率表与生成来源。每隔约 100k–250k E 快照一次（按吞吐调整）；只有实际能力或风格变化才保留到 active pool，旧 checkpoint 归档不丢失。保持一定数量近期、历史、不同家族及专门 exploiter，避免池不断膨胀。

建议初期 opponent mix（待测）：约 30% 最新冻结自身，30% 历史 Zero/BC，25% 多风格代理，15% 其他可快速运行的历史策略；某类为空则显式重分配。根据配对胜率 w_i 优先抽学习价值较高的对手，例如 w_i(1−w_i) 的 curriculum，以及 (1−w_i)^2 的弱点采样；两种可分比例混合，并加 uniform floor/家族上限。禁止将所有 rollout 都给当前最难且几乎必输的策略。对手采样权重只在批次边界更新，对手在一整局内冻结，避免 non-stationarity 和数据归因混乱。

出现 A 胜 B、B 胜 C、C 胜 A 时，用 payoff matrix 保留相互克制策略，而不是只按单个 Elo 淘汰。若主策略有长期弱点，另训 best-response/exploiter 针对冻结主策略，再把 exploit 加入联赛；训练 exploiter 的数据不直接冒充主 learner 的 on-policy 数据。可选 PSRO/近似 Nash mixture 是后续路线，不能在小规模结果上声称求得 Nash。AlphaStar 的 league/PFSP 是可参考机制，不是 Carbon 中已经验证的具体配比。[8]

## 9. 最小实验矩阵与里程碑

先在相同架构、数据预算和固定对手池上识别最重要的机制。第一轮不做全排列：

| ID | 初始化 | 在线训练 | 目的 |
|---|---|---|---|
| Z0 | BC | 无 | 纯网络基线，真正脱离旧搜索 |
| Z1 | BC（同一 Z0 checkpoint） | PPO，胜负 + 势函数 | 核心假说 |
| Z2 | 随机（同 P0 架构） | PPO，同 Z1 reward | 判断示范是否只是加速、还是限制后续策略 |
| Z3 | BC（同一 Z0 checkpoint） | PPO，仅胜负 | 在 Z1 管线稳定后，验证 shaping 的样本效率 |
| 可选 Z4 | BC | 继续同等计算量的 BC，无 RL | 区分额外训练计算与环境反馈的收益 |

Z1、Z2 使用相同训练 E 档位、固定对手池、采样概率、地图分布、评估协议；Z1 额外消耗 BC 成本单列。可以按总训练算力额外画一条公平曲线。首轮先交 Z0 和 Z1 的小预算，Z2 可并行或紧随其后，不要求 scratch 达到 BC 初始水平才认为 RL 有效。Z3 在主线跑通后执行，而不是先网格搜 reward 系数。后续比较动态 League vs 固定 pool，P1 vs P0、P2 vs P0，最后才比较 search vs no-search。

每阶段预先固定 checkpoint 间隔、开发评估集、选择规则和终局测试集。建议：

- M0（环境）：参考环境运行/快速引擎 parity、无信息泄漏测试、端到端吞吐和预算表通过。
- M1（BC）：训练/留出 manifest、独立纯网络导出、与若干冻结对手完整换边、记录可学习的初始能力。不能因为 BC 输给带搜索 v8 就取消 RL。
- M2（PPO smoke）：完成 10k E 附近试点，数据一致性、奖励、恢复和合法运行通过。
- M3（可学习性）：100k/500k/1M E 的 Z1 学习曲线，相对 Z0 对多个固定对手的提升；不要用 self-play 自己对自己的 50% 胜率作为进步指标。先至少两个独立训练 seed 做短预算复验，再扩大有效路线。
- M4（联赛/表示）：M3 获得明确进展后，再做一次单变量改进。
- M5（晋级）：候选对冻结 v8 的完整限时换边，随后独立留图；并对多个不同历史/公开强对手检查回归，最后由维护者决定上线、用真实前六对局检验。

## 10. 评估：目标是能打赢更广的对手，而不是挑出一次好成绩

评估至少分三层：

1. **训练诊断**：每个 checkpoint 在固定开发池上的 W/D/L、平均/中位终局分差、提前淘汰、各席表现、每对手家族结果；记录训练步数、seed、墙钟、环境和策略 hash。开发池可选 checkpoint，结果不得当最终测试。
2. **冻结验证**：预留地图、冻结对手版本，不参与训练/调参；用多张不同地图换边。第一轮可每对手 8 张图×2 席作筛查，再扩大到至少 16–32 张独立图（根据不确定性和实测成本）。以地图为聚类单位做配对 bootstrap，报告 win+0.5 draw、现金差及 95% 区间；不能把单图的多回合、多分叉当独立比赛。对多随机 seed / checkpoint 选择保留所有结果，未查看 final panel 只用于少量预先选定候选。
3. **发布确认**：遵守 AGENTS.md；在真实线上时限和 CPU 约束下与冻结 v8 正式换边。建议初筛 16 局至少 12 胜、每席至少 5 胜、平均分差正，再独立 8 局至少 5 胜、每席至少 2 胜（作为起始晋级协议，按最新维护要求事先确认；不得根据结果改门槛）。即使过线，也需要检查不同风格对手而不是只针对 v8。对于闭源真实前六，唯一真实对手棋力证据仍然来自正式比赛。

示例：导出文件实际生成后，使用现有已核对的入口；SEEDS 由预登记清单替换：

~~~sh
python scripts/selfplay.py \
  --candidate /path/to/zero-export.py --baseline yan-v8 \
  --seeds $SEEDS --jobs 1 --record-actions \
  --output arena-runs/zero-v8-confirm.jsonl
python arena_report.py arena-runs/zero-v8-confirm.jsonl
~~~

首轮研究成功不要求立刻胜过 v8，而是相对自身 BC 在多个未参与更新的对手和地图上表现出可复现的增长；若只是一个训练对手胜率提升、其他全部退化，应扩大训练分布而非晋级。若多阶段 checkpoint、至少两个 seed 都无学习趋势，停止放大并检查环境、rollout/old logprob、奖励、策略表示和对手难度。不要因为一个低样本正均值反复增加手工机制。

## 11. 搜索可以回归，但必须服务于自主学习闭环

Carbon 的联合动作空间很大、同时行动且碳量部分可见，不能直接把围棋的完整信息单动作 MCTS 搬过来。等无搜索策略达到有竞争力的水平，再尝试：
- 在同一根用训练策略采样若干整队候选，使用合法观测 belief 和共同的敌方回复分布进行短 lookahead；
- value 对齐真实终局效用，对所有候选使用一致模拟；对手不能先见我方动作，隐藏资源通过可用历史估计而非读取真实棋盘；
- 比较 no-search 与固定预算 search 的完整对局、耗时和稳定性，若有收益，把搜索改善的动作分布蒸馏回 policy，并再做 on-policy RL；
- 低置信和不稳定排序不是硬 teacher 标签，记录改进收益及不确定性。搜索是否保留由整局指标决定。

这是可选的 policy-improvement 算法，而不是第一轮必须完成的线上复杂控制器。若无搜索模型强于旧主力，还可以直接部署更轻、更容易归因的 policy。若学习确实停滞，再考虑学习型分层任务、value-based credit assignment 或 offline/online 混合；先定位具体瓶颈，不同时推倒全部实现。

## 12. Codex 第一批任务清单（请立即从这里开始）

**本次只要求 M0–M2 的最小闭环和 M3 的首个可行检查点，不要无边界扩 scope，不要自动上传榜单。**

- [ ] 拉取最新 main，本文件为起点，创建独立 experiment/carbon-zero 工作树，冻结 v8、裁判、配置和原始数据 manifest。
- [ ] 实现参考环境适配器与合法观测编码；验证 reward/status/视野/联合动作语义，加入边界单测和回放逐步核验。
- [ ] 运行 throughput benchmark，给出参考环境及批推理的 E/s、L/s、内存与 100k/1M E 预计成本；只有必要时启动 fast engine 工作。
- [ ] 实现 P0 CNN + factorized joint action，建立纯网络导出入口；用最小真实数据集验证 BC 样本、分组、训练和导出。原始回放缺失则明确缺失，先用实际可用训练图和简单对手打通，不伪造数据。
- [ ] 建立固定训练/开发/最终测试 opponent manifest 与 seed/layout manifest；先测 Z0 完整对局，再用同 checkpoint 建 Z1。
- [ ] 实现同步 PPO（胜负+势函数），通过 reward telescoping、old logprob、GAE/terminal、checkpoint 恢复测试；做 10k E smoke 和可负担的 100k E 首次学习曲线。
- [ ] 保存全部胜败、性能、训练状态、模型参数量、导出 hash，明确哪些实验实际完成、哪些尚未运行；实验代码与可复现结果提交到独立分支。大权重/回放遵守仓库大小与来源许可，必要时用可获取存储并提交校验 manifest；不提交本地账号/凭据。
- [ ] 交付简短报告：是否存在 RL 学习信号？主要时间花在哪？是否需要快模拟？是否值得扩到 0.5M–1M E？下一批只选择一个最关键变量。

建议报告不要只写胜率；至少包含训练/验证分图、对手分组、吞吐、失败原因、PPO 指标、数据规模、同预算 BC 对照和完整棋局记录。若 Zero 尚不如 v8，照实记录，不更改 main 的策略文件。每轮按 AGENTS.md 归档。最终目标是建立稳定、可扩展、直接由真实游戏结果驱动的棋力增长流程，而不是再造一个更长的规则脚本。

## 参考与仓库证据

[1] [上一轮实验交接](yan-v8-next-experiments-20261124.md)：持续任务、回本及联合动作的固定版本、实验边界。  
[2] [最新评分审计](https://github.com/Zhaoyi-Yan/carbon/blob/6ade8e72b2efbc2b345f76a1bd45000aede20ce3/analysis/v8-score-audit/README.md)：关键回复独立验证尚未改善选招。  
[3] [own-search](https://github.com/Zhaoyi-Yan/carbon/blob/6ade8e72b2efbc2b345f76a1bd45000aede20ce3/analysis/v7/own-search.md)；[selfplay-value](https://github.com/Zhaoyi-Yan/carbon/blob/6ade8e72b2efbc2b345f76a1bd45000aede20ce3/analysis/v7/selfplay-value.md)。  
[4] [ARENA.md](../ARENA.md)、[AGENTS.md](../AGENTS.md)、[scripts/selfplay.py](../scripts/selfplay.py)、[benchmark.py](../benchmark.py)。  
[5] [Carbon 配置](https://github.com/Zhaoyi-Yan/carbon/blob/f2d1f6a5cff5e08dd2f24ebcfa6e73ad0618019d/vendor/zerosum_env/envs/carbon/carbon.json)、[裁判结算与视野](https://github.com/Zhaoyi-Yan/carbon/blob/f2d1f6a5cff5e08dd2f24ebcfa6e73ad0618019d/vendor/zerosum_env/envs/carbon/carbon.py#L155)、[实际招聘、树价、移动与碰撞](https://github.com/Zhaoyi-Yan/carbon/blob/f2d1f6a5cff5e08dd2f24ebcfa6e73ad0618019d/vendor/zerosum_env/envs/carbon/helpers.py#L951)。  
[6] Ng, Harada, Russell (1999), [Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)（势函数塑形）；Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。  
[7] [elite-refit 数据与训练入口](https://github.com/Zhaoyi-Yan/carbon/tree/5bda5e5b033b26ac925c02e229e2ba1b6751ab88/experiments/elite_refit)；[覆盖和训练归档](https://github.com/Zhaoyi-Yan/carbon/tree/5bda5e5b033b26ac925c02e229e2ba1b6751ab88/analysis/elite-refit)。  
[8] Vinyals et al. (2019), [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://doi.org/10.1038/s41586-019-1724-z)（联赛训练与 PFSP）；Yu et al. (2022), [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)（PPO 的多智能体 baseline 参考，非 Carbon 结果）。

以上外部工作用于说明方法来源，不构成 Carbon-Zero 成功的实验证据。
