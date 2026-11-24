# Carbon-Zero：从终局目标出发的独立策略学习方案

日期：2026-11-24（v2，吸收 Claude 复核意见）  
状态：更新后的实验计划，尚无本方案新增训练或棋力结果。原方案见 [fe8a342](https://github.com/Zhaoyi-Yan/carbon/blob/fe8a3420f53311f35cb18b5e7fec821e829cf6fd/analysis/carbon-zero-research-plan-20261124.md)。  
目的：不再以修复 yan-v8 某个失败案例为主要研发范式，建立一个可持续产生更强策略的学习闭环。本文为新的独立研究主线，不要求完成旧评分审计后才开始。保留旧路线和所有负结果作为对照。

> **首轮执行状态与后续优先级（2026-11-24）**：200k PPO 试点已完成，出现退化；见 [首轮报告](https://github.com/Zhaoyi-Yan/carbon/blob/c73c2837e7a0ead8c29f0ffe0c17f9416db66e72/analysis/carbon-zero/README.md)。接下来以 [完整终局 RL 实验交接](carbon-zero-rl-followup-20261124.md) 为准，不重复执行本文件第 12 节的首轮主跑。以下保留原 v2 设计，便于复现和追溯。

## 0. 核心决策：重建学习机制，不必从随机权重重来

此前持续树任务、回本筛选和评分审计没有获得稳定棋力改善；单树回本也不等于全队终局收益。[1][2] v7 的 own-search、selfplay-value 是历史动作与现金标签监督，不是当前策略用真实对局奖励反复更新整个 actor 的 RL。[3] 这不证明 v8 已达到理论上限，更不证明 PPO 必然成功。

**新主线 Zero-W：先复用强网络、合法特征和已验证环境，快速检验 BC→RL；Zero-P（独立新网络）保留为后续对照。** 改变的是策略提升机制，不是无条件丢弃既有资产。第一轮固定 teacher style、校准温度，利用约 10–20 万次双边环境推进，测试纯 policy 和相同旧搜索中的提升。不要先搭完整 League、新架构或 AlphaZero 系统再测试。

### 0.1 Claude 的补充证据与边界

下列数字来自用户提供的 Claude 复核文本（2026-11-24），本次没有取得全部对应逐局原始记录或平台页面，**是待复现的外部实验报告，不是本仓库新跑出的结果**。

| Claude 报告 | 结果 | 本方案决定 |
| --- | --- | --- |
| 榜单 | v8 56 局、2946±37；v7 2955±27；第六 3025 | 不把击败上一代当成缩小前六差距 |
| 平台普通局评分 | 胜负与分差软分各占一半 | 混合效用作为主目标，核验官方细则 |
| generic / 固定专属风格 BC | 对原网络 8–8 / 16–0，后者 +1390 | 优先强固定风格，保留 generic 对照 |
| BC 温度 | T≈1 的测试 16 局全负、均差 −1984 至 −3198；0.25–0.4 更好 | 先复现并校准真实行为分布 |
| 纯网络与搜索 | yan-v2 纯网络 3–13，但配搜索后线上强于 v1；ft3 纯网络 15–1，但配旧搜索对 v7 0–8 | 各重要 checkpoint 同时评估有/无搜索 |
| 单线程 CPU profile | next_state≈0.2ms、encode≈0.15ms/侧、96×6 网络≈1.5ms/行 | 优先验证 Ryanx 快速引擎与批推理吞吐 |

用户转述的 Master3s2 自博弈 RL 训练经历是可行性线索，但不能由此推出 Master7sE 的准确训练算法或我们必然成功。本次实际核对到：仓库已有完整 CNN 权重训练/导出、合法观测碳图推断工具；**v5 README 还记载过 Apollo 风格名称与 embedding 行号错位**。[9] 因此复用既有资产很有价值，但先锁定正确权重、编码、style、采样与部署一致性。

### 0.2 核心假说

保留原裁判、合法特征、历史权重、回放、兼容旧搜索及竞技场。训练不调用复杂 v8 搜索，不继承新增种树、护树、回库等规则作为奖励或强制动作。允许旧搜索作为独立评估器。首要检验：**同一起点、同一 decoder/温度、同一评估条件下，on-policy RL 是否优于冻结 BC，并能向相同搜索迁移。** 如果仅战胜训练对手，不能宣布整体变强；如果纯网络提高而旧搜索退化，先查接口/续局失配，不立即否定 RL。

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

第一轮只做现成网络适配、参考环境与快引擎验证、固定风格和温度的基线、短 PPO 与双模式评估。不要先实现动态联赛、新网络、复杂任务规划、全量搜索或异步分布式训练。

## 2. 数学问题：一个队伍是一个 agent

Carbon 是部分可观测的双人同时行动有限时域博弈。以裁判状态 s、各自合法观测历史 h^i、联合动作 a^i 表示：

$$
s_{t+1}=F(s_t,a_t^1,a_t^2), \qquad
a_t^i \sim \pi_\theta^i(\cdot\mid h_t^i).
$$

同一回合中，双方先各自决定完整动作，再统一结算；不能让任意一方看到另一方尚未执行的动作。自身控制的所有 worker 和基地可共享信息，因此第一版把整队视作一个策略 agent，而不是十个独立 RL agent。游戏中的队内冲突、空载压制、树与运输的机会成本，都应成为策略学习的对象。

标准配置为 15×15 环面、每方两个基地、最多十名工人、树寿命 50；episodeSteps=300 对应最多 299 次正常动作推进。当前 vendored helpers 的实际招募只检查工人总量与现金，角色上限检查被注释掉，不能直接根据 JSON 中的 planterLimit/collectorLimit 进行错误 mask。平台配置若不同，按实际裁判和配置处理。[5]

裁判状态包含隐藏的 full_carbon。actor 只能读取玩家可见的 carbon、公开双方资产、合法历史；未观察格的零值不能等价于已知零碳。视野的构造以裁判实现为准。禁止用地图 seed 反推资源图，禁止借另一席观测或未来回放帧填补自己的隐藏信息。

## 3. 奖励：对齐实际计分，保留目标不变的密集反馈

### 3.1 单局结算与平台 rating 不是一回事

正常满局以裁判最终分数判断胜负，通常等于现金。提前淘汰时被淘汰方得负分、游戏提前结束，不能使用剩余现金冒充胜负。超时、非法指令、环境错误单列，不能隐藏这些失败或当正常软分样本。

Claude 转述的规则是正常对局胜负与分差软分各半。本次未能独立打开平台说明核验，也未取得准确软分函数、归一化方式和淘汰/弃权计分。Codex 首先存档官方原文、URL 和时间戳；如精确效用公开，则单独登记。不要从几个 Elo 变化推定唯一公式。

### 3.2 主实验使用混合终局效用

正常完赛令 D_T=C_T^1-C_T^2，采用有界、反对称的代理：

$$
U_1=\beta\operatorname{sign}(D_T)+(1-\beta)\tanh(D_T/K),\qquad U_2=-U_1.
$$

首轮 beta=0.5、K=1500（Claude 建议 1000–2000 的中点）。平局为 0；正常提前淘汰先按裁判赢家给 ±1，作为等待平台特殊计分核验的临时约定，不把负裁判分与现金差机械混算。纯胜负 beta=1 为独立对照。K=1000、2000 可用于离线敏感性报告，不在最终测试集反选 K。若官方公开不同函数，另登记同预算比较，不静默修改已运行协议。

纯胜负不直接奖励同为胜局的大胜或同为败局的小负；纯现金差又可能过于偏好巨额分差。混合目标更符合目前已知的计分描述，但 **tanh 是近似，不是已验证官方软分，最大化 U 也不保证最大化动态 rating**。同时报告胜率、实际分差和混合效用；不把软分提升直接表述成胜率提高。

### 3.3 Potential-based shaping 不改变混合目标

非终局势函数仅使用公开现金：

$$
\Phi_1(s_t)=\alpha\tanh((C_t^1-C_t^2)/K_\Phi),\qquad \Phi_2=-\Phi_1.
$$

终局（含淘汰）设 Phi(s_T)=0。原始 r_t^0 非终局为 0，最后一步为 U。训练奖励：

$$
r_t=r_t^0+\gamma\Phi(s_{t+1})-\Phi(s_t).
$$

则折现累计满足：

$$
\sum_{t=0}^{T-1}\gamma^t r_t=\sum_{t=0}^{T-1}\gamma^t r_t^0-\Phi(s_0).
$$

gamma=1、固定初始状态时只相差动作无关常数。首轮 gamma=1、alpha=0.5、K_Phi=1000、GAE lambda=0.95（可配置初值，非实测最优）。将 beta=.5 vs 1 与 alpha=.5 vs 0 分成两个同预算单变量对照；不要同时改变目标与 shaping。塑形是否提高样本效率仍需实证，完美 critic 下其作用甚至可被价值基线抵消。[6] 不再添加树数、死亡、货量等永久奖励。

单测满局、淘汰、平局、随机轨迹、两席反对称、telescoping 和 chunk bootstrap。训练切片不是终局，不得漏掉末步势函数抵消。gamma<1 可能改变提前结束偏好，不宣称完全保留不折现目标。

RL critic 的塑形标签为 E[U|h]−Phi(s_t)（gamma=1）。旧 value 是现金尺度且可能包含现金 residual，不可直接用作新效用。首轮在复用 trunk 的基础上加独立 scalar RL value head（或重置旧价值分支并验证尺度），短 warmup 新 critic 再联合更新 actor；旧搜索 value 保留完整冻结快照，见 §11。

## 4. 环境：先证明能大规模正确生成经验

### 4.1 E0 参考环境

先封装现有 benchmark.py 的 configure_engine/new_game/observe 和 vendored carbon.compute_next_board，不另写结算逻辑。训练 API 建议 reset(seed, config) -> obs_both；step(action_both) -> obs_both, reward_both, terminated, info；两个 actor 从同一旧状态各自获得合法视野。内部全量状态与 actor 输入明确分离。

现有 arena 每席独立进程、计入通信与墙钟，用于可靠实战评估，但不是高吞吐 RL collector。训练可以在一个 worker 中运行多个独立环境，或批处理多个棋局。每个 episode 自己持有状态、单位 ID 映射、观测历史和随机数。先确保不会跨局共享 v8 的模块全局状态、复用 belief 或混淆棋手随机数。

需要核对：隐藏资源、初始状态、环面坐标、ID 生成/顺序、招聘顺序、现金消耗、人数限制、两人同时移动、碰撞赢家/平局、碰撞货物转移、在任意基地发生的归仓、种/夺树、树龄、碳再生、结束与计分。尤其同类动作之间的判定顺序、浮点相等导致的碰撞分支、树的市场价格不能被粗略近似。原版实现可能存在特殊边界，训练环境要忠实于裁判，不自行修规则。

### 4.2 高速模拟器的验收

先用参考环境测速度，拆开 step-only / encode / batch inference / rollout end-to-end / optimizer。记录 CPU 核数、线程、设备、内存、batch、每秒双边环境推进数和每秒学习方 team-transitions 数。

优先评估 Master3s2 内嵌的 Ryanx next_state 快引擎候选，只有通过逐步 parity 才用于训练。检查是否仅支持特定配置，是否完整实现真实全量状态、部分视野、终局/淘汰、ID 顺序和各类规则。旧 search 的 belief rollout 不是训练全状态环境。若不能严格同义，则回退参考引擎或实现紧凑 batched engine。先做最小 parity，包含：

- 人工构造的边界状态（至少覆盖上述事件每一种以及多个同时发生）；
- 固定初始布局上的随机合法联合动作；
- 高手真实回放及 BC/历史策略生成的高密度状态；
- 至少 10,000 个逐步对照，另留至少 100 个完整/提前结束的独立轨迹做连续一致性检查，数量为工程起始门槛，不是正确性证明。

比较每一步双方现金、货物、位置、角色、树及树龄、隐藏碳图、每席观测、status/reward、下一步动作合法集合；ID 可做一致映射，不允许把不同单位混同。离散事件必须一致，浮点误差的容许值事先明确，并验证不会改变相等/大小比较、合法性、赢家。首个 mismatch 保存可复现状态与动作，修复并加 regression。parity 失败就回退参考环境。

### 4.3 预算与吞吐

统计 E = 双边环境推进次数（不是单个 worker 动作数），L = 参与训练的 team-transitions。一次完整对局约 299 E；1M E 约对应 3344 场满局，提前结束会增加实际场数。双方均参与训练时 L 约等于 2E，与冻结对手训练时通常 L 约等于 E。报告真实数据而非仅报 epoch。

以实测端到端吞吐 q（E/s）计算 rollout 时间 E/q，另加优化、评测、保存成本。Mac CPU/MPS、Linux CPU 或有 GPU 的环境均可用；先 benchmark，GPU 不一定比小 CNN 的 CPU 批量推理快。避免每一步创建进程/重复加载权重；限制 BLAS 线程，减少 Python 对象与跨进程序列化。不要在每个 rollout step 调用带完整搜索的 v8。

Claude 的 profile 粗算 0.2+2×0.15+2×1.5≈3.5ms/E，单核理论约 286 E/s，尚未计入 belief 更新、采样、序列化、优化和评估。多进程不一定线性加速；不能承诺 1M E 的完成时间。实测单进程批量与多进程单线程的 q，再估计成本，重点 batch 网络 forward。

首轮只安排 10k E smoke、100k/200k E 学习检查点（约 334/669 场满局）。若有信号，另一个训练 seed 复验，再考虑 0.5M、1M E；更大规模取决于学习曲线与实际成本。若 200k 无增益，先查温度、decoder、logprob、critic、探索、对手难度，必要时单变量短复验，不直接扩大。报告 E、L、实际局数、forward 数、总墙钟、优化成本与评测成本。

## 5. 策略：优先 Zero-W，后续 Zero-P

### 5.1 复用现成强权重、54 通道与 BeliefTracker

首选 Claude 提到的 **ft3 Apollo 固定风格**。本次未确认 ft3 的准确文件与权重；请 Codex 从 Claude 实验交接或本地资产目录恢复，存档源码/权重 SHA-256、训练来源、模型结构、style 名称与实际 embedding 行号、动作/温度/评测协议。若缺失，明确记录，不阻塞；从仓库冻结 Master3s2 或相容的 v5/v6r 全网权重中选可复现替代，并重新建立 baseline，不能把替代品叫 ft3。

已有参考实现：experiments/v5/distill.py 的 Student、refit.py 的 load_full、experiments/v6r/train.py 的 load_parent，以及 model_export.py 的嵌入权重导出。[9] 优先复用相应 54×15×15 encode、辅助全局输入、归一化/量化、CNN worker/base logits。网络宽度与 block 数以实测权重为准。训练前逐头比对原模型、可微 PyTorch 版本、导出文件（含两个出生位、合法历史、旧/new value 的接口），并测试初始化时动作概率一致性。原仓库历史 Apollo 命名/索引错位必须核对。

合法 BeliefTracker 可从可见碳量、公开资产和已知生成对称结构推断未知资源；估计不是真值。按相同真实历史逐步更新 tracker，严格分局、分席独立；禁止 full_carbon、对方不可见观测、未来帧及 seed 反推。测试隐藏状态改变但合法历史相同时 actor 输出不变。对称先验必须符合平台实际规则，不直接假设旧估计在任意阶段更准确。训练与导出都复用同一版本。

**先固定一个强 style，不做动态路由。** 训练只使用锁定 style；共享 trunk 更新可能改变其他风格输出，不能认为未更新 embedding 就保留旧风格。冻结对手必须独立持有完整旧参数。

### 5.2 温度校准必须先于 RL

设 BC logits 为 z：

$$
\pi_{\theta,T}(a\mid h)=\operatorname{softmax}(z_\theta(h)/T).
$$

Claude 的 T=0.4 为首选，开发面板先复现 T=1 与 0.4（必要时再测 0.25），固定权重、同解码、同图换边。worker/base 若原先温度不同，分别核验。先记录强实际行为，再锁定 ZW0 冻结起点；温度改变带来的收益**不计入 PPO 增益**。

建议以显式 logit_scale=1/T 适配；只有最后一层确是直接产生 logits 的 affine 时，才等价折叠 W/T、b/T。不要修改 trunk、embedding、critic。旧搜索可能还有 _TEMP 或不同 root/rollout/rival 温度；核算有效总温度，不重复除 T。若 logits 幅值用于排序、阈值或 gate，另做兼容核对。PPO old/new logprob、entropy、KL 必须基于同一合法采样分布；首轮固定 T 并记录人口分层的 KL/clip、entropy。过尖分布的探索问题后续单独消融。

### 5.3 联合动作和 decoder：防止强化学习的概率与实际动作错位

整队作为一个 agent。worker 五动作（STAY/四方向）、base 三动作（NOOP/招募两类）；STAY/NOOP 对应不下发 key。简单因子化：

$$
\pi_\theta(\mathbf a\mid h)=\prod_{j\in U_t\cup B_t}\pi_\theta(a_j\mid h,j).
$$

旧 policy 可能进行按采样概率重排序、冲突避让、招聘修正等，**禁用 _search 不一定等价于纯神经策略**；仓库 v5 policy_duels.py 是这种历史诊断。需确认 Claude 实测的无搜索条件是否还有其它后处理，复现 W-raw，再建立 ZW0 的可训练解码。

ZW0/ZW1 共享同一可重放 decoder。首选固定顺序、显式合法 mask 的实际动作采样。可保留语法合法但失败的招聘/同格碰撞，由裁判处理；若使用条件避让、预算 mask，则每一条件概率、顺序和前缀都要记录并重算。不能采样后经随 theta 改变的规则强改最终动作，再把原始 categorical 概率视为最终指令概率。若必须用固定、参数无关的 deterministic latent decoder，可记录 latent 采样与概率并明确优化 latent policy；不把 latent 概率当作最终指令概率。首轮优先简单可审计实现，不继承 v8 手工选招。

保存 UID、mask、动作顺序、采样指令、最终下发指令与 old logprob；同模型同输入同随机流复现。decoder 适配可能改变原 ft3 棋力，必须报告 W-raw→ZW0 变化。RL 增益相对适配后的 ZW0，而非不同温度或规则的原网络。训练时不可查看敌方当步动作。

### 5.4 Zero-P 是独立对照，不是前置条件

首先比较相同 54 通道、相同架构的随机初始化与 warm-start（避免将初始化与网络改动混在一起）。然后再做独立 P0 小 CNN（48–64 通道、4 residual block，合法可见/历史碳+mask、双方单位/树/基地及现金）、P1 Unit Attention、P2 自回归联合动作。P1 gather 双方单位、基地、全局 token，环面相对位置和 padding mask；P2 的条件概率为：

$$
\pi(\mathbf a\mid h)=\prod_j\pi(a_j\mid h,a_{<j}).
$$

先基地后 worker、顺序及前缀可重放，BC teacher forcing、RL 用实际采样前缀。旋转/镜像/阵营增强要同步变换动作、历史、视角，不泄漏等价地图。GRU 或独立训练期全状态 critic 可在 pipeline 可靠后做单独消融，不能泄漏给 actor。不同时加入新编码、Attention、P2；若 W 有稳定增益，先延续 W 而非重写网络。

## 6. 初始化：分风格 BC、generic 与 RL 分开验证

第一轮直接使用已恢复、核验过的强 BC 权重，不强制重新收全量回放。Claude 的 16–0、8–8 与仓库旧多风格记录并非同一协议，先复现固定 style 的优势，再确定基线。

需要扩充时：收集公开前六所有可获取的精确版本回放，尤其互战、与其他对手、胜负局；前十中的自身与旧版单列。冻结 manifest（版本、来源、时间、对手、布局 SHA、缺失与分页覆盖），整局/地图分组 train/dev/locked test，两席、换边、等价布局不能跨集合。重建本方合法观测历史；只从真实指令取确定标签，若由后继状态推断（出生、死亡、STAY、失败招聘）则明确不确定并 mask，禁止完整资源图进入 actor。现有 elite-refit 入口在实验分支 [7]。

按教师、对局、阶段均衡，保留 teacher embedding 及 generic 行，用同预算、同温度、同 decoder 在开发面板选择一个固定专属 style；不要求运行时识别对手/教师身份。也可将不同专属 BC 策略冻结为联赛对手。整队 BC loss 按有效 worker/base 归一：

$$
L_{\rm BC}=-\mathbb E[(\sum_{j\in U}\log p(a_j)+w_B\sum_{j\in B}\log p(a_j))/(|U|+w_B|B|)].
$$

w_B=1 起步。单列招工、普通移动、种夺树、返库、碰撞的 NLL 与闭环成绩。教师胜局不保证每步最优；teacher-forced NLL 不能替代实际对局。新 RL value 若预训练，目标使用混合 U−Phi，而非旧 20/52/100 步现金预测。后续可加同额外计算预算的继续 BC 组，区分额外计算与环境反馈。

## 7. PPO：让网络从当前自己犯的错误中学习

### 7.1 算法与实现细节

用同步 PPO 先建立正确 baseline。首轮从同一已校准的 ZW0 复制 learner、独立冻结对手，整批 pilot 固定对手，均衡两席与地图；仅 learner 的 transition 用于 PPO。一次 rollout 固定行为策略版本。新效用 critic 短 warmup，再联合更新 actor；其环境步与计算成本也计入预算。存储 h_t（或可无损重算的输入）、实际指令、有效 action mask、解码顺序、旧 joint logprob、value、reward、terminal、seat、opponent_id、policy_version、布局。重放前缀时必须精确复现 logprob；在 optimizer step 前随机抽样核对 new/old ratio ≈1。

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

初期 BC+RL 可小步更新（lr=1e-4 起步，必要时降到 3e-5）；若示范能力严重退化，再试逐步减弱的 BC loss 或 KL 到冻结 BC reference，必须对照无 anchor 的 RL。anchor 是优化正则，不是长期必须保留的规则；不能用它把策略锁死在老师分布。禁止把旧策略、冻结对手或专家的轨迹当当前 PPO on-policy 样本。它们可另做 BC，或者训练一个独立 exploiter。

### 7.2 小步先验收再放大

在投入百万步前做以下 smoke：
- 手工短 episode 的 reward、GAE、terminal bootstrap 核验；小型可计算的 toy simultaneous game 或固定简单对手可学习性测试。
- 固定 seed 的 rollout 可复现；logprob 对齐、网络更新有非零梯度、KL/entropy/value loss 有合理范围。
- 观察随机和 BC 策略的合法出招、招聘发生率、是否长期空动作、提前淘汰率。
- 至少几次 PPO update 后，checkpoint 能恢复训练、导出与原模型输出一致。
- 在真实原裁判（不是同一个快速 engine）上跑完整对局，包含未训练 seed。未过这些检查不扩训练。

## 8. 多对手训练：从第一天就避免“只会打上一代”

### 8.1 首轮先固定、分层的对手池

首轮固定一个 ZW0 对手为了低成本隔离学习机制，不是最终训练分布。**从第一批评估就用多样外部冻结对手。** 有信号后先扩固定训练池，再做动态 League。扩展后的训练池包含：
1. 可快速运行的合法简单策略与弱版本（保证训练早期有可学习信号）；
2. 冻结 BC 快照（generic 和若干真实教师风格克隆，前提是已有足够数据且实战验证其风格差异）；
3. 冻结 Zero 最近快照/初始 BC，对手每局固定；
4. 少量历史 v6/v7/v8 或它们明确标记的快速无搜索版本。完整 v8 推理昂贵，先只用于评估或极少量独立训练，不要把无搜索代理成绩当作完整 v8 成绩。

训练双方可来自同一个冻结行为版本，也可为当前 learner vs frozen opponent；只有符合当前 on-policy 条件的一侧/双方进入 PPO。采样先按类别均衡，保留随机对手和不同难度。开发评估池按预登记规则选择 checkpoint；最终 locked eval 不参与 PPO、选样本或调整采样。评估池包含固定 hash 的不同版本、公开强策略、未参与训练的代理家族及最新真实回放诊断。旧 v8 可以作为训练锚点或 locked eval，若训练使用了 v8，就另留未训练的强对手来测外推，不能把该项称为未见对手泛化。

### 8.2 稳定后再建立动态 League

为每个 frozen checkpoint 保存模型、编码、配置、文件 hash、胜率表与生成来源。每隔约 100k–250k E 快照一次（按吞吐调整）；只有实际能力或风格变化才保留到 active pool，旧 checkpoint 归档不丢失。保持一定数量近期、历史、不同家族及专门 exploiter，避免池不断膨胀。

建议初期 opponent mix（待测）：约 30% 最新冻结自身，30% 历史 Zero/BC，25% 多风格代理，15% 其他可快速运行的历史策略；某类为空则显式重分配。根据配对胜率 w_i 优先抽学习价值较高的对手，例如 w_i(1−w_i) 的 curriculum，以及 (1−w_i)^2 的弱点采样；两种可分比例混合，并加 uniform floor/家族上限。禁止将所有 rollout 都给当前最难且几乎必输的策略。对手采样权重只在批次边界更新，对手在一整局内冻结，避免 non-stationarity 和数据归因混乱。

出现 A 胜 B、B 胜 C、C 胜 A 时，用 payoff matrix 保留相互克制策略，而不是只按单个 Elo 淘汰。若主策略有长期弱点，另训 best-response/exploiter 针对冻结主策略，再把 exploit 加入联赛；训练 exploiter 的数据不直接冒充主 learner 的 on-policy 数据。可选 PSRO/近似 Nash mixture 是后续路线，不能在小规模结果上声称求得 Nash。AlphaStar 的 league/PFSP 是可参考机制，不是 Carbon 中已经验证的具体配比。[8]

## 9. 最小实验矩阵：先验证 warm-start RL

| ID | 起点 | 训练/用途 |
| --- | --- | --- |
| W-raw | 原始 ft3（或明确命名的可复现替代） | 无；复现既有采样、温度和棋力 |
| ZW0 | 同权重，固定 style、校准有效 T≈0.4、可计算 logprob 的 decoder | 无；冻结正式基线和训练对手 |
| ZW1 | ZW0 的独立副本 | PPO，混合终局效用 + 势函数（首轮主实验） |
| ZW2 | ZW0 | PPO，同预算，只改 beta=1（纯胜负） |
| ZW3 | ZW0 | PPO，同预算，只改 alpha=0（无塑形） |

ZW1 先 10k E smoke，再 100k、200k E 两个检查点。首轮固定 ZW0 对手，均衡出生位，只有 learner 侧参与 PPO。ZW2、ZW3 待 pipeline 与 ZW1 有可解释信号后执行，不做第一轮全网格。建议一个 seed 先跑通，再用第二 seed 复验。保留全部失败和停止记录。

每个重要 checkpoint 用两条独立评估路径：
- P：ZW1 无搜索 vs ZW0 无搜索；二者分别对同一冻结外部对手面板。
- S：ZW1+冻结相容旧搜索 vs ZW0+完全相同搜索；二者分别对同一外部面板。§11 明确只替换 actor 的干净对照。10k smoke 可先检查 S adapter parity，100k/200k E 应尝试 S；若兼容失败，报告 blocker 而非把失配当 RL 失败。

M0 环境/权重/速度通过；M1 W-raw→ZW0 固定风格、温度、decoder、基线通过；M2 PPO smoke；M3 100k/200k E 学习曲线、外部对手及第二 seed 复验。若正趋势，下一批先扩固定多对手池，再动态 League；再做相同架构 scratch、generic/专属 BC、继续同计算预算 BC、Zero-P 和协同表示消融。0.5M/1M E 和更大规模需要实证与预算支持。若无信号，先定位，不直接扩大或恢复规则补丁。

M5 正式晋级仍在冻结 v8、不同家族和独立地图下完整限时换边，最终由线上真实前六验证。首轮不以必须战胜 v8 为完成条件，而以是否获得可复现、自主的学习增益为目标。

## 10. 评估：混合效用、胜率、现金差、迁移与泛化

每个 checkpoint 报 W/D/L、win+0.5 draw、正常完赛现金差（均值、中位、分位）、U(beta=.5,K=1500)、K=1000/2000 离线敏感性、提前淘汰/异常率、两席、各对手家族、耗时。若官方软分能复现另报其口径；不能只报有利指标，不能将异常偷偷从分母剔除。

评估固定对手（包括可复现 Master3s2、历史 v4/v6/v7、少量完整 v8，以及未参加训练的代理/公开策略）。ZW0 与 ZW1 各自对同一外部面板，用配对差比较，不仅看 ZW1 战胜 ZW0。开发初筛可每对手 4–8 张新图换边，完整搜索昂贵时预登记更小子面板，之后至少 16–32 张独立图确认。按地图聚类 bootstrap、保存所有训练 seed/checkpoint；开发用于选模，locked test 不反复查看。相同根的多步、多次续局不是独立比赛。

若 ZW1 相对同温度/decoder 的 ZW0 在外部面板也有正趋势，第二训练 seed 复现后扩大；若只对 ZW0 改善，称为特定对手学习，优先扩训练分布；若 P 改善但 S 退化，检查 §11 的适配、旧 value/rollout，不立即否定 actor；若 P/S 均无增益，先停放大。小样本正均值只是复验理由。不得把本地克隆当真实前六。

发布按 AGENTS.md、ARENA.md 最新约定，冻结 v8，初筛及独立新图换边；可沿既有 16 局≥12 胜、每席≥5、确认 8 局≥5、每席≥2、均正分差作为建议起始门槛，以事先登记维护协议为准；同时报告混合效用与多对手回归。实际候选导出后：

~~~sh
python scripts/selfplay.py \
  --candidate /path/to/zero-export.py --baseline yan-v8 \
  --seeds $SEEDS --jobs 1 --record-actions \
  --output arena-runs/zero-v8-confirm.jsonl
python arena_report.py arena-runs/zero-v8-confirm.jsonl
~~~

SEEDS 来自锁定协议。真实平台规则/对手状态只能由正式比赛检验，上传由维护者决定。

## 11. 搜索：提前测试迁移，而不重新堆搜索规则

Claude 的无搜索/有搜索差异是重要提醒。**兼容旧搜索是早期评估适配器，不是首轮 PPO rollout 的控制器。** 先用相容的旧版（例如冻结 yan-v1/v3-search），核对网络接口、风格、54 通道、附加输入、量化、坐标、logits/value 尺度、温度以及原 search 的 root/rollout/rival 路径。不能声称直接换权重一定兼容；失败则报告。

S0=ZW0+冻结相容搜索；S1=ZW1+同搜索。第一项最小对照只替换己方 root proposal/policy logits，保留原 opponent policy、rollout policy、旧 critic 完整冻结参数（包括共享 trunk；只冻结 value head 不够）。RL critic 学的是混合效用，**不可直接塞入旧现金评分**。如果旧共享网络需另取冻结 value，保持 S0/S1 相同 forward/预算。核验目标 style、等效温度（不二次缩放）、候选进入实际搜索、最终执行、序列化和延迟；先固定搜索形状，再相同真实限时，检查搜索次数/深度/慢步/回退。两者只改 actor，不能混入新的对手模型与经济规则。

若 P 有增益但 S1 无，可能旧 rollout、critic 与新 actor 分布失配。再单变量做 S2=进一步替换己方 rollout policy；其后才考虑对齐整个 search score 与混合终局效用，保留 opponent 不变。不要将 S2 对 S0 的总变化归给根策略；若旧搜索限制了新 actor，保留无搜索版本作为有效实验。

真正的学习型 search improvement/蒸馏闭环仍放在后面。Carbon 是部分可见、双方同时行动、多单位联合动作；不能照搬完全信息 MCTS。待 actor 有竞争力，再在合法 belief 上比较多个整队候选，使用共同敌方回复、对齐终局效用的 critic；若同预算 search 有收益，将改进策略蒸馏回 actor 再 RL。不得借真实隐藏资源或敌方未执行动作选招，也不重新堆未经验证的经济评分项。

## 12. Codex 第一批执行任务（更新版，先从这里开始）

**立即执行 Zero-W 的最小试点，不要等待 Zero-P 或完整联赛。第一轮目标 10k E smoke、100k/200k E 检查点（受资源约束可缩减并报告）；不自动上传榜单。**

- [ ] 拉取最新 main，独立 experiment/carbon-zero worktree；保留 v8 与原实验，冻结裁判、配置、源码/模型 hash。
- [ ] 从 Claude 交接恢复真实 ft3 Apollo 权重、style 索引、encode、BeliefTracker、温度与 decoder，以及其无搜索/有搜索原始评估（若可用）；缺失时采用明示的可复现替代起点。核验官方计分说明，记录未能核实的细节。
- [ ] 复用已有可微 CNN 训练代码和 export，逐头核对输出及风格索引，核验合法观测、跨局/席隔离；复现 W-raw 并校准 T=0.4，建立可计算联合 logprob 的 ZW0。冻结 ZW0，记录因温度、decoder 改动产生的独立影响。
- [ ] 原裁判适配，Ryanx 快引擎 parity（边界、随机动作、回放、真实长局、淘汰、双方观测）；不通过就回退。测实际 E/s、L/s、batch/多进程、训练与评测总成本。
- [ ] 实现混合 U、potential shaping、新 RL value 与同步 PPO，验证 telescoping、old/new joint logprob、GAE、terminal/切片、恢复训练。固定 ZW0 对手，从相同 actor 开始 ZW1。
- [ ] 10k E smoke；在 100k/200k E 保存 checkpoint，冻结对手及开发/locked 图协议。P 模式用同温度/decoder 的 ZW0 对照；同时做 S0/S1 相同旧搜索迁移测试及固定外部对手面板。若兼容失败，单列原因，不使用错误接口结果定性 RL。
- [ ] 保存全部胜败、正常/淘汰/异常分层、分差/U、各席、PPO 指标、环境步、实际运行成本、训练/模型 hash 和可复现导出。报告实际完成与尚未跑的实验。实验代码和结果提交推送独立分支，权重/回放遵守许可、大小限制，必要时提供可获取存储及校验 manifest；不提交凭据。
- [ ] 有可复现增益后才扩数据、对手池、第二 seed、reward 单变量对照与更大预算；若无增益，先排查实现/起点/训练稳定性，报告并选择一个变量短复验。不要在本阶段改变 main 的 submission.py 或恢复旧规则补丁。

首轮报告应明确回答：① 温度和适配后的起点是否达到可用棋力？② 100k/200k E 相对 ZW0 是否提升、是否只会打训练对手？③ 旧搜索能否利用新 policy？④ 训练的真实瓶颈及扩预算的理由？  
**本次目标不是马上打赢前六，而是用最少的工程与计算证实（或否定）基于强 BC 起点的 RL 学习机制，随后才扩大。**

## 参考与仓库证据

[1] [上一轮实验交接](yan-v8-next-experiments-20261124.md)：持续任务、回本及联合动作的固定版本、实验边界。  
[2] [最新评分审计](https://github.com/Zhaoyi-Yan/carbon/blob/6ade8e72b2efbc2b345f76a1bd45000aede20ce3/analysis/v8-score-audit/README.md)：关键回复独立验证尚未改善选招。  
[3] [own-search](https://github.com/Zhaoyi-Yan/carbon/blob/6ade8e72b2efbc2b345f76a1bd45000aede20ce3/analysis/v7/own-search.md)；[selfplay-value](https://github.com/Zhaoyi-Yan/carbon/blob/6ade8e72b2efbc2b345f76a1bd45000aede20ce3/analysis/v7/selfplay-value.md)。  
[4] [ARENA.md](../ARENA.md)、[AGENTS.md](../AGENTS.md)、[scripts/selfplay.py](../scripts/selfplay.py)、[benchmark.py](../benchmark.py)。  
[5] [Carbon 配置](https://github.com/Zhaoyi-Yan/carbon/blob/f2d1f6a5cff5e08dd2f24ebcfa6e73ad0618019d/vendor/zerosum_env/envs/carbon/carbon.json)、[裁判结算与视野](https://github.com/Zhaoyi-Yan/carbon/blob/f2d1f6a5cff5e08dd2f24ebcfa6e73ad0618019d/vendor/zerosum_env/envs/carbon/carbon.py#L155)、[实际招聘、树价、移动与碰撞](https://github.com/Zhaoyi-Yan/carbon/blob/f2d1f6a5cff5e08dd2f24ebcfa6e73ad0618019d/vendor/zerosum_env/envs/carbon/helpers.py#L951)。  
[6] Ng, Harada, Russell (1999), [Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)（势函数塑形）；Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。  
[7] [elite-refit 数据与训练入口](https://github.com/Zhaoyi-Yan/carbon/tree/5bda5e5b033b26ac925c02e229e2ba1b6751ab88/experiments/elite_refit)；[覆盖和训练归档](https://github.com/Zhaoyi-Yan/carbon/tree/5bda5e5b033b26ac925c02e229e2ba1b6751ab88/analysis/elite-refit)。  
[8] Vinyals et al. (2019), [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://doi.org/10.1038/s41586-019-1724-z)（联赛训练与 PFSP）；Yu et al. (2022), [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)（PPO 的多智能体 baseline 参考，非 Carbon 结果）。

[9] 仓库已核对的复用资产：[v5 README（Apollo embedding 索引、旧无搜索结果）](https://github.com/Zhaoyi-Yan/carbon/blob/fe8a3420f53311f35cb18b5e7fec821e829cf6fd/experiments/v5/README.md)、[Student](../experiments/v5/distill.py)、[load_full](../experiments/v5/refit.py)、[load_parent](../experiments/v6r/train.py)、[model_export](../experiments/v5/model_export.py)、[历史 no-search 诊断](../experiments/v5/policy_duels.py)、[合法碳图推断](../experiments/v5/fog.py)、[旧快速引擎所在 Master3s2](../opponents/claude-AlphaCarbon-Master3s2.py)。

以上外部工作用于说明方法来源；Claude 的新数字来自用户提供的复核文本，需按 §0 和 §12 复现；均不构成 Carbon-Zero 成功的实验证据。
