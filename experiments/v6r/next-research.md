# 晋级文件之后的离线候选

本页保留在实验分支。已在 12 张新图打过换边赛的策略固定于标签 v9-yan-v6-estuary。下面是可在该单文件上继续运行的实验，尚未进行完整换边评测。

- 依据各工人的公开位置重建对手已经看见哪些格子。10 个不同公开回放、双方视角、最初 150 步，资源可见性标记的错误率由 15.48% 降至 0.35%。保留己方对资源数值的估计后，3,054 个实际对手工人动作的对数损失从 0.6114 降到 0.6021。对照中同时模拟对手的资源数值会固化错误的猜测，损失反而增加到 0.9002。[可见性对照](opponent-view-mask-check.json) · [同时模拟资源数值](opponent-view-check.json)
- 用双方公开的工人位置、货量、消失及现金变化，估计矿场产出与碰撞缴获。48 场先前的实战、192 个阶段，对原版裁判地面采集差的平均绝对误差 13.28，实际采集差的平均绝对值 214.86；缴获差的对应数字为 38.28、122.92。候选在持续少采且缴获与现金优势不足时，提高安全矿区内 v5 采集网络的占比。[对照记录](flow-check.json)
- [联合走位](clearance.py) 使用已有的最优分配算法，在工人、基地与对手可能落点之间协调单回合行动。[提前结算](knockout.py) 在推演中识别没有工人、没有树且无法再招募的玩家。

依次运行下列命令可生成包含四项改动的后续候选。晋级文件仍可单独作为对手。

```sh
python experiments/v6r/clearance.py yan-v6.py /tmp/carbon-clearance.py
python experiments/v6r/opponent_view.py /tmp/carbon-clearance.py /tmp/carbon-perspective.py
python experiments/v6r/knockout.py /tmp/carbon-perspective.py /tmp/carbon-bankruptcy.py
python experiments/v6r/flow_switch.py /tmp/carbon-bankruptcy.py /tmp/carbon-challenger.py
```
