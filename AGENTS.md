# 策略迭代交接

每完成一轮策略迭代，把策略代码和该轮已有的评测结果一起提交并推送到本仓库的 origin。采用的版本更新 main 和 ITERATIONS.md，并打一个新标签。实验候选保留在实验分支，尚未评测时标明状态。

对局结果注明对手版本、地图种子和出生位，同一版本保留败局记录。游戏排行榜的上传由维护者决定。

每轮新候选先挑战我们自己线上已验证的最强版本。冻结基线的 Git 标签和文件 SHA-256，候选与基线在相同地图交换出生位，以双方都正常结束的完整对局判断胜负。跨代对照 yan-QQ-v6 使用标签 v9-yan-v6-estuary 的交付文件。yan-v5 的冻结标签为 v8-yan-v5-multistyle。前代 yan-v4 为 Claude 本地最终版，yan-v3 对应 v6-tree-guard-batch。保留 yan-v2（v5-tree-value）和 yan-v1（v3-search）用于跨代回归。Master3s2 仅作历史参考。晋级前用开发时未使用的多张地图复核，记录逐局分差和单边失利。基线随新主力一起更新。

维护者于 2026-11-25 确认 v8 own-prior 已作为 yan-QQ-v9 上线，公开策略 ID 为 `0f9896ed3dc3cd7bfad29fe2dd38d6fb`。本地冻结文件为提交 `e97a386b290a1727f26331470094ab57cd452948` 的 `yan-v8-own-prior.py`，SHA-256 `83bc00ca52ba42ece876830649c9bc488105e763179b62018558f555bb94fa9c`；线上回放不含部署源码 SHA，该对应依赖维护者说明。当前具体实验协议见 `analysis/v9-leader-study-20261125.md`，从完整 v9 做单候选开局消融。`scripts/selfplay.py` 尚无 `yan-v9` 别名，必须显式传冻结文件路径作为 `--baseline`，不能省略参数或误用 `yan-v8`。前代 yan-v8 标签 `v11-yan-v8-elite-refit` 和 `--baseline yan-v8` 保留供跨代复测；此次只更新研究交接，不创建新的策略晋级标签。
