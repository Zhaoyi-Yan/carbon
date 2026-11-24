# v8 own-search 候选 prior 重拟合与端到端验收

2026-11-25。状态：**单一候选的有限尝试；预注册确认门槛通过，但幅度很薄，未上传、未替换发布文件。**
路线依据：[Carbon 路线调整](../../carbon-route-reset-20261125.md)（main `82b7906`）。

## 1. 协议

- 冻结基线：`v11-yan-v8-elite-refit`，`submission.py` SHA-256 `a87f8e6b83656c771cb5a25d6b6f383b245127e7c782e4c6c7c4a0a2cc15d396`。
- 只改一个已有模块：v8 的 own-search 工人候选 prior（`_V7_OWN_COEF` / `_V7_OWN_SCALE`）。网络、搜索配置、候选数上限、原生解码、招募/锁定工人、单步限时全部不动。
- 拟合只做一次，模型族与配方沿用 `experiments/v7/own_search_fit.py`（按职业的线性修正 + 温度）；不做模型/超参扫描。
- 对战验收：候选 vs 冻结 v8，12 张新图 × 换边 = 24 局；先 screen、后 confirm 两批独立地图；主指标为候选视角的平均终局 U。
- 预注册门槛：screen 平均 U > 0 且无运行故障才做 confirm；confirm 平均 U > 0 且按地图 bootstrap 的 95% 下界 > 0 才算独立确认。

## 2. 数据与来源

优先复用可追溯的线上 `yan-QQ-v8` 对局（回放仓库策略 id `52af7735a7677815efb48cdf94269e29`）：

- 33 场完整对局，**12,322 个训练工人标签**，按布局分 **22 训练 / 11 留出**；标签取最终实际下发动作（含搜索后修正），保留胜负局。
- 教师计算：0 秒新增（全部复用回放），报告在 `own-search-fit-v8.json`。
- 回放文件不带源码 SHA；本机冻结 v8（`a87f8e6b…`）是代理来源，这一点写入报告的 `teacher.note`，不把在线策略名当成源码哈希。
- 回放仓库 index SHA：`5bb471ce7e7b1ee4…`；逐场文件 SHA 与布局指纹见 `own-search-fit-v8.json` 的 `sources`。

## 3. 拟合

- 留出 NLL **0.33744 → 0.33545**，top-1 **87.71% → 87.97%**（11 张留出图，均值）。
- 模型：`scale=[1.5208715, 1.6333947]`，`coefficients` 2×27；只替换 v8 的两个 prior 槽位，未改其他代码。
- 候选文件 `yan-v8-own-prior.py`：**9,137,552 字节**，SHA-256 `83bc00ca52ba42ece876830649c9bc488105e763179b62018558f555bb94fa9c`；编译通过，构建记录见 `yan-v8-own-prior.json`。

## 4. 对战结果

两批地图与回放布局、v7 面板均不重叠（`panels.json`）；每批 12 图 × 2 席。

| 批次 | W–D–L | 平均 U | 平均分差 | 0 席 U / 1 席 U | 地图 bootstrap 95% | 正向地图 | 失败 / 最慢步 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| screen | 14–0–10 | +0.0428 | −156.3 | +0.061 / +0.025 | [−0.2394,+0.3175] | 5/12 | 0 / 0.758 s |
| **confirm** | **15–0–9** | **+0.1562** | **+112.1** | **+0.312 / −0.000** | **[+0.0060,+0.3313]** | 6/12 | 0 / 0.798 s |
| 合并（仅参考） | 29–0–19 | +0.0995 | −22.1 | — | [−0.0634,+0.2603] | 11/24 | 0 / 0.798 s |

48 局全部 299 步正常结束；0 failure；最慢单步 0.798 s。

## 5. 结论

- **按预注册门槛，confirm 通过**：平均 U +0.156 > 0，地图区间下界 +0.006 > 0。因此这个固定候选（SHA `83bc00ca…`）在两批独立地图上留下了“有改善证据”的记录。
- **但证据很薄，不能称稳定优势**：下界仅 +0.006；confirm 的 12 张图只有 6 张为正；提升几乎全部来自 0 席（+0.312），1 席为 0.000；screen 的区间跨零。合并 48 局的区间仍跨零。
- 这**不**证明能击败前六，也不等于线上排名提升；只说明“更新 own-search 候选 prior”这条接口在本轮预算下有一个方向性的正号。
- 上传与是否替换发布文件由维护者决定；本分支不修改 `submission.py`、不动 v8、未上传排行榜。

## 6. 复现

```sh
git clone --depth 1 https://github.com/JUZI1/carbon-league-replays.git /tmp/carbon-league-replays
python experiments/v8/own_prior_fit.py /tmp/carbon-league-replays analysis/v8/own-prior
python experiments/v8/own_prior_build.py yan-v8.py analysis/v8/own-prior/own-search-model-v8.json yan-v8-own-prior.py
python experiments/v8/own_prior_panel.py analysis/v8/own-prior/own-search-fit-v8.json analysis/v8/own-prior/panels.json
python scripts/selfplay.py --candidate yan-v8-own-prior.py --baseline yan-v8 \
  --seeds <12 个 screen 种子> --jobs 2 --output analysis/v8/own-prior/screen-24.jsonl --record-actions
python scripts/selfplay.py --candidate yan-v8-own-prior.py --baseline yan-v8 \
  --seeds <12 个 confirm 种子> --jobs 2 --output analysis/v8/own-prior/confirm-24.jsonl --record-actions
```

## 7. 边界

- 只重拟合了一个轻量候选 prior；离线 NLL 改善 0.26 个百分点。它可能引导旧搜索选错，也可能改变去重和实际搜索深度；本轮用完整对局判断，没有把离线指标当棋力。
- 回放来源没有源码 SHA；在线 `yan-QQ-v8` 与冻结 v8 的等价性依赖部署记录，不是回放文件能证明的。
- 未做完整搜索配置扫描、未加对手混合、未改 anchor/lr、未接新网络。
