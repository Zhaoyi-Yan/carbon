# 双策略主干压缩与校验

默认片段共 **5,826,403 ASCII 字节**，距 5,912,710 字节的上限还有 86,307 字节。冻结 v6 的 Base85 初始化占 5,146,610 字节；Harbormaster 差量及重建函数再占 679,793 字节。函数输出常规的 Base64 NPZ，两个策略都能用原有的 `_Policy` 构造器加载。

冻结 v6 的完整原始 NPZ 逐字节相同，SHA-256 为 `5b6bc7e4a169c16701752bc9909e31eb86d874d5e512194f0c5a8d053e6f79ae`。Harbormaster 的卷积与大矩阵按每个输出通道、每 108 个输入系数拟合差值尺度和码表。约 15% 的标量使用 6 位，其余使用 4 位。形状改变的 27 行风格嵌入及偏置等小层按原样保存。每个策略继续使用各自精确的独立价值头 `_VALUE`。

## 行动一致率

从 320 张不同的公开地图各取一个局面，剔除旋转、镜像和交换座位后的重复布局。用观察者实际可见碳、此前真实动作和累计随机抽样位置重建决策。两份 Harbormaster 在同一合法局面启动相同的路由状态，比较根提议与后续经济修正，跳过整棵搜索。

| 检验项 | 与原始 Harbormaster 一致 |
| --- | ---: |
| 合法工人根指令 | 2,720 / 2,722（99.93%） |
| 持货至少 65 的工人根指令 | 700 / 700 |
| 合法基地根指令 | 640 / 640，其中招聘 69 / 69 |
| 整队合法根指令 | 318 / 320（99.38%） |

工人和基地根 logits 的平均绝对误差分别为 0.00628 和 0.00675。价值误差均值为 0.000366，相当于 1.10 个现金单位，最大为 0.002584。两次整队差异都各来自一个工人。逐局错误、抽样地图、座位和回合均收入验证 JSON。执行完整限时搜索时，接近阈值的估值及时间预算都可能改变最终选招，未在这项数值检验中测量胜率。

## 初始化与集成

五个全新低优先级单线程 Python 进程，读取片段、编译、导入 NumPy、还原原权重、重建第二份 NPZ 并加载两个标准策略的冷启动墙钟中位数为 0.303 秒。这里只计权重初始化，卷积预热另计。

```python
from pathlib import Path
from dual_weights import build_pair

code = build_pair(Path("yan-v6.py"), Path("experiments/v7/checkpoints/harbormaster.py"))
```

用返回的代码替换原单行 `_WEIGHTS` 赋值。它定义精确的 `_WEIGHTS` 和函数 `_dual_harbor_weights(_WEIGHTS)`。运行时只需要 Python 标准库和 NumPy。默认组合片段 SHA-256 为 `5a8dc5effa4ff02fe2792883d62f74e7c4150ec906c4c051151161575930ae86`。

```sh
python experiments/v7/dual_weights.py yan-v6.py experiments/v7/checkpoints/harbormaster.py ../v7-data/dual-weights-init.py
python experiments/v7/dual_weights.py yan-v6.py experiments/v7/checkpoints/harbormaster.py ../v7-data/dual-weights-both.py --pair
python experiments/v7/dual_weights_check.py ../v7-data
python experiments/v7/dual_weights_startup.py ../v7-data/dual-weights-both.py
```
