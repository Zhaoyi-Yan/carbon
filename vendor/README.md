# Carbon 裁判

该目录保存 Carbon 对局结算所需的 zerosum_env 子集，上游版本是 [moliqingwa/carbon_challenge 的 3a24c67](https://github.com/moliqingwa/carbon_challenge/tree/3a24c67f0b3bb75e4c528fcbe8594f0e4bc29948)。游戏规则、地图生成、配置、辅助模块与该提交的 Git blob 逐一核对。许可证见 [LICENSE](LICENSE)。

本项目替换了顶层包入口，并为 envs 目录添加了空的包标记，方便在只有 NumPy 和 jsonschema 的环境中直接导入裁判。上游的网页服务与通用比赛注册入口不在此子集内。策略如需神经网络，另装对应的 CPU 推理依赖。
