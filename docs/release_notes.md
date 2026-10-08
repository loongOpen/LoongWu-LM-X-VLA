# 龙悟LM-X 推理运行时版本说明

## Action-only 推理整理 · 2026-09-20

- 当前开源运行时暂时不提供 keyframe（ETG）推理代码、配置项和输出字段。
- 保留 action、value 与 uncertainty 推理路径；checkpoint 的动作解码契约不变。
- ZeroMQ 服务兼容已有机器人客户端使用的 NumPy 消息格式。
- 重构位姿、动作片段、归一化、状态动作处理与流模型内部实现。

## 论文文档对齐 · 2026-09-11

- 按 [LM-X 论文 v3](https://arxiv.org/abs/2608.25757v3)更新模型定位、架构概览与引用。
- README 补充可解释信号与论文结果，明确评测协议、论文版本和结果来源。
- 模型卡区分研究模型、运行时实现与 checkpoint 验证；新增[论文对齐说明](paper_alignment.md)。
- 记录 RTG→ETG、方差传播和默认配置差异，以及实机汇总数据的待核对项。
- 本次为文档更新；未更改网络连接、权重加载行为、推理公式或运行时版本号，未加入训练代码。

## 0.2.0 · 2026-09-11

本次更新完善从模型加载到本地/远程动作调用的工程链路，并将项目首页、模型卡、推理指南分别组织。

### 代码改进

- 修复默认语言预处理缺少 `re` 导入导致的运行错误。
- 复用 processor、tokenizer 与 collator；直接使用模型时按需初始化 collator。
- 将动作位置嵌入移出迭代循环，去掉共享 timestep 路径每步的 `.item()`。
- 提供 NumPy 输入/输出校验，覆盖批大小、时间长度、维度、dtype 和非有限值，在 `python -O` 下仍有效。
- 增加两端一致的 `get_io_spec()` 与 `make_sample_observation()`，可直接运行合成推理示例。
- 增加可安装的 `lm-x-smoke` 命令，支持固定 seed 与 JSON 结果；保留旧脚本入口。
- 加强 checkpoint JSON 与跨平台分片路径检查。
- 完善服务资源释放、空闲退出和共享 token 比较；增加真实 ZeroMQ 回环与超时恢复测试。
- 增加依赖锁文件、预提交检查和 CPU CI 配置。

这些优化减少了重复加载、重复计算和一处逐步设备同步。实际时延、吞吐与显存改善尚需在目标 GPU
和真实 checkpoint 上测量，本次不提供性能提升百分比。

### 兼容性

`LMXPolicy.get_action()` 和服务端现有请求保持原有输入输出结构。
新增严格校验会提前拒绝原先可能进入底层计算的多余流、空 batch、错误维度和 NaN/Inf。
外层异常类型为 `InferenceValidationError`，继承 `ValueError`；依赖原 `AssertionError` 的调用方
应更新捕获逻辑。原有 `get_modality_config()` 保留，新代码推荐使用 `get_io_spec()`。

### 项目介绍建议

龙悟LM-X 面向可解释机器人操作，围绕任务进展、下一语义事件与局部动作可靠性，
构建视觉—语言—动作模型的预测接口。模型研究以论文为依据，部署能力以具体 checkpoint
及推理实现为准。

本次代码交付聚焦模型加载与推理接入，提供标准化 Python API、远程推理服务、输入输出规格以及
可复现的链路验证入口，便于开发者将兼容 checkpoint 接入已有机器人软件系统。具体权重、适配本体
及任务效果以对应模型资源和验证报告为准；模型权重与训练数据的发布状态独立于推理代码。

## 文档组织参考

以下资料核对于 2026-09-11。借鉴的是信息组织方式；龙悟LM-X 的研究描述以论文为依据，
运行时接口与验证状态以本仓库实现为依据。

| 官方来源 | 采用的组织方式 | 在本项目中的落点 |
| --- | --- | --- |
| [Physical Intelligence · openpi](https://github.com/Physical-Intelligence/openpi) | 按模型/checkpoint 用途组织资源，并提供本地、远程和无机器人推理入口 | README 的发布内容、最短推理路径和服务示例 |
| [AgiBot · GO-1](https://github.com/OpenDriveLab/AgiBot-World#how-to-get-started-with-our-go-1-model) | 连接模型资源、归一化统计、观测输入和部署调用 | checkpoint 资源契约与输入适配文档 |
| [Galaxea · GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) | 先解释模型设计，再按 checkpoint、本体和运行环境组织使用入口 | 模型概览、模型卡及适用范围 |

这些项目中的数据规模、参数量、效果指标和适配机器人不属于龙悟LM-X的验证结果。
