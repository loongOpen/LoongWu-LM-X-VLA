# LoongWu-LM-X-VLA

这些代码是论文 **[LM-X: Explainable Vision--Language--Action Modeling via Progress, Event, and Uncertainty Prediction](https://arxiv.org/abs/2608.25757)** 的开源版本。

[English](README.md) | **简体中文**

[论文 v3](https://arxiv.org/abs/2608.25757v3) · [GitHub](https://github.com/loongOpen/LoongWu-LM-X-VLA) · [AtomGit](https://atomgit.com/openloong/LoongWu-LM-X-VLA) · [模型（HF）](https://huggingface.co/OpenLoong/LoongWu-LM-X-VLA-v1) · [模型（AtomGit）](https://ai.atomgit.com/openloong/LoongWu-LM-X-VLA-v1)

LM-X 是一个面向机器人操作的视觉—语言—动作（VLA）模型。它接收多视角图像、语言指令和机器人状态，生成连续动作片段。论文同时研究三个预测信号：任务进展（RTG）、下一语义事件（ETG）和动作不确定性。

本仓库提供推理代码，不包含训练代码或训练数据。模型下载地址与使用方法见下文。

## 模型概览

![LM-X 论文架构：RTG、ETG、动作生成与不确定性预测](docs/assets/paper-v3/figure-1-overview.png)

*图 1：论文中的 LM-X 架构。原图来自 [LM-X v3，Figure 1](https://arxiv.org/html/2608.25757v3#S1.F1)。*

视觉语言基座使用 **NVIDIA Cosmos-Reason2-2B（Qwen3-VL 架构）**，编码图像与任务指令。机器人状态直接送入事件和动作专家。

论文中的三个信号分别表示：

- **RTG（Return-to-Go）**：当前状态下的任务进展与状态质量。
- **ETG（Event-to-Go）**：通向下一语义事件的动作片段，例如抓取、对齐或插入。
- **动作不确定性**：动作流预测的方差，用于分析局部动作可靠性。

当前代码提供动作推理，并按 checkpoint 配置启用 `value` 和 `uncertainty` 分支。**ETG 推理代码尚未发布。** 图 1 展示的是论文完整架构；当前不确定性实现与论文的差异见[论文对齐说明](docs/paper_alignment.md)。

## 论文结果

以下为论文 v3 报告的结果，均包含任务适配，不是本仓库的测试结果。

| 评测 | 平均成功率 | 设置 |
| --- | --- | --- |
| RoboTwin2.0 | **74.1%** | 50 个 randomized-hard 任务；每任务 50 条适配示范、100 次测试 |
| 实机操作 | **73.5%**¹ | 4 种机器人本体、7 项任务；每任务 20 次测试 |

来源：[论文 §4.3](https://arxiv.org/html/2608.25757v3#S4.SS3)、[§4.4](https://arxiv.org/html/2608.25757v3#S4.SS4)。¹ 实机汇总值按原文保留，表内复算说明见[数值核对](docs/paper_alignment.md#论文数值核对)。

![Astribot S1 精密插装任务：拾取、插入、位姿微调与完成装配](docs/assets/paper-v3/figure-2-precision-insertion.png)

*图 2：Astribot S1 精密插装任务，从拾取零件到完成装配。原图来自 [LM-X v3，Figure 2](https://arxiv.org/html/2608.25757v3#S2.F2)。*

![论文中的动作不确定性与失败预警分析](docs/assets/paper-v3/figure-7-uncertainty.png)

*图 7：精密插装中的动作不确定性曲线及其梯度告警。原图来自 [LM-X v3，Figure 7](https://arxiv.org/html/2608.25757v3#S4.F7)；对应实验说明见[论文 §4.7](https://arxiv.org/html/2608.25757v3#S4.SS7)。*

## 发布内容

| 内容 | 状态 |
| --- | --- |
| 推理代码 | 模型加载、图像与状态预处理、动作生成、Python API 和 ZeroMQ 服务 |
| LM-X v1 checkpoint | [Hugging Face](https://huggingface.co/OpenLoong/LoongWu-LM-X-VLA-v1) / [AtomGit](https://ai.atomgit.com/openloong/LoongWu-LM-X-VLA-v1)；通过 `--model-path` 指定 |
| Cosmos-Reason2-2B | [NVIDIA 官方下载](https://huggingface.co/nvidia/Cosmos-Reason2-2B)；单独准备权重、tokenizer 和 processor，通过 `--backbone-path` 指定 |
| 训练代码与训练数据 | 不在本次发布范围内 |

## 快速开始

### 安装

使用 Python **3.12** 和 [uv](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/loongOpen/LoongWu-LM-X-VLA.git
cd LoongWu-LM-X-VLA
uv sync --frozen --extra server
```

Linux 默认依赖为 PyTorch 2.9.0 / CUDA 12.8。其他环境的配置见[推理指南](docs/inference.md#安装与环境)。

### 准备两份模型资源

运行模型推理需要 **LM-X checkpoint** 和 **`nvidia/Cosmos-Reason2-2B`**。checkpoint 中的配置、权重和归一化统计需要配套；Cosmos 目录还需包含 tokenizer 与 processor 文件。文件布局和分片支持见[资源说明](docs/inference.md#checkpoint-资源契约)。

LM-X 从上方 Hugging Face 或 AtomGit 任选一处下载即可。Cosmos 使用 [GR00T 官方安装说明](https://github.com/NVIDIA/Isaac-GR00T#installation)中提供的 [NVIDIA 模型地址](https://huggingface.co/nvidia/Cosmos-Reason2-2B)。先在 Cosmos 模型页面获取访问权限，再使用同一 Hugging Face 账号登录。

安装完成后，在仓库根目录执行：

```bash
uv run --frozen hf auth login
uv run --frozen hf download OpenLoong/LoongWu-LM-X-VLA-v1 \
  --local-dir ./checkpoints/LoongWu-LM-X-VLA-v1
uv run --frozen hf download nvidia/Cosmos-Reason2-2B \
  --local-dir ./checkpoints/Cosmos-Reason2-2B
```

若从 AtomGit 下载 LM-X，将完整模型目录保存到 `./checkpoints/LoongWu-LM-X-VLA-v1`，即可跳过第一条下载命令。

### 运行一次推理

下方示例使用上述下载目录。将 `CHECKPOINT_EMBODIMENT_TAG` 替换为 checkpoint 支持的本体标签；如果模型保存在其他位置，请相应修改路径。

```bash
uv run --frozen lm-x-smoke \
  --model-path ./checkpoints/LoongWu-LM-X-VLA-v1 \
  --backbone-path ./checkpoints/Cosmos-Reason2-2B \
  --embodiment-tag CHECKPOINT_EMBODIMENT_TAG \
  --device cuda:0 \
  --dtype bfloat16 \
  --local-files-only \
  --seed 0
```

程序使用合成观测检查模型加载、输入输出形状和数值是否正常，成功时输出 `Smoke inference passed`。这一步不评估任务成功率。

### Python 调用

```python
from lm_x import LMXPolicy, make_sample_observation

policy = LMXPolicy(
    model_path="./checkpoints/LoongWu-LM-X-VLA-v1",
    backbone_path="./checkpoints/Cosmos-Reason2-2B",
    embodiment_tag="CHECKPOINT_EMBODIMENT_TAG",
    device="cuda:0",
    local_files_only=True,
)

spec = policy.get_io_spec()
observation = make_sample_observation(spec, instruction="move to the target")
actions, info = policy.get_action(observation)
```

接入机器人或评测环境时，用实际相机、状态和指令替换合成观测，并按 `spec` 对齐字段和采样时序。

## 文档

- [推理指南](docs/inference.md)：环境安装、输入输出、服务端与客户端、验证方式。
- [模型卡](docs/model_card.md)：模型组成、资源要求与适用范围。
- [论文对齐说明](docs/paper_alignment.md)：当前实现与论文模型的对应关系。
- [版本说明](docs/release_notes.md)。

## 许可与来源

代码采用 [Apache-2.0](LICENSE)，保留 NVIDIA 上游文件的版权及修改声明。模型权重的许可由各自发布方说明。

本页三张图片均为 Jin Lou 等作者的 LM-X v3 论文原图，按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 引用，未作修改。原始文件地址与校验值见[图片来源](docs/assets/paper-v3/README.md)。

## 引用

如果本项目对你的研究有帮助，请引用 [LM-X 论文 v3](https://arxiv.org/abs/2608.25757v3)。

```bibtex
@misc{lou2026lmx,
  title = {LM-X: Explainable Vision--Language--Action Modeling via Progress, Event, and Uncertainty Prediction},
  author = {Jin Lou and Zhiyuan Jing and Xupeng Wang and Andong Chen and Xingdong Zhu and Yuexuan Li and Yuan Xu and Zhijie Zhu and Yingwei Ji and Wenpeng Nie and Renxing Feng and Liangliang Chen and Ying Chu and Jingyi Li and Jinyan Liu and Zhiqi Song and Jingxuan Zhu and Jidong Zhang and Yufei Liu and Boyang Xing and Lei Jiang and Yan Cui and Hongming Li and Yuchen Zhu},
  year = {2026},
  eprint = {2608.25757},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  note = {Version 3, 2026-09-08},
  url = {https://arxiv.org/abs/2608.25757v3}
}
```
