# 龙悟LM-X 推理指南

## 安装与环境

使用 Python 3.12。仓库通过 `uv.lock` 固定依赖解析结果：

```bash
uv sync --frozen --extra server
```

Linux 配置选择 PyTorch 2.9.0 / CUDA 12.8 wheel。Jetson 等平台需要对应 Python/CUDA 与
PyTorch wheel；请在目标设备上独立安装与验证。该安装配置不代表已经测量各平台的显存或性能。

开发环境使用 `uv sync --frozen --all-extras`。如果使用 pip，请自行安装平台对应的 PyTorch，
再执行 `python -m pip install -e '.[server]'`；pip 不读取 uv 的 CUDA index 配置与锁文件。

## Checkpoint 资源契约

模型目录需要 `config.json`，以及以下任一完整权重布局：

- `model.safetensors`
- `pytorch_model.bin`
- `model.safetensors.index.json` 与索引引用的全部分片
- `pytorch_model.bin.index.json` 与索引引用的全部分片

processor 的 `processor_config.json`、`statistics.json`、`embodiment_id.json` 应全部位于模型根目录，
或全部位于 `processor/` 子目录。根目录存在 `processor_config.json` 时，优先使用根目录布局。

```text
checkpoint/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
└── processor/
    ├── processor_config.json
    ├── statistics.json
    └── embodiment_id.json
```

本地清单检查会拒绝缺失/空文件、Git LFS 指针、非法 JSON 对象和不安全的分片引用。
该检查验证文件可读性与清单，不验证权重数值内容。模型加载后还会检查参数缺失、尺寸不匹配及
意外参数；独立加载的 backbone 参数和被 `select_layer` 裁掉的高层参数有明确兼容例外。

`model_path` 支持本地目录或 Hub model ID。`backbone_path` 覆盖 `config.json` 的 `model_name`。
推荐本地使用绝对路径；离线运行需要 checkpoint、backbone、tokenizer 与 processor 文件全部
已经存在于本地或缓存，并设置 `local_files_only=True` / `--local-files-only`。

访问受限资源时设置 `HF_TOKEN`，或先完成 Hugging Face 登录。Hub model ID 由加载后端解析；
本地目录的自定义清单预检不等于 Hub 资源或 backbone 经过同一套预检。

## 输入适配

`LMXPolicy.get_io_spec()` 与 `LMXClient.get_io_spec()` 返回相同的普通字典：

- `schema_version`：当前为 1。
- `embodiment_tag`：已解析的本体值。
- `modalities`：各模态的 `keys`、`delta_indices`、`dtype`，以及状态/动作的 `dimensions`。
- `example_image_size`：合成输入所用尺寸，不是禁止真实相机使用其他尺寸。

`make_sample_observation(spec)` 可用于快速验证；其图像与状态为零值。
`lm-x-smoke` 会优先从状态统计中的有限均值等信息构造代表性合成状态。
业务侧应按相同结构填写真实数据，并保持历史时间戳、多视角和状态采样对齐。

```python
from lm_x import LMXClient, make_sample_observation

with LMXClient(timeout_ms=60000) as client:
    spec = client.get_io_spec()
    observation = make_sample_observation(spec, batch_size=1)
    # 实际业务在这里替换每一路 video/state 和 language 数据。
    # 所有流必须保持相同的 batch；时间长度和状态维度遵循 spec。
    actions, info = client.get_action(observation)
```

`strict=True` 默认拒绝缺失或未知流、错误 batch、空数组、错误 dtype/rank/时间长度、
非 RGB 视频、错误状态/动作维度与非有限状态/动作值。检查在 `python -O` 下仍有效，
失败抛出 `InferenceValidationError`（`ValueError` 子类）。
`strict=False` 仅跳过 policy 外层检查，不关闭 processor 或模型内部约束。

为保持已有接口兼容，`get_modality_config()` 仍可使用；本地返回 `ModalityConfig` 对象，
经 ZeroMQ 传输后为字典。新接入代码优先使用两端格式一致且包含维度的 `get_io_spec()`。

### 多本体与辅助输出

初始化时可指定默认本体，也可传 `embodiment_tag=None`，随后每次调用显式选择：

```python
spec = policy.get_io_spec("CHECKPOINT_EMBODIMENT_TAG")
observation = make_sample_observation(spec)
actions, info = policy.get_action(
    observation, options={"embodiment_tag": "CHECKPOINT_EMBODIMENT_TAG"}
)
```

标签不是任意机器人名称：需被 `EmbodimentTag.resolve` 识别，并存在于该 checkpoint 的配置中。
按请求选择本体不表示该 checkpoint 已具备任意新机器人的泛化能力。

`value_pred`、`action_uncertainty`、`uncertainty_score` 仅在模型返回有效对应结果时加入。
当前开源运行时暂时不提供 keyframe（ETG）推理代码或相关输出字段。

## 服务部署

```bash
uv run --frozen --extra server lm-x-serve \
  --model-path /absolute/path/to/checkpoint \
  --backbone-path /absolute/path/to/backbone \
  --embodiment-tag CHECKPOINT_EMBODIMENT_TAG \
  --device cuda:0 --dtype bfloat16 --local-files-only \
  --host 127.0.0.1 --port 5555
```

服务使用 ZeroMQ REQ/REP，一次处理一个请求。协议支持 NumPy 数组，不使用 pickle 反序列化。
`LMXClient` 默认超时为 15 秒；耗时请求可通过 `timeout_ms=60000` 等设置调整。
超时会重建客户端 socket，后续请求可继续发送；已经送到服务端的推理不会被自动取消或重试。

| 客户端接口 | 作用 |
| --- | --- |
| `ping()` | 检查连接与服务响应 |
| `get_io_spec()` | 获取包含维度的可序列化契约 |
| `get_modality_config()` | 读取已有模态配置接口 |
| `get_action(observation, options)` | 生成动作 |
| `reset(options)` | 调用 policy 的 reset；当前实现校验本体后返回空字典 |
| `stop_server()` | 请求服务退出，所有持有有效 token 的客户端均可调用 |

服务端 `--api-token` 默认读取 `LMX_API_TOKEN`，客户端需传入同一个值。未设置时不鉴权；
显式设置空字符串会被拒绝。token 是共享密钥校验，不提供 TLS，适用于本机或受控私网。
不同机器运行时须通过受控网络或加密通道连接，并配置实际服务地址。

Python 内嵌服务提供 `stop()`，可从其他线程请求退出；`run()` 在当前请求完成后释放 socket。

## 验证与复现

```bash
uv build
uv run --frozen --all-extras pytest -q --timeout=60
```

构建后再测试可覆盖 wheel 内容。CPU 回归包括实际 processor 预处理、动作积分参考对比、
mock 模型加载及真实回环 ZeroMQ 请求；这些测试不下载正式模型权重。

真实模型验收：

```bash
uv run --frozen lm-x-smoke \
  --model-path /absolute/path/to/checkpoint \
  --backbone-path /absolute/path/to/backbone \
  --embodiment-tag CHECKPOINT_EMBODIMENT_TAG \
  --device cuda:0 --dtype bfloat16 --local-files-only --seed 0 --json
```

也可以显式启用 pytest 用例：

```bash
LMX_MODEL_PATH=/absolute/path/to/checkpoint \
LMX_BACKBONE_PATH=/absolute/path/to/backbone \
LMX_EMBODIMENT_TAG=CHECKPOINT_EMBODIMENT_TAG \
LMX_DEVICE=cuda:0 LMX_DTYPE=bfloat16 \
uv run --frozen --all-extras pytest tests/test_real_checkpoint_smoke.py -v
```

真实验收须看到 `PASSED`；环境变量缺失或请求的 GPU 不可用会导致 `SKIPPED`。
记录代码提交、checkpoint/backbone 版本、本体标签、设备、依赖与 dtype。固定随机种子便于
同环境复现，但不是跨硬件逐位一致承诺。

## 常见问题

| 现象 | 排查方向 |
| --- | --- |
| Git LFS pointer | 拉取或重新下载实际模型文件，不只是指针 |
| Missing processor metadata | 检查三个元数据文件是否位于同一受支持目录 |
| 不受支持的本体 | 检查 enum 名/value、模态配置和 `embodiment_id.json` |
| 维度不匹配 | 使用 `get_io_spec()` 核对每个状态/动作组，不使用全局固定维度 |
| NaN / infinity | 检查输入状态或模型输出；错误不会通过外层验证 |
| 无法获取远程资源 | 检查路径、完整缓存、访问权限及离线参数 |
| 请求超时 | 检查服务进程、端口与推理耗时，按需调大 `timeout_ms` |
| `invalid API token` | 确保服务和客户端使用同一个非空 token |
