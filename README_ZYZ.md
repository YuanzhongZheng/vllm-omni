# vLLM-Omni 仓库梳理

> 作者：ZYZ | 整理日期：2026-05-20

---

## 一、项目定位

**vLLM-Omni** 是 vLLM 的多模态扩展框架，核心目标是在单一推理服务栈中同时支持：

- **多模态输入**：文本、图像、视频、音频
- **多模态输出**：文本、音频（TTS）、图像（DiT）、视频（DiT）
- **异构执行**：自回归（AR）模型 + 扩散变换器（DiT）混合推理
- **多阶段流水线**：将复杂模型拆分为多个 Stage，跨 GPU/节点并行执行

当前版本 **v0.20.0**（2026 年 5 月），对应 vLLM v0.21.0。

---

## 二、仓库结构

```
vllm-omni/
├── vllm_omni/                  # 核心 Python 包
│   ├── entrypoints/            # 用户入口（Omni、AsyncOmni、OpenAI 兼容服务）
│   ├── engine/                 # 引擎编排（AsyncOmniEngine、Orchestrator、StagePool）
│   ├── model_executor/         # 模型加载与执行（20+ AR 模型实现）
│   │   ├── models/             # 具体模型（qwen3_omni、qwen3_tts、wan、ltx …）
│   │   └── stage_configs/      # 模型 Stage 拆分 YAML（旧版，已逐步迁移）
│   ├── diffusion/              # DiT 非自回归引擎
│   │   ├── models/             # 30+ 扩散模型实现
│   │   ├── distributed/        # TP/SP/USP/HSDP/CFG-Parallel
│   │   └── cache/              # Cache-DiT 集成（DBCache、TeaCache）
│   ├── distributed/            # 跨 Stage 通信（SharedMemory / Mooncake）
│   ├── config/                 # 配置解析（StageConfig、ModelConfig）
│   ├── deploy/                 # 内置部署 YAML（生产级配置模板）
│   ├── quantization/           # 量化支持（FP8、AWQ）
│   ├── lora/                   # LoRA 适配
│   └── platforms/              # 硬件后端（cuda/rocm/npu/xpu/musa）
├── examples/
│   ├── offline_inference/      # 离线推理示例脚本
│   └── online_serving/         # 在线服务示例（curl / Gradio / OpenAI Client）
├── docs/                       # 文档（MkDocs 构建）
├── tests/                      # 测试套件（unit / integration / e2e / benchmark）
├── docker/                     # 多平台 Dockerfile
├── requirements/               # 平台分类依赖文件
├── benchmarks/                 # 性能基准测试
├── apps/ComfyUI-vLLM-Omni/     # ComfyUI 插件集成
├── recipes/                    # 社区模型集成 recipes
├── pyproject.toml              # 项目元数据
└── setup.py                    # 平台感知动态安装脚本
```

### 关键模块说明

| 模块 | 职责 |
|------|------|
| `engine/orchestrator.py` | 请求路由，跨 Stage 调度 |
| `engine/async_omni_engine.py` | 异步推理引擎主体 |
| `diffusion/diffusion_engine.py` | DiT 扩散步执行引擎 |
| `distributed/omni_connectors/` | Stage 间 KV 数据传输（共享内存 / 网络） |
| `config/stage_config.py` | Stage 配置解析（设备分配、并行度、内存比例等） |
| `entrypoints/openai/` | FastAPI 实现的 OpenAI 兼容 API |

---

## 三、支持模型（70+）

### 全模态大模型（AR + DiT）
| 模型 | 能力 |
|------|------|
| Qwen3-Omni | 文本输入 → 文本 / 语音输出 |
| Qwen2.5-Omni | 文本 / 图像 / 音频输入 → 文本 / 语音输出 |
| MiMo-Audio | 音频理解与生成 |

### 图像生成（DiT）
Qwen-Image、Z-Image-Turbo、HunyuanImage-3.0、GLM-Image、ERNIEImage、FLUX.1/2、Stable-Diffusion-3、BAGEL、OmniGen2 等

### 视频生成（DiT）
Wan2.1 / 2.2（T2V、I2V、S2V、VACE）、LTX-2、HunyuanVideo-1.5、NextStep-1.1 等

### 语音合成（TTS）
Qwen3-TTS、Fish Speech、CosyVoice3、Voxtral-TTS、StableAudio、AudioX 等

### 硬件支持矩阵
| 模型类型 | CUDA | ROCm | Ascend NPU | Intel XPU | MUSA |
|----------|:----:|:----:|:----------:|:---------:|:----:|
| AR 模型 | ✅ | ✅ | ✅ | ✅ | ✅ |
| DiT 图像 | ✅ | ✅ | ✅ | - | - |
| DiT 视频 | ✅ | ✅ | ✅ | - | - |
| TTS | ✅ | - | ✅ | - | - |

---

## 四、依赖环境

### Python 版本
**Python 3.10 – 3.13**（推荐 3.12）

### 核心依赖（通用）

| 包 | 版本要求 | 用途 |
|----|---------|------|
| vllm | ==0.21.0 | 底层推理引擎（必须版本匹配） |
| diffusers | ==0.38.0 | HuggingFace 扩散模型库 |
| accelerate | ==1.12.0 | 分布式加速 |
| cache-dit | ==1.3.0 | DiT 推理缓存加速（Cache-DiT） |
| av | >=14.0.0 | 音视频编解码 |
| safetensors | >=0.8.0 | 安全权重序列化 |
| soundfile | >=0.13.1 | 音频 I/O |
| x-transformers | >=2.12.2 | 扩展 Transformer 架构 |
| openai-whisper | >=20250625 | 语音识别（ASR） |
| omegaconf | >=2.3.0 | 配置管理 |

### 平台专属依赖

```
requirements/
├── common.txt     # 所有平台通用
├── cuda.txt       # onnxruntime>=1.23.2, fa3-fwd==0.0.3
├── rocm.txt       # onnxruntime-rocm>=1.22.2
├── npu.txt        # onnxruntime-cann>=1.23.2, torchaudio==2.9.0
├── xpu.txt        # onnxruntime>=1.23.2
└── musa.txt       # torchada, mate, flash_attn_3>=0.1.4
```

> `setup.py` 会自动检测硬件平台并安装对应依赖，无需手动选择。

### Docker 镜像
```
docker/
├── Dockerfile.cuda          # NVIDIA CUDA
├── Dockerfile.rocm          # AMD ROCm
├── Dockerfile.npu           # 昇腾 NPU（基础版）
├── Dockerfile.npu.a3        # 昇腾 NPU（A3 系列）
├── Dockerfile.xpu           # Intel XPU
└── Dockerfile.musa          # 摩尔线程 MUSA
```

---

## 五、快速 POC

### 5.1 环境安装

```bash
# 1. 创建虚拟环境（推荐 uv）
uv venv --python 3.12 --seed
source .venv/bin/activate

# 2. 安装 vLLM 基础（版本必须匹配）
uv pip install vllm==0.21.0 --torch-backend=auto   # CUDA
# ROCm 用户：uv pip install vllm==0.21.0+rocm721

# 3. 克隆并安装 vllm-omni
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

---

### 5.2 POC 场景一：文本生成图像（离线推理）

```python
from vllm_omni.entrypoints.omni import Omni

# 加载模型（首次运行会自动下载权重）
omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")

# 单条生成
outputs = omni.generate("a cup of coffee on a wooden table")
outputs[0].request_output.images[0].save("coffee.png")

# 批量生成
prompts = [
    "a futuristic city at night with neon lights",
    "a panda eating bamboo in a forest",
]
for i, out in enumerate(omni.generate(prompts)):
    out.request_output.images[0].save(f"output_{i}.jpg")
```

---

### 5.3 POC 场景二：启动 OpenAI 兼容服务（在线服务）

**单卡启动（适合 7B 量级模型）：**

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091
```

**多 Stage 启动（适合 Qwen3-Omni 等复杂模型，需要多 GPU）：**

```bash
# Stage 0：Thinker + API Server（GPU 0）
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --port 8091 --stage-id 0 \
    --omni-master-address 127.0.0.1 --omni-master-port 26000

# Stage 1：Talker（GPU 1）
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 1 --headless \
    --omni-master-address 127.0.0.1 --omni-master-port 26000

# Stage 2：Code2Wav（GPU 1，可与 Stage 1 共卡）
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 2 --headless \
    --omni-master-address 127.0.0.1 --omni-master-port 26000
```

---

### 5.4 POC 场景三：curl 调用（文本 → 图像）

```bash
# 服务启动后，发送生图请求
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "a red sports car on a mountain road"}],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 50,
      "guidance_scale": 4.0,
      "seed": 42
    }
  }' | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
img_b64 = data['choices'][0]['message']['content'][0]['image_url']['url'].split(',')[1]
open('result.png','wb').write(base64.b64decode(img_b64))
print('Saved result.png')
"
```

---

### 5.5 POC 场景四：Gradio Web Demo（一键启动）

```bash
cd examples/online_serving/qwen3_omni
./run_gradio_demo.sh \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --server-port 8091 \
    --gradio-port 7861

# 浏览器打开 http://localhost:7861/
```

---

### 5.6 POC 场景五：视频生成（Wan2.1）

```bash
# 启动文生视频服务
vllm serve Wan-AI/Wan2.1-T2V-14B-Diffusers --omni --port 8091

# 发送请求
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "a cat playing piano"}],
    "extra_body": {
      "height": 480, "width": 832,
      "num_frames": 81,
      "num_inference_steps": 30,
      "guidance_scale": 5.0
    }
  }' | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
vid = data['choices'][0]['message']['content'][0]['video_url']['url'].split(',')[1]
open('output.mp4','wb').write(base64.b64decode(vid))
print('Saved output.mp4')
"
```

---

## 六、常用配置调优

### 内存优化（多 Stage 场景）

```bash
vllm serve <model> --omni --port 8091 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --stage-overrides '{"1": {"gpu_memory_utilization": 0.5}}'
```

### 使用内置 YAML 部署配置

```bash
# 内置配置位于 vllm_omni/deploy/ 目录
vllm serve <model> --omni --port 8091 \
    --deploy-config vllm_omni/deploy/text_to_image.yaml
```

### YAML 配置结构示例

```yaml
async_chunk: true                    # 开启 Stage 间异步流式传输

connectors:
  shm_connector:
    name: SharedMemoryConnector      # 同机 KV 传输

stages:
  - stage_id: 0
    devices: "0"
    max_num_batched_tokens: 32768
    gpu_memory_utilization: 0.9

  - stage_id: 1
    devices: "1"
    gpu_memory_utilization: 0.6
    input_connectors:
      from_stage_0: shm_connector
```

---

## 七、API 端点速查

| 端点 | 协议 | 用途 |
|------|------|------|
| `POST /v1/chat/completions` | HTTP | 文本/多模态对话（AR + DiT） |
| `POST /v1/images/generations` | HTTP | 纯文生图（DALL-E 兼容） |
| `WS /v1/realtime` | WebSocket | 实时音视频流式交互 |

---

## 八、参考目录

| 资源 | 路径 |
|------|------|
| 离线推理示例 | `examples/offline_inference/` |
| 在线服务示例 | `examples/online_serving/` |
| 内置部署配置 | `vllm_omni/deploy/` |
| 文档站点配置 | `mkdocs.yml` |
| 测试套件 | `tests/` |
| ComfyUI 集成 | `apps/ComfyUI-vLLM-Omni/` |
