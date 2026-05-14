# index-tts-fastapi

基于官方 [IndexTTS2](https://github.com/index-tts/index-tts) 二次开发的 FastAPI 服务封装，用来把 IndexTTS2 部署成可被业务应用调用的 TTS 基础能力。

这个仓库保留了上游 IndexTTS2 的推理代码和模型加载逻辑，主要新增的是生产友好的 API server：

- 异步任务接口，避免 API 网关 30s 超时影响长文本生成
- 任务进度查询和 SSE 实时进度推送
- `voice_id` 参考音频资产模式，按 sha256 去重，避免重复上传和重复存储
- 单 worker 队列串行推理，避免 IndexTTS2 内部缓存被并发请求污染
- API Key 鉴权、CORS、健康检查、术语表管理
- 支持 multipart 上传和 JSON/base64 调用

更完整的接口示例见 [docs/API_SERVER_ZH.md](docs/API_SERVER_ZH.md)。

## 与上游的关系

本项目不是官方 IndexTTS2 仓库。模型、推理能力、许可证和使用限制请以官方项目为准：

- 官方仓库：[index-tts/index-tts](https://github.com/index-tts/index-tts)
- 模型地址：[HuggingFace IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2)
- ModelScope：[IndexTTS-2](https://modelscope.cn/models/IndexTeam/IndexTTS-2)

商业使用、模型授权、语音克隆合规等问题，请自行确认上游许可证和相关法律要求。

## 安装

建议使用 `uv`：

```bash
uv sync --extra api
```

如果还需要保留 WebUI 或 DeepSpeed：

```bash
uv sync --extra api --extra webui
uv sync --extra api --extra deepspeed
```

## 准备模型

默认模型目录为 `checkpoints/`。至少需要：

- `bpe.model`
- `gpt.pth`
- `config.yaml`
- `s2mel.pth`
- `wav2vec2bert_stats.pt`

从 HuggingFace 下载：

```bash
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
```

或从 ModelScope 下载：

```bash
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

首次运行时，IndexTTS2 还会自动拉取若干小模型或配置，例如 `facebook/w2v-bert-2.0`、`amphion/MaskGCT`、`funasr/campplus`。如果服务器访问 HuggingFace 较慢，可以配置镜像：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

## 启动 API 服务

默认启动不会加载 IndexTTS2 模型，也不会在 startup 阶段访问 HuggingFace。第一次生成、分句预览、配置查询或术语表操作等需要模型的请求到来时，服务才会加载模型；如果辅助依赖缓存缺失，也会在那时下载。

Mac：

```bash
uv run indextts-api --host 0.0.0.0 --port 7861 --model_dir checkpoints --device mps
```

NVIDIA 服务器：

```bash
uv run indextts-api --host 0.0.0.0 --port 7861 --model_dir checkpoints --device cuda:0 --fp16
```

也可以直接运行根目录入口：

```bash
uv run api_server.py --host 0.0.0.0 --port 7861 --model_dir checkpoints
```

精度行为：

- For CUDA / XPU：`--fp16` 会生效
- For CPU：强制关闭 fp16
- For Mac MPS：也强制关闭 fp16

常用环境变量：

```bash
export INDEXTTS_API_KEY="change-me"
export INDEXTTS_MODEL_DIR="/opt/index-tts-fastapi/checkpoints"
export INDEXTTS_OUTPUT_DIR="/data/indextts-outputs"
export INDEXTTS_FP16=true
export INDEXTTS_MAX_QUEUE_SIZE=100
uv run indextts-api
```

如果希望服务启动时就预热模型，可以显式使用：

```bash
uv run indextts-api --host 0.0.0.0 --port 7861 --model_dir checkpoints --device cuda:0 --fp16 --eager_load
```

## API 调用流程

生产推荐流程是先上传参考音频资产，再提交 TTS 任务。

### 1. 上传参考音频，获得 voice_id

```bash
curl -X POST http://127.0.0.1:7861/v1/voices \
  -H "Authorization: Bearer change-me" \
  -F "audio=@examples/voice_01.wav"
```

同一段音频会按 sha256 去重，重复上传会返回相同的 `voice_id`，不会重复保存文件。

### 2. 提交 TTS 任务

```bash
curl -X POST http://127.0.0.1:7861/v1/tts/jobs \
  -H "Authorization: Bearer change-me" \
  -F "text=欢迎使用 IndexTTS2 API 服务。" \
  -F "prompt_voice_id=<voice_id>" \
  -F "emotion_mode=speaker"
```

接口立即返回 `task_id`，不会等待音频生成完成。

### 3. 查询任务状态

```bash
curl -H "Authorization: Bearer change-me" \
  http://127.0.0.1:7861/v1/tasks/<task_id>
```

任务状态包括：

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

进度字段：

- `progress`：`0.0` 到 `1.0`
- `message`：当前阶段说明
- `queue_position`：排队位置

`progress` 会包含排队、模型加载、GPT 生成、mel diffusion step 和保存音频等阶段。

### 4. SSE 实时进度

```bash
curl -N -H "Authorization: Bearer change-me" \
  http://127.0.0.1:7861/v1/tasks/<task_id>/events
```

### 5. 下载生成音频

```bash
curl -H "Authorization: Bearer change-me" \
  http://127.0.0.1:7861/v1/tasks/<task_id>/audio \
  --output out.wav
```

## 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `GET` | `/v1/config` | 查看服务配置 |
| `POST` | `/v1/voices` | multipart 上传参考音频资产 |
| `POST` | `/v1/voices/json` | JSON/base64 上传参考音频资产 |
| `GET` | `/v1/voices` | 查看参考音频资产列表 |
| `GET` | `/v1/voices/{voice_id}` | 查看参考音频资产 |
| `DELETE` | `/v1/voices/{voice_id}` | 删除参考音频资产 |
| `POST` | `/v1/tts/jobs` | multipart 提交 TTS 任务 |
| `POST` | `/v1/tts/jobs/json` | JSON/base64 提交 TTS 任务 |
| `GET` | `/v1/tasks` | 查看最近任务 |
| `GET` | `/v1/tasks/{task_id}` | 查看任务状态 |
| `GET` | `/v1/tasks/{task_id}/events` | SSE 实时进度 |
| `GET` | `/v1/tasks/{task_id}/audio` | 下载生成音频 |
| `DELETE` | `/v1/tasks/{task_id}` | 删除任务 |
| `POST` | `/v1/segments/preview` | 预览分句 |
| `GET` | `/v1/glossary` | 查看术语表 |
| `PUT` | `/v1/glossary` | 新增或更新术语 |
| `DELETE` | `/v1/glossary/{term}` | 删除术语 |

## 输出目录

默认输出目录为 `outputs/api`：

- `outputs/api/voices/`：参考音频资产，按 sha256 去重
- `outputs/api/voices/voices.json`：voice asset 元数据
- `outputs/api/results/`：生成的 wav 文件

建议定期清理旧任务结果。`voice` 资产建议通过 API 显式删除，避免误删业务仍在使用的音色。

## 生产部署建议

- 公网部署务必设置 `INDEXTTS_API_KEY`，并放在 HTTPS 后面
- 单张 GPU 建议一个服务进程独占；服务内部已经串行推理
- 多 GPU 可以启动多个进程，再由网关分发
- API 网关可以保持 30s 甚至更短超时，因为 TTS 生成已经改成异步任务
- 如果前端直接订阅 SSE，原生 `EventSource` 不方便带 `Authorization` header，可使用 `fetch-event-source` 或由后端代理订阅

## 开发验证

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache .venv/bin/python -m py_compile indextts/api_server.py api_server.py
uv run --extra api --extra webui ruff check indextts/api_server.py api_server.py
git diff --check
```

## 上游 WebUI

上游的 `webui.py` 仍保留，可用于本地调试模型效果：

```bash
uv sync --extra webui
uv run webui.py
```
