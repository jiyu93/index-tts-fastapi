# IndexTTS2 FastAPI 服务部署说明

这个 API 服务会在进程启动时加载一次 IndexTTS2 模型，并通过后台任务队列串行执行推理，避免并发请求互相污染模型内部的参考音频缓存。生成语音接口是异步任务形态：提交任务立即返回 `task_id`，客户端通过轮询或 SSE 查看进度，完成后再下载音频。参考音频支持 `voice_id` 资产模式，服务端按 sha256 去重，同一段音频只保存一份。这样适合放在有 30s 超时限制的生产 API 网关后面。

## 安装

```bash
uv sync --extra api
```

如果你也要保留 WebUI 或 DeepSpeed：

```bash
uv sync --extra api --extra webui
uv sync --extra api --extra deepspeed
```

模型文件默认放在 `checkpoints/`，至少需要：

- `bpe.model`
- `gpt.pth`
- `config.yaml`
- `s2mel.pth`
- `wav2vec2bert_stats.pt`

首次运行时，IndexTTS2 还会自动从 HuggingFace 拉取几个小模型或配置，例如 `facebook/w2v-bert-2.0`、`amphion/MaskGCT` 和 `funasr/campplus`。服务器无法直连 HuggingFace 时，建议先在有网络的环境预热缓存，或配置 `HF_ENDPOINT` 镜像。

## 启动

默认启动不会加载 IndexTTS2 模型，也不会在 startup 阶段访问 HuggingFace。第一次生成、分句预览、配置查询或术语表操作等需要模型的请求到来时，服务才会加载模型；如果辅助依赖缓存缺失，也会在那时下载。

精度行为：

- For CUDA / XPU：`--fp16` 会生效。
- For CPU：强制关闭 fp16。
- For Mac MPS：也强制关闭 fp16。

推荐启动命令：

Mac 用：

```bash
uv run indextts-api --host 0.0.0.0 --port 7861 --model_dir checkpoints --device mps
```

NVIDIA 服务器用：

```bash
uv run indextts-api --host 0.0.0.0 --port 7861 --model_dir checkpoints --device cuda:0 --fp16
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--api_key` | 开启 API Key 鉴权。客户端使用 `Authorization: Bearer <key>` 或 `X-API-Key: <key>` |
| `--output_dir` | 输出目录，默认 `outputs/api` |
| `--fp16` | CUDA/XPU 上使用半精度推理 |
| `--deepspeed` | 尝试启用 DeepSpeed |
| `--cuda_kernel` | 尝试启用 BigVGAN CUDA kernel |
| `--device` | 指定 `cuda:0`、`cpu`、`mps`、`xpu` 等 |
| `--lazy_load` | 默认行为：启动 API 时不加载模型，首个需要模型的请求到来时再加载 |
| `--eager_load` | 启动 API 时立即加载模型，用于预热服务 |
| `--allow_local_paths` | 允许请求传服务器本地音频路径。默认关闭，推荐用上传或 base64 |
| `--max_queue_size` | 最大排队任务数，默认 `100`；队列满时返回 `429` |
| `--cors_origins` | 逗号分隔的跨域白名单，如 `https://app.example.com,http://localhost:3000` |

环境变量同名可用，例如：

```bash
export INDEXTTS_API_KEY="change-me"
export INDEXTTS_MODEL_DIR="/opt/indextts/checkpoints"
export INDEXTTS_OUTPUT_DIR="/data/indextts-outputs"
export INDEXTTS_FP16=true
uv run indextts-api
```

## 健康检查

```bash
curl http://127.0.0.1:7861/health
```

带鉴权的服务调用：

```bash
curl -H "Authorization: Bearer change-me" http://127.0.0.1:7861/v1/config
```

## 上传参考音频资产

推荐先把常用参考音频上传成 voice asset，拿到稳定的 `voice_id`，后续 TTS 任务只传 `voice_id`，不再重复上传音频。

```bash
curl -X POST http://127.0.0.1:7861/v1/voices \
  -H "Authorization: Bearer change-me" \
  -F "audio=@examples/voice_01.wav"
```

响应示例：

```json
{
  "voice_id": "d2a2f5a3b1...",
  "sha256": "d2a2f5a3b1...",
  "original_filename": "voice_01.wav",
  "path": "/data/indextts-outputs/voices/d2a2f5a3b1....wav",
  "content_type": "audio/wav",
  "size_bytes": 123456,
  "created_at": 1778460000.0,
  "last_used_at": null,
  "use_count": 0
}
```

同一段音频重复上传时会返回相同的 `voice_id`，不会重复保存文件。也可以用 JSON/base64 上传：

```bash
VOICE_B64=$(base64 -i examples/voice_01.wav)

curl -X POST http://127.0.0.1:7861/v1/voices/json \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d "{
    \"audio_base64\": \"$VOICE_B64\",
    \"filename\": \"voice_01.wav\",
    \"content_type\": \"audio/wav\"
  }"
```

查看或删除 voice asset：

```bash
curl -H "Authorization: Bearer change-me" http://127.0.0.1:7861/v1/voices
curl -H "Authorization: Bearer change-me" http://127.0.0.1:7861/v1/voices/<voice_id>
curl -X DELETE -H "Authorization: Bearer change-me" http://127.0.0.1:7861/v1/voices/<voice_id>
```

## 生成语音：multipart 上传任务

使用 `prompt_voice_id` 提交任务，立即返回 `task_id`：

```bash
curl -X POST http://127.0.0.1:7861/v1/tts/jobs \
  -H "Authorization: Bearer change-me" \
  -F "text=欢迎使用 IndexTTS2 API 服务。" \
  -F "prompt_voice_id=<voice_id>" \
  -F "emotion_mode=speaker" \
  -F "max_text_tokens_per_segment=120"
```

为了兼容临时调用，`/v1/tts/jobs` 仍然支持直接传 `prompt_audio=@...`；这类上传也会进入 voice asset 存储并按 sha256 去重。但生产客户端推荐使用 `voice_id`，避免每次请求重复传音频。

响应示例：

```json
{
  "task_id": "9f6b5d9f7f8d4d3b9b7b0e4f3d2c1a00",
  "status": "queued",
  "progress": 0.0,
  "message": "queued",
  "task_url": "http://127.0.0.1:7861/v1/tasks/9f6b5d9f7f8d4d3b9b7b0e4f3d2c1a00",
  "events_url": "http://127.0.0.1:7861/v1/tasks/9f6b5d9f7f8d4d3b9b7b0e4f3d2c1a00/events",
  "audio_url": "http://127.0.0.1:7861/v1/tasks/9f6b5d9f7f8d4d3b9b7b0e4f3d2c1a00/audio",
  "queue_position": 1
}
```

使用情感参考音频：

```bash
curl -X POST http://127.0.0.1:7861/v1/tts/jobs \
  -H "Authorization: Bearer change-me" \
  -F "text=这是一段带情感参考音频的测试。" \
  -F "prompt_voice_id=<voice_id>" \
  -F "emotion_mode=audio" \
  -F "emo_voice_id=<emotion_voice_id>" \
  -F "emo_alpha=0.8"
```

## 查询进度和下载音频

轮询任务状态：

```bash
curl -H "Authorization: Bearer change-me" \
  http://127.0.0.1:7861/v1/tasks/<task_id>
```

任务状态字段：

| 字段 | 说明 |
| --- | --- |
| `status` | `queued`、`running`、`succeeded`、`failed`、`cancelled` |
| `progress` | `0.0` 到 `1.0` |
| `message` | 当前阶段说明，例如 `text processing...`、`saving audio...` |
| `queue_position` | 排队位置；运行后为 `null` |
| `audio_url` | 音频下载地址；任务完成后可用 |

SSE 实时进度：

```bash
curl -N -H "Authorization: Bearer change-me" \
  http://127.0.0.1:7861/v1/tasks/<task_id>/events
```

下载音频：

```bash
curl -H "Authorization: Bearer change-me" \
  http://127.0.0.1:7861/v1/tasks/<task_id>/audio \
  --output out.wav
```

## 生成语音：JSON/base64 任务

适合没有 multipart 上传能力的客户端。

```bash
curl -X POST http://127.0.0.1:7861/v1/tts/jobs/json \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"这是 JSON base64 调用示例。\",
    \"prompt_voice_id\": \"<voice_id>\",
    \"emotion_mode\": \"speaker\"
  }"
```

## 情感控制

`emotion_mode` 支持四种：

| 值 | 说明 |
| --- | --- |
| `speaker` | 情感跟随音色参考音频 |
| `audio` | 使用单独的情感参考音频，需要传 `emo_audio` 或 `emo_audio_base64` |
| `vector` | 使用 8 维情感向量 `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]` |
| `text` | 使用情感描述文本，需要传 `emo_text`；不传时会用正文作为情感描述 |

情感向量 multipart 示例：

```bash
curl -X POST http://127.0.0.1:7861/v1/tts/jobs \
  -H "Authorization: Bearer change-me" \
  -F "text=对不起嘛，我真的不是故意的。" \
  -F "prompt_voice_id=<voice_id>" \
  -F "emotion_mode=vector" \
  -F "emo_vector=[0,0,0.8,0,0,0,0,0]"
```

情感文本示例：

```bash
curl -X POST http://127.0.0.1:7861/v1/tts/jobs \
  -H "Authorization: Bearer change-me" \
  -F "text=快躲起来，他要来了！" \
  -F "prompt_voice_id=<voice_id>" \
  -F "emotion_mode=text" \
  -F "emo_text=非常紧张和害怕" \
  -F "emo_alpha=0.6"
```

## 其他接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查，不需要鉴权 |
| `GET` | `/v1/config` | 当前模型和生成范围 |
| `POST` | `/v1/voices` | 上传参考音频资产 |
| `POST` | `/v1/voices/json` | JSON/base64 上传参考音频资产 |
| `GET` | `/v1/voices` | 查看参考音频资产列表 |
| `GET` | `/v1/voices/{voice_id}` | 查看参考音频资产 |
| `DELETE` | `/v1/voices/{voice_id}` | 删除参考音频资产 |
| `POST` | `/v1/tts/jobs` | multipart 上传并提交 TTS 任务 |
| `POST` | `/v1/tts/jobs/json` | JSON/base64 提交 TTS 任务 |
| `POST` | `/v1/segments/preview` | 查看分句结果 |
| `GET` | `/v1/tasks` | 查看最近任务列表 |
| `GET` | `/v1/tasks/{task_id}` | 查看任务元数据 |
| `GET` | `/v1/tasks/{task_id}/events` | SSE 实时任务进度 |
| `GET` | `/v1/tasks/{task_id}/audio` | 下载生成音频 |
| `DELETE` | `/v1/tasks/{task_id}` | 删除任务音频 |
| `GET` | `/v1/glossary` | 查看术语表 |
| `PUT` | `/v1/glossary` | 新增或更新术语读音 |
| `DELETE` | `/v1/glossary/{term}` | 删除术语 |

分句预览：

```bash
curl -X POST http://127.0.0.1:7861/v1/segments/preview \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"text":"这是一段较长的文本，可以先看服务会怎么分句。","max_text_tokens_per_segment":120}'
```

更新术语读音：

```bash
curl -X PUT http://127.0.0.1:7861/v1/glossary \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"term":"IndexTTS2","zh":"Index T T S 二","en":"Index T T S two"}'
```

## 生产部署建议

### systemd

`/etc/systemd/system/indextts-api.service`：

```ini
[Unit]
Description=IndexTTS2 FastAPI Server
After=network.target

[Service]
WorkingDirectory=/opt/index-tts
Environment=INDEXTTS_API_KEY=change-me
Environment=INDEXTTS_MODEL_DIR=/opt/index-tts/checkpoints
Environment=INDEXTTS_OUTPUT_DIR=/data/indextts-outputs
Environment=INDEXTTS_FP16=true
ExecStart=/usr/local/bin/uv run indextts-api --host 127.0.0.1 --port 7861
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Nginx 反向代理

```nginx
server {
    listen 443 ssl http2;
    server_name tts.example.com;

    client_max_body_size 50m;

    location / {
        proxy_pass http://127.0.0.1:7861;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

建议：

- 公网部署务必开启 `INDEXTTS_API_KEY`，并放在 HTTPS 后面。
- 单张 GPU 通常让一个服务进程独占即可；服务内部已经用后台队列串行推理。
- 如果要提高吞吐量，可以按 GPU 数量启动多个进程，再用网关分发。
- `outputs/api/voices` 保存去重后的参考音频资产，`outputs/api/results` 保存生成结果；可以用定时任务清理旧任务结果，voice asset 建议通过接口按需删除。
