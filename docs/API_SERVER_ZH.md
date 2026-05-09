# IndexTTS2 FastAPI 服务部署说明

这个 API 服务会在进程启动时加载一次 IndexTTS2 模型，然后把推理请求串行送进模型，避免并发请求互相污染模型内部的参考音频缓存。适合作为你自己应用调用的基础语音合成能力。

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
| `--lazy_load` | 首个请求到来时再加载模型 |
| `--allow_local_paths` | 允许请求传服务器本地音频路径。默认关闭，推荐用上传或 base64 |
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

## 生成语音：multipart 上传

直接返回 wav 文件：

```bash
curl -X POST http://127.0.0.1:7861/v1/tts \
  -H "Authorization: Bearer change-me" \
  -F "text=欢迎使用 IndexTTS2 API 服务。" \
  -F "prompt_audio=@examples/voice_01.wav" \
  -F "emotion_mode=speaker" \
  -F "max_text_tokens_per_segment=120" \
  --output out.wav
```

返回 JSON，里面包含可下载音频 URL：

```bash
curl -X POST http://127.0.0.1:7861/v1/tts \
  -H "Authorization: Bearer change-me" \
  -F "text=这是一段带情感参考音频的测试。" \
  -F "prompt_audio=@examples/voice_07.wav" \
  -F "emotion_mode=audio" \
  -F "emo_audio=@examples/emo_sad.wav" \
  -F "emo_alpha=0.8" \
  -F "response_format=json"
```

`response_format` 支持：

- `file`：直接返回 `audio/wav`
- `json`：返回任务 ID、音频 URL、耗时等元数据
- `base64`：返回 JSON，并内联 `audio_base64`

## 生成语音：JSON/base64

适合没有 multipart 上传能力的客户端。

```bash
PROMPT_B64=$(base64 -i examples/voice_01.wav)

curl -X POST http://127.0.0.1:7861/v1/tts/json \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"这是 JSON base64 调用示例。\",
    \"prompt_audio_base64\": \"$PROMPT_B64\",
    \"prompt_audio_filename\": \"voice_01.wav\",
    \"emotion_mode\": \"speaker\",
    \"include_audio_base64\": false
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
curl -X POST http://127.0.0.1:7861/v1/tts \
  -H "Authorization: Bearer change-me" \
  -F "text=对不起嘛，我真的不是故意的。" \
  -F "prompt_audio=@examples/voice_09.wav" \
  -F "emotion_mode=vector" \
  -F "emo_vector=[0,0,0.8,0,0,0,0,0]" \
  --output sad.wav
```

情感文本示例：

```bash
curl -X POST http://127.0.0.1:7861/v1/tts \
  -H "Authorization: Bearer change-me" \
  -F "text=快躲起来，他要来了！" \
  -F "prompt_audio=@examples/voice_12.wav" \
  -F "emotion_mode=text" \
  -F "emo_text=非常紧张和害怕" \
  -F "emo_alpha=0.6" \
  --output fear.wav
```

## 其他接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查，不需要鉴权 |
| `GET` | `/v1/config` | 当前模型和生成范围 |
| `POST` | `/v1/segments/preview` | 查看分句结果 |
| `GET` | `/v1/tasks/{task_id}` | 查看任务元数据 |
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
- 单张 GPU 通常让一个服务进程独占即可；服务内部已经用锁串行推理。
- 如果要提高吞吐量，可以按 GPU 数量启动多个进程，再用网关分发。
- `outputs/api/results` 会保存生成结果；可以用定时任务清理旧文件。
