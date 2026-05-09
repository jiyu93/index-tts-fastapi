import argparse
import asyncio
import base64
import binascii
import hmac
import json
import os
import time
import uuid
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from indextts.infer_v2 import IndexTTS2


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm"}
REQUIRED_MODEL_FILES = ("bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt")
EMOTION_MODES = {"speaker", "audio", "vector", "text"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _safe_task_id() -> str:
    return uuid.uuid4().hex


@dataclass
class ServerSettings:
    model_dir: Path = Path(os.getenv("INDEXTTS_MODEL_DIR", "checkpoints"))
    cfg_path: Path | None = None
    output_dir: Path = Path(os.getenv("INDEXTTS_OUTPUT_DIR", "outputs/api"))
    host: str = os.getenv("INDEXTTS_HOST", "0.0.0.0")
    port: int = _env_int("INDEXTTS_PORT", 7861)
    api_key: str | None = os.getenv("INDEXTTS_API_KEY")
    cors_origins: list[str] | None = None
    use_fp16: bool = _env_bool("INDEXTTS_FP16", False)
    use_deepspeed: bool = _env_bool("INDEXTTS_DEEPSPEED", False)
    use_cuda_kernel: bool = _env_bool("INDEXTTS_CUDA_KERNEL", False)
    use_accel: bool = _env_bool("INDEXTTS_ACCEL", False)
    use_torch_compile: bool = _env_bool("INDEXTTS_TORCH_COMPILE", False)
    device: str | None = os.getenv("INDEXTTS_DEVICE")
    lazy_load: bool = _env_bool("INDEXTTS_LAZY_LOAD", False)
    allow_local_paths: bool = _env_bool("INDEXTTS_ALLOW_LOCAL_PATHS", False)
    max_upload_mb: int = _env_int("INDEXTTS_MAX_UPLOAD_MB", 50)

    def __post_init__(self) -> None:
        self.model_dir = self.model_dir.expanduser().resolve()
        self.cfg_path = (self.cfg_path or self.model_dir / "config.yaml").expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()
        if self.cors_origins is None:
            raw_origins = os.getenv("INDEXTTS_CORS_ORIGINS", "")
            self.cors_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


class GenerationParams(BaseModel):
    do_sample: bool = True
    top_p: float = Field(0.8, ge=0.0, le=1.0)
    top_k: int = Field(30, ge=0, le=1000)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    length_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    num_beams: int = Field(3, ge=1, le=10)
    repetition_penalty: float = Field(10.0, ge=0.1, le=20.0)
    max_mel_tokens: int = Field(1500, ge=50)
    max_text_tokens_per_segment: int = Field(120, ge=20)
    interval_silence: int = Field(200, ge=0, le=5000)
    verbose: bool = False


class TTSJsonRequest(GenerationParams):
    text: str = Field(..., min_length=1)
    prompt_audio_base64: str | None = None
    prompt_audio_path: str | None = None
    prompt_audio_filename: str = "prompt.wav"
    emotion_mode: Literal["speaker", "audio", "vector", "text"] = "speaker"
    emo_audio_base64: str | None = None
    emo_audio_path: str | None = None
    emo_audio_filename: str = "emotion.wav"
    emo_alpha: float = Field(1.0, ge=0.0, le=1.0)
    emo_vector: list[float] | None = None
    emo_text: str | None = None
    use_random: bool = False
    normalize_emo_vector: bool = True
    include_audio_base64: bool = False


class TTSResponse(BaseModel):
    task_id: str
    audio_url: str
    duration_seconds: float | None = None
    elapsed_seconds: float
    sample_rate: int = 22050
    model_version: str | int | float | None = None
    audio_base64: str | None = None


class SegmentPreviewRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_text_tokens_per_segment: int = Field(120, ge=20)


class SegmentPreviewResponse(BaseModel):
    segments: list[dict[str, Any]]
    total_tokens: int


class GlossaryTermRequest(BaseModel):
    term: str = Field(..., min_length=1)
    zh: str | None = None
    en: str | None = None


class TaskRecord(BaseModel):
    task_id: str
    output_path: str
    created_at: float
    text: str
    emotion_mode: str
    duration_seconds: float | None = None
    elapsed_seconds: float


class IndexTTSService:
    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self.tts: IndexTTS2 | None = None
        self.load_started_at: float | None = None
        self.loaded_at: float | None = None
        self.load_error: str | None = None
        self.infer_lock = asyncio.Lock()
        self.tasks: dict[str, TaskRecord] = {}
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.output_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (self.settings.output_dir / "results").mkdir(parents=True, exist_ok=True)

    @property
    def loaded(self) -> bool:
        return self.tts is not None

    def validate_model_files(self) -> None:
        if not self.settings.model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {self.settings.model_dir}")
        missing = [name for name in REQUIRED_MODEL_FILES if not (self.settings.model_dir / name).exists()]
        if missing:
            missing_list = ", ".join(missing)
            raise FileNotFoundError(f"Missing model files in {self.settings.model_dir}: {missing_list}")
        if not self.settings.cfg_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {self.settings.cfg_path}")

    def load(self) -> None:
        if self.tts is not None:
            return
        self.validate_model_files()
        self.load_started_at = time.time()
        self.load_error = None
        try:
            self.tts = IndexTTS2(
                cfg_path=str(self.settings.cfg_path),
                model_dir=str(self.settings.model_dir),
                use_fp16=self.settings.use_fp16,
                device=self.settings.device,
                use_cuda_kernel=self.settings.use_cuda_kernel,
                use_deepspeed=self.settings.use_deepspeed,
                use_accel=self.settings.use_accel,
                use_torch_compile=self.settings.use_torch_compile,
            )
            self.loaded_at = time.time()
        except Exception as exc:
            self.load_error = repr(exc)
            raise

    async def ensure_loaded(self) -> IndexTTS2:
        if self.tts is None:
            try:
                await asyncio.to_thread(self.load)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Failed to load IndexTTS2 model: {exc}") from exc
        if self.tts is None:
            raise RuntimeError("IndexTTS2 model is not loaded")
        return self.tts

    async def save_upload(self, upload: UploadFile, stem: str) -> Path:
        suffix = Path(upload.filename or "").suffix.lower() or ".wav"
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported audio extension: {suffix}")

        output_path = self.settings.output_dir / "prompts" / f"{stem}{suffix}"
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        written = 0
        with output_path.open("wb") as file:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    output_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"Audio upload exceeds {self.settings.max_upload_mb} MB")
                file.write(chunk)
        if written == 0:
            output_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Uploaded audio is empty")
        return output_path

    def save_base64_audio(self, audio_base64: str, filename: str, stem: str) -> Path:
        suffix = Path(filename).suffix.lower() or ".wav"
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported audio extension: {suffix}")
        try:
            if "," in audio_base64:
                audio_base64 = audio_base64.split(",", 1)[1]
            data = base64.b64decode(audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 audio") from exc
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Audio payload exceeds {self.settings.max_upload_mb} MB")
        if not data:
            raise HTTPException(status_code=400, detail="Audio payload is empty")
        output_path = self.settings.output_dir / "prompts" / f"{stem}{suffix}"
        output_path.write_bytes(data)
        return output_path

    def resolve_local_audio_path(self, value: str | None, field_name: str) -> Path | None:
        if not value:
            return None
        if not self.settings.allow_local_paths:
            raise HTTPException(status_code=400, detail=f"{field_name} is disabled. Upload audio or use base64 instead.")
        path = Path(value).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=400, detail=f"{field_name} does not exist: {value}")
        if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported audio extension: {path.suffix}")
        return path

    def get_task(self, task_id: str) -> TaskRecord:
        task = self.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if not Path(task.output_path).exists():
            raise HTTPException(status_code=404, detail="Task audio file not found")
        return task

    async def synthesize(
        self,
        *,
        text: str,
        prompt_audio_path: Path,
        emotion_mode: str,
        emo_audio_path: Path | None,
        emo_alpha: float,
        emo_vector: list[float] | None,
        emo_text: str | None,
        use_random: bool,
        normalize_emo_vector: bool,
        generation: GenerationParams,
    ) -> TaskRecord:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is empty")
        if not 0.0 <= emo_alpha <= 1.0:
            raise HTTPException(status_code=400, detail="emo_alpha must be between 0.0 and 1.0")
        if emotion_mode not in EMOTION_MODES:
            raise HTTPException(status_code=400, detail=f"Unsupported emotion_mode: {emotion_mode}")
        if emotion_mode == "audio" and emo_audio_path is None:
            raise HTTPException(status_code=400, detail="emo_audio is required when emotion_mode=audio")
        if emotion_mode == "vector":
            if emo_vector is None:
                raise HTTPException(status_code=400, detail="emo_vector is required when emotion_mode=vector")
            if len(emo_vector) != 8:
                raise HTTPException(status_code=400, detail="emo_vector must contain exactly 8 numbers")
            if any(value < 0.0 for value in emo_vector):
                raise HTTPException(status_code=400, detail="emo_vector values must be non-negative")

        tts = await self.ensure_loaded()
        task_id = _safe_task_id()
        output_path = self.settings.output_dir / "results" / f"{task_id}.wav"

        vector = emo_vector
        if emotion_mode == "vector" and vector is not None and normalize_emo_vector:
            vector = tts.normalize_emo_vec(vector, apply_bias=True)

        infer_kwargs = {
            "spk_audio_prompt": str(prompt_audio_path),
            "text": text.strip(),
            "output_path": str(output_path),
            "emo_audio_prompt": str(emo_audio_path) if emotion_mode == "audio" and emo_audio_path else None,
            "emo_alpha": emo_alpha,
            "emo_vector": vector if emotion_mode == "vector" else None,
            "use_emo_text": emotion_mode == "text",
            "emo_text": emo_text.strip() if emo_text else None,
            "use_random": use_random,
            "interval_silence": generation.interval_silence,
            "verbose": generation.verbose,
            "max_text_tokens_per_segment": generation.max_text_tokens_per_segment,
            "do_sample": generation.do_sample,
            "top_p": generation.top_p,
            "top_k": generation.top_k if generation.top_k > 0 else None,
            "temperature": generation.temperature,
            "length_penalty": generation.length_penalty,
            "num_beams": generation.num_beams,
            "repetition_penalty": generation.repetition_penalty,
            "max_mel_tokens": generation.max_mel_tokens,
        }

        started_at = time.perf_counter()
        async with self.infer_lock:
            result = await asyncio.to_thread(tts.infer, **infer_kwargs)
        elapsed = time.perf_counter() - started_at
        if result is None or not output_path.exists():
            raise HTTPException(status_code=500, detail="Inference failed to produce audio")

        record = TaskRecord(
            task_id=task_id,
            output_path=str(output_path),
            created_at=time.time(),
            text=text.strip(),
            emotion_mode=emotion_mode,
            duration_seconds=get_wav_duration(output_path),
            elapsed_seconds=elapsed,
        )
        self.tasks[task_id] = record
        return record


def get_wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except wave.Error:
        return None


def parse_emo_vector(raw: str | None) -> list[float] | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in raw.split(",")]
    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail="emo_vector must be a JSON array or comma-separated list")
    try:
        return [float(value) for value in parsed]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="emo_vector must contain numbers") from exc


def response_from_task(
    request: Request,
    service: IndexTTSService,
    task: TaskRecord,
    include_audio_base64: bool = False,
) -> TTSResponse:
    audio_base64 = None
    if include_audio_base64:
        audio_base64 = base64.b64encode(Path(task.output_path).read_bytes()).decode("ascii")
    return TTSResponse(
        task_id=task.task_id,
        audio_url=str(request.url_for("get_task_audio", task_id=task.task_id)),
        duration_seconds=task.duration_seconds,
        elapsed_seconds=task.elapsed_seconds,
        model_version=service.tts.model_version if service.tts is not None else None,
        audio_base64=audio_base64,
    )


def create_auth_dependency(settings: ServerSettings):
    async def require_api_key(request: Request) -> None:
        if not settings.api_key:
            return
        authorization = request.headers.get("authorization", "")
        token = request.headers.get("x-api-key", "")
        if authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
        if not hmac.compare_digest(token, settings.api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    return require_api_key


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()
    service = IndexTTSService(settings)
    require_api_key = create_auth_dependency(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service
        if not settings.lazy_load:
            await service.ensure_loaded()
        yield

    app = FastAPI(
        title="IndexTTS2 API Server",
        version="1.0.0",
        description="FastAPI service wrapper for IndexTTS2 zero-shot speech synthesis.",
        lifespan=lifespan,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "IndexTTS2 API Server",
            "docs_url": "/docs",
            "health_url": "/health",
            "model_loaded": service.loaded,
        }

    @app.get("/health")
    async def health() -> JSONResponse:
        status_code = 200 if service.load_error is None else 503
        payload = {
            "ok": service.load_error is None,
            "model_loaded": service.loaded,
            "model_dir": str(settings.model_dir),
            "cfg_path": str(settings.cfg_path),
            "device": service.tts.device if service.tts is not None else settings.device,
            "model_version": service.tts.model_version if service.tts is not None else None,
            "load_started_at": service.load_started_at,
            "loaded_at": service.loaded_at,
            "load_error": service.load_error,
            "tasks": len(service.tasks),
        }
        return JSONResponse(payload, status_code=status_code)

    @app.get("/v1/config", dependencies=[Depends(require_api_key)])
    async def config() -> dict[str, Any]:
        tts = await service.ensure_loaded()
        return {
            "model_version": tts.model_version,
            "device": tts.device,
            "use_fp16": tts.use_fp16,
            "max_text_tokens": int(tts.cfg.gpt.max_text_tokens),
            "max_mel_tokens": int(tts.cfg.gpt.max_mel_tokens),
            "emotion_modes": sorted(EMOTION_MODES),
            "outputs_dir": str(settings.output_dir),
            "allow_local_paths": settings.allow_local_paths,
        }

    @app.post("/v1/tts", dependencies=[Depends(require_api_key)])
    async def synthesize_multipart(
        request: Request,
        text: str = Form(...),
        prompt_audio: UploadFile | None = File(None),
        prompt_audio_path: str | None = Form(None),
        emotion_mode: str = Form("speaker"),
        emo_audio: UploadFile | None = File(None),
        emo_audio_path: str | None = Form(None),
        emo_alpha: float = Form(1.0),
        emo_vector: str | None = Form(None),
        emo_text: str | None = Form(None),
        use_random: bool = Form(False),
        normalize_emo_vector: bool = Form(True),
        response_format: Literal["file", "json", "base64"] = Form("file"),
        do_sample: bool = Form(True),
        top_p: float = Form(0.8),
        top_k: int = Form(30),
        temperature: float = Form(0.8),
        length_penalty: float = Form(0.0),
        num_beams: int = Form(3),
        repetition_penalty: float = Form(10.0),
        max_mel_tokens: int = Form(1500),
        max_text_tokens_per_segment: int = Form(120),
        interval_silence: int = Form(200),
        verbose: bool = Form(False),
    ):
        task_stem = _safe_task_id()
        if prompt_audio is not None:
            prompt_path = await service.save_upload(prompt_audio, f"{task_stem}_prompt")
        else:
            prompt_path = service.resolve_local_audio_path(prompt_audio_path, "prompt_audio_path")
        if prompt_path is None:
            raise HTTPException(status_code=400, detail="prompt_audio is required")

        if emo_audio is not None:
            emotion_path = await service.save_upload(emo_audio, f"{task_stem}_emotion")
        else:
            emotion_path = service.resolve_local_audio_path(emo_audio_path, "emo_audio_path")

        generation = GenerationParams(
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            max_mel_tokens=max_mel_tokens,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            interval_silence=interval_silence,
            verbose=verbose,
        )
        task = await service.synthesize(
            text=text,
            prompt_audio_path=prompt_path,
            emotion_mode=emotion_mode,
            emo_audio_path=emotion_path,
            emo_alpha=emo_alpha,
            emo_vector=parse_emo_vector(emo_vector),
            emo_text=emo_text,
            use_random=use_random,
            normalize_emo_vector=normalize_emo_vector,
            generation=generation,
        )

        if response_format == "file":
            return FileResponse(
                task.output_path,
                media_type="audio/wav",
                filename=f"{task.task_id}.wav",
                headers={
                    "X-Task-ID": task.task_id,
                    "X-Elapsed-Seconds": f"{task.elapsed_seconds:.4f}",
                },
            )
        payload = response_from_task(request, service, task, include_audio_base64=response_format == "base64")
        return payload

    @app.post("/v1/tts/json", response_model=TTSResponse, dependencies=[Depends(require_api_key)])
    async def synthesize_json(request: Request, body: TTSJsonRequest) -> TTSResponse:
        task_stem = _safe_task_id()
        if body.prompt_audio_base64:
            prompt_path = service.save_base64_audio(body.prompt_audio_base64, body.prompt_audio_filename, f"{task_stem}_prompt")
        else:
            prompt_path = service.resolve_local_audio_path(body.prompt_audio_path, "prompt_audio_path")
        if prompt_path is None:
            raise HTTPException(status_code=400, detail="prompt_audio_base64 is required")

        emotion_path = None
        if body.emo_audio_base64:
            emotion_path = service.save_base64_audio(body.emo_audio_base64, body.emo_audio_filename, f"{task_stem}_emotion")
        elif body.emo_audio_path:
            emotion_path = service.resolve_local_audio_path(body.emo_audio_path, "emo_audio_path")

        task = await service.synthesize(
            text=body.text,
            prompt_audio_path=prompt_path,
            emotion_mode=body.emotion_mode,
            emo_audio_path=emotion_path,
            emo_alpha=body.emo_alpha,
            emo_vector=body.emo_vector,
            emo_text=body.emo_text,
            use_random=body.use_random,
            normalize_emo_vector=body.normalize_emo_vector,
            generation=body,
        )
        return response_from_task(request, service, task, include_audio_base64=body.include_audio_base64)

    @app.post("/v1/segments/preview", response_model=SegmentPreviewResponse, dependencies=[Depends(require_api_key)])
    async def preview_segments(body: SegmentPreviewRequest) -> SegmentPreviewResponse:
        tts = await service.ensure_loaded()
        text_tokens = tts.tokenizer.tokenize(body.text)
        segments = tts.tokenizer.split_segments(text_tokens, body.max_text_tokens_per_segment)
        return SegmentPreviewResponse(
            total_tokens=len(text_tokens),
            segments=[
                {"index": index, "text": "".join(segment), "tokens": len(segment)}
                for index, segment in enumerate(segments)
            ],
        )

    @app.get("/v1/tasks/{task_id}", response_model=TaskRecord, dependencies=[Depends(require_api_key)])
    async def get_task(task_id: str) -> TaskRecord:
        return service.get_task(task_id)

    @app.get("/v1/tasks/{task_id}/audio", name="get_task_audio", dependencies=[Depends(require_api_key)])
    async def get_task_audio(task_id: str) -> FileResponse:
        task = service.get_task(task_id)
        return FileResponse(task.output_path, media_type="audio/wav", filename=f"{task.task_id}.wav")

    @app.delete("/v1/tasks/{task_id}", dependencies=[Depends(require_api_key)])
    async def delete_task(task_id: str) -> dict[str, Any]:
        task = service.get_task(task_id)
        Path(task.output_path).unlink(missing_ok=True)
        service.tasks.pop(task_id, None)
        return {"deleted": True, "task_id": task_id}

    @app.get("/v1/glossary", dependencies=[Depends(require_api_key)])
    async def get_glossary() -> dict[str, Any]:
        tts = await service.ensure_loaded()
        return {
            "enabled": tts.normalizer.enable_glossary,
            "path": tts.glossary_path,
            "terms": tts.normalizer.term_glossary,
        }

    @app.put("/v1/glossary", dependencies=[Depends(require_api_key)])
    async def put_glossary_term(body: GlossaryTermRequest) -> dict[str, Any]:
        tts = await service.ensure_loaded()
        reading: dict[str, str] = {}
        if body.zh:
            reading["zh"] = body.zh
        if body.en:
            reading["en"] = body.en
        if not reading:
            raise HTTPException(status_code=400, detail="At least one reading is required: zh or en")
        tts.normalizer.term_glossary[body.term.rstrip()] = reading
        tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
        return {"saved": True, "term": body.term.rstrip(), "reading": reading}

    @app.delete("/v1/glossary/{term}", dependencies=[Depends(require_api_key)])
    async def delete_glossary_term(term: str) -> dict[str, Any]:
        tts = await service.ensure_loaded()
        existed = term in tts.normalizer.term_glossary
        tts.normalizer.term_glossary.pop(term, None)
        tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
        return {"deleted": existed, "term": term}

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IndexTTS2 FastAPI server")
    parser.add_argument("--host", default=os.getenv("INDEXTTS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=_env_int("INDEXTTS_PORT", 7861))
    parser.add_argument("--model_dir", default=os.getenv("INDEXTTS_MODEL_DIR", "checkpoints"))
    parser.add_argument("--cfg_path", default=os.getenv("INDEXTTS_CFG_PATH"))
    parser.add_argument("--output_dir", default=os.getenv("INDEXTTS_OUTPUT_DIR", "outputs/api"))
    parser.add_argument("--api_key", default=os.getenv("INDEXTTS_API_KEY"))
    parser.add_argument("--cors_origins", default=os.getenv("INDEXTTS_CORS_ORIGINS", ""))
    parser.add_argument("--fp16", action="store_true", default=_env_bool("INDEXTTS_FP16", False))
    parser.add_argument("--deepspeed", action="store_true", default=_env_bool("INDEXTTS_DEEPSPEED", False))
    parser.add_argument("--cuda_kernel", action="store_true", default=_env_bool("INDEXTTS_CUDA_KERNEL", False))
    parser.add_argument("--accel", action="store_true", default=_env_bool("INDEXTTS_ACCEL", False))
    parser.add_argument("--torch_compile", action="store_true", default=_env_bool("INDEXTTS_TORCH_COMPILE", False))
    parser.add_argument("--device", default=os.getenv("INDEXTTS_DEVICE"))
    parser.add_argument("--lazy_load", action="store_true", default=_env_bool("INDEXTTS_LAZY_LOAD", False))
    parser.add_argument("--allow_local_paths", action="store_true", default=_env_bool("INDEXTTS_ALLOW_LOCAL_PATHS", False))
    parser.add_argument("--max_upload_mb", type=int, default=_env_int("INDEXTTS_MAX_UPLOAD_MB", 50))
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload for development")
    return parser


def main() -> None:
    import uvicorn

    args = build_arg_parser().parse_args()
    settings = ServerSettings(
        model_dir=Path(args.model_dir),
        cfg_path=Path(args.cfg_path) if args.cfg_path else None,
        output_dir=Path(args.output_dir),
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        cors_origins=[origin.strip() for origin in args.cors_origins.split(",") if origin.strip()],
        use_fp16=args.fp16,
        use_deepspeed=args.deepspeed,
        use_cuda_kernel=args.cuda_kernel,
        use_accel=args.accel,
        use_torch_compile=args.torch_compile,
        device=args.device,
        lazy_load=args.lazy_load,
        allow_local_paths=args.allow_local_paths,
        max_upload_mb=args.max_upload_mb,
    )

    if args.reload:
        os.environ["INDEXTTS_MODEL_DIR"] = str(settings.model_dir)
        os.environ["INDEXTTS_CFG_PATH"] = str(settings.cfg_path)
        os.environ["INDEXTTS_OUTPUT_DIR"] = str(settings.output_dir)
        if settings.api_key:
            os.environ["INDEXTTS_API_KEY"] = settings.api_key
        os.environ["INDEXTTS_CORS_ORIGINS"] = ",".join(settings.cors_origins or [])
        os.environ["INDEXTTS_FP16"] = str(settings.use_fp16).lower()
        os.environ["INDEXTTS_DEEPSPEED"] = str(settings.use_deepspeed).lower()
        os.environ["INDEXTTS_CUDA_KERNEL"] = str(settings.use_cuda_kernel).lower()
        os.environ["INDEXTTS_ACCEL"] = str(settings.use_accel).lower()
        os.environ["INDEXTTS_TORCH_COMPILE"] = str(settings.use_torch_compile).lower()
        if settings.device:
            os.environ["INDEXTTS_DEVICE"] = settings.device
        os.environ["INDEXTTS_LAZY_LOAD"] = str(settings.lazy_load).lower()
        os.environ["INDEXTTS_ALLOW_LOCAL_PATHS"] = str(settings.allow_local_paths).lower()
        os.environ["INDEXTTS_MAX_UPLOAD_MB"] = str(settings.max_upload_mb)
        uvicorn.run("indextts.api_server:create_app", factory=True, host=settings.host, port=settings.port, reload=True)
    else:
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
