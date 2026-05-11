import argparse
import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import os
import time
import uuid
import wave
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from indextts.infer_v2 import IndexTTS2


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm"}
REQUIRED_MODEL_FILES = ("bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt")
EMOTION_MODES = {"speaker", "audio", "vector", "text"}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


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
    max_queue_size: int = _env_int("INDEXTTS_MAX_QUEUE_SIZE", 100)

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
    prompt_voice_id: str | None = None
    prompt_audio_base64: str | None = None
    prompt_audio_path: str | None = None
    prompt_audio_filename: str = "prompt.wav"
    emotion_mode: Literal["speaker", "audio", "vector", "text"] = "speaker"
    emo_voice_id: str | None = None
    emo_audio_base64: str | None = None
    emo_audio_path: str | None = None
    emo_audio_filename: str = "emotion.wav"
    emo_alpha: float = Field(1.0, ge=0.0, le=1.0)
    emo_vector: list[float] | None = None
    emo_text: str | None = None
    use_random: bool = False
    normalize_emo_vector: bool = True


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


class VoiceJsonRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1)
    filename: str = "voice.wav"
    content_type: str | None = None


class VoiceAsset(BaseModel):
    voice_id: str
    sha256: str
    original_filename: str
    path: str
    content_type: str | None = None
    size_bytes: int
    created_at: float
    last_used_at: float | None = None
    use_count: int = 0


class JobSpec(BaseModel):
    text: str
    prompt_audio_path: str
    prompt_voice_id: str | None = None
    emotion_mode: str
    emo_audio_path: str | None = None
    emo_voice_id: str | None = None
    emo_alpha: float = 1.0
    emo_vector: list[float] | None = None
    emo_text: str | None = None
    use_random: bool = False
    normalize_emo_vector: bool = True
    generation: GenerationParams = Field(default_factory=GenerationParams)


class TaskRecord(BaseModel):
    task_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] = "queued"
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str = "queued"
    created_at: float
    updated_at: float
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_seconds: float | None = None
    queue_position: int | None = None
    text: str
    emotion_mode: str
    prompt_voice_id: str | None = None
    emo_voice_id: str | None = None
    output_path: str | None = None
    audio_url: str | None = None
    duration_seconds: float | None = None
    sample_rate: int = 22050
    error: str | None = None


class JobSubmitResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    task_url: str
    events_url: str
    audio_url: str
    queue_position: int | None = None


class JobProgress:
    def __init__(self, service: "IndexTTSService", task_id: str) -> None:
        self.service = service
        self.task_id = task_id

    def __call__(self, value: float | None = None, desc: str | None = None) -> None:
        self.service.update_progress(self.task_id, value=value, message=desc)


class IndexTTSService:
    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self.tts: IndexTTS2 | None = None
        self.load_started_at: float | None = None
        self.loaded_at: float | None = None
        self.load_error: str | None = None
        self.load_lock = asyncio.Lock()
        self.infer_lock = asyncio.Lock()
        self.job_lock = RLock()
        self.jobs: dict[str, TaskRecord] = {}
        self.job_specs: dict[str, JobSpec] = {}
        self.queued_ids: deque[str] = deque()
        self.queue: asyncio.Queue[str] | None = None
        self.worker_task: asyncio.Task[None] | None = None
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.output_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (self.settings.output_dir / "results").mkdir(parents=True, exist_ok=True)
        self.voices_dir = self.settings.output_dir / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.voice_registry_path = self.voices_dir / "voices.json"
        self.voice_lock = RLock()
        self.voices: dict[str, VoiceAsset] = self._load_voice_registry()

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

    async def start(self) -> None:
        if self.queue is None:
            self.queue = asyncio.Queue(maxsize=self.settings.max_queue_size)
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self.worker_task is None:
            return
        self.worker_task.cancel()
        try:
            await self.worker_task
        except asyncio.CancelledError:
            pass

    async def ensure_loaded(self) -> IndexTTS2:
        if self.tts is None:
            async with self.load_lock:
                if self.tts is None:
                    try:
                        await asyncio.to_thread(self.load)
                    except Exception as exc:
                        raise HTTPException(status_code=503, detail=f"Failed to load IndexTTS2 model: {exc}") from exc
        if self.tts is None:
            raise RuntimeError("IndexTTS2 model is not loaded")
        return self.tts

    def _load_voice_registry(self) -> dict[str, VoiceAsset]:
        if not self.voice_registry_path.exists():
            return {}
        try:
            raw = json.loads(self.voice_registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        voices = {}
        for item in raw.get("voices", []):
            asset = VoiceAsset.model_validate(item)
            voices[asset.voice_id] = asset
        return voices

    def _save_voice_registry(self) -> None:
        payload = {
            "voices": [
                asset.model_dump()
                for asset in sorted(self.voices.values(), key=lambda item: item.created_at)
            ]
        }
        tmp_path = self.voice_registry_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.voice_registry_path)

    def _voice_destination(self, sha256: str, suffix: str) -> Path:
        return self.voices_dir / f"{sha256}{suffix}"

    def _register_voice(
        self,
        *,
        tmp_path: Path,
        sha256: str,
        suffix: str,
        original_filename: str,
        content_type: str | None,
        size_bytes: int,
    ) -> VoiceAsset:
        voice_id = sha256
        with self.voice_lock:
            existing = self.voices.get(voice_id)
            if existing is not None and Path(existing.path).exists():
                tmp_path.unlink(missing_ok=True)
                return existing

            destination = self._voice_destination(sha256, suffix)
            if not destination.exists():
                tmp_path.replace(destination)
            else:
                tmp_path.unlink(missing_ok=True)

            now = time.time()
            asset = VoiceAsset(
                voice_id=voice_id,
                sha256=sha256,
                original_filename=original_filename,
                path=str(destination),
                content_type=content_type,
                size_bytes=size_bytes,
                created_at=now,
            )
            self.voices[voice_id] = asset
            self._save_voice_registry()
            return asset

    async def create_voice_from_upload(self, upload: UploadFile) -> VoiceAsset:
        suffix = Path(upload.filename or "").suffix.lower() or ".wav"
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported audio extension: {suffix}")

        tmp_path = self.voices_dir / f".{_safe_task_id()}.tmp"
        digest = hashlib.sha256()
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        written = 0
        with tmp_path.open("wb") as file:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    tmp_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"Audio upload exceeds {self.settings.max_upload_mb} MB")
                digest.update(chunk)
                file.write(chunk)
        if written == 0:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Uploaded audio is empty")
        return self._register_voice(
            tmp_path=tmp_path,
            sha256=digest.hexdigest(),
            suffix=suffix,
            original_filename=upload.filename or f"voice{suffix}",
            content_type=upload.content_type,
            size_bytes=written,
        )

    def create_voice_from_base64(self, audio_base64: str, filename: str, content_type: str | None = None) -> VoiceAsset:
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
        sha256 = hashlib.sha256(data).hexdigest()
        tmp_path = self.voices_dir / f".{_safe_task_id()}.tmp"
        tmp_path.write_bytes(data)
        return self._register_voice(
            tmp_path=tmp_path,
            sha256=sha256,
            suffix=suffix,
            original_filename=filename,
            content_type=content_type,
            size_bytes=len(data),
        )

    def get_voice(self, voice_id: str) -> VoiceAsset:
        with self.voice_lock:
            asset = self.voices.get(voice_id)
            if asset is None:
                raise HTTPException(status_code=404, detail="Voice not found")
            if not Path(asset.path).exists():
                raise HTTPException(status_code=404, detail="Voice audio file not found")
            return asset

    def list_voices(self) -> list[VoiceAsset]:
        with self.voice_lock:
            return sorted(self.voices.values(), key=lambda item: item.created_at, reverse=True)

    def resolve_voice_path(self, voice_id: str) -> Path:
        with self.voice_lock:
            asset = self.get_voice(voice_id)
            updated = asset.model_copy(update={
                "last_used_at": time.time(),
                "use_count": asset.use_count + 1,
            })
            self.voices[voice_id] = updated
            self._save_voice_registry()
            return Path(updated.path)

    def delete_voice(self, voice_id: str) -> VoiceAsset:
        with self.voice_lock:
            asset = self.voices.get(voice_id)
            if asset is None:
                raise HTTPException(status_code=404, detail="Voice not found")
            active_statuses = {"queued", "running"}
            for spec_task_id, spec in self.job_specs.items():
                task = self.jobs.get(spec_task_id)
                if task and task.status in active_statuses and voice_id in {spec.prompt_voice_id, spec.emo_voice_id}:
                    raise HTTPException(status_code=409, detail="Voice is used by an active task")
            self.voices.pop(voice_id)
            self._save_voice_registry()
        Path(asset.path).unlink(missing_ok=True)
        return asset

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

    def validate_job_spec(self, spec: JobSpec) -> None:
        if not spec.text.strip():
            raise HTTPException(status_code=400, detail="Text is empty")
        if not 0.0 <= spec.emo_alpha <= 1.0:
            raise HTTPException(status_code=400, detail="emo_alpha must be between 0.0 and 1.0")
        if spec.emotion_mode not in EMOTION_MODES:
            raise HTTPException(status_code=400, detail=f"Unsupported emotion_mode: {spec.emotion_mode}")
        if spec.emotion_mode == "audio" and spec.emo_audio_path is None:
            raise HTTPException(status_code=400, detail="emo_audio is required when emotion_mode=audio")
        if spec.emotion_mode == "vector":
            if spec.emo_vector is None:
                raise HTTPException(status_code=400, detail="emo_vector is required when emotion_mode=vector")
            if len(spec.emo_vector) != 8:
                raise HTTPException(status_code=400, detail="emo_vector must contain exactly 8 numbers")
            if any(value < 0.0 for value in spec.emo_vector):
                raise HTTPException(status_code=400, detail="emo_vector values must be non-negative")

    async def enqueue(self, spec: JobSpec) -> TaskRecord:
        if self.queue is None:
            await self.start()
        if self.queue is None:
            raise HTTPException(status_code=503, detail="Job queue is not available")
        self.validate_job_spec(spec)
        if self.queue.full():
            raise HTTPException(status_code=429, detail="TTS job queue is full")

        task_id = _safe_task_id()
        now = time.time()
        record = TaskRecord(
            task_id=task_id,
            created_at=now,
            updated_at=now,
            text=spec.text.strip(),
            emotion_mode=spec.emotion_mode,
            prompt_voice_id=spec.prompt_voice_id,
            emo_voice_id=spec.emo_voice_id,
            queue_position=self.queue.qsize() + 1,
        )
        with self.job_lock:
            self.jobs[task_id] = record
            self.job_specs[task_id] = spec
            self.queued_ids.append(task_id)
        await self.queue.put(task_id)
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> TaskRecord:
        with self.job_lock:
            task = self.jobs.get(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="Task not found")
            return task.model_copy(update={"queue_position": self._queue_position(task_id)})

    def list_tasks(self, limit: int = 50) -> list[TaskRecord]:
        with self.job_lock:
            tasks = sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)[:limit]
            return [task.model_copy(update={"queue_position": self._queue_position(task.task_id)}) for task in tasks]

    def _queue_position(self, task_id: str) -> int | None:
        try:
            return list(self.queued_ids).index(task_id) + 1
        except ValueError:
            return None

    def update_progress(self, task_id: str, value: float | None = None, message: str | None = None) -> None:
        with self.job_lock:
            task = self.jobs.get(task_id)
            if task is None or task.status in TERMINAL_STATUSES:
                return
            update: dict[str, Any] = {"updated_at": time.time()}
            if value is not None:
                update["progress"] = min(0.99, max(task.progress, float(value)))
            if message:
                update["message"] = message
            self.jobs[task_id] = task.model_copy(update=update)

    def _set_status(
        self,
        task_id: str,
        *,
        status_value: Literal["queued", "running", "succeeded", "failed", "cancelled"],
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
        output_path: Path | None = None,
        duration_seconds: float | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        with self.job_lock:
            task = self.jobs.get(task_id)
            if task is None:
                return
            now = time.time()
            update: dict[str, Any] = {"status": status_value, "updated_at": now}
            if status_value == "running" and task.started_at is None:
                update["started_at"] = now
            if status_value in TERMINAL_STATUSES:
                update["finished_at"] = now
            if progress is not None:
                update["progress"] = min(1.0, max(0.0, progress))
            if message is not None:
                update["message"] = message
            if error is not None:
                update["error"] = error
            if output_path is not None:
                update["output_path"] = str(output_path)
            if duration_seconds is not None:
                update["duration_seconds"] = duration_seconds
            if elapsed_seconds is not None:
                update["elapsed_seconds"] = elapsed_seconds
            self.jobs[task_id] = task.model_copy(update=update)

    async def _worker_loop(self) -> None:
        if self.queue is None:
            return
        while True:
            task_id = await self.queue.get()
            try:
                with self.job_lock:
                    try:
                        self.queued_ids.remove(task_id)
                    except ValueError:
                        pass
                    task = self.jobs.get(task_id)
                    spec = self.job_specs.get(task_id)
                if task is None or spec is None:
                    continue
                if task.status == "cancelled":
                    continue
                await self._run_job(task_id, spec)
            finally:
                self.queue.task_done()

    async def _run_job(self, task_id: str, spec: JobSpec) -> None:
        started = time.perf_counter()
        output_path = self.settings.output_dir / "results" / f"{task_id}.wav"
        self._set_status(task_id, status_value="running", progress=0.01, message="loading model...")
        try:
            tts = await self.ensure_loaded()
            vector = spec.emo_vector
            if spec.emotion_mode == "vector" and vector is not None and spec.normalize_emo_vector:
                vector = tts.normalize_emo_vec(vector, apply_bias=True)

            infer_kwargs = {
                "spk_audio_prompt": spec.prompt_audio_path,
                "text": spec.text.strip(),
                "output_path": str(output_path),
                "emo_audio_prompt": spec.emo_audio_path if spec.emotion_mode == "audio" else None,
                "emo_alpha": spec.emo_alpha,
                "emo_vector": vector if spec.emotion_mode == "vector" else None,
                "use_emo_text": spec.emotion_mode == "text",
                "emo_text": spec.emo_text.strip() if spec.emo_text else None,
                "use_random": spec.use_random,
                "interval_silence": spec.generation.interval_silence,
                "verbose": spec.generation.verbose,
                "max_text_tokens_per_segment": spec.generation.max_text_tokens_per_segment,
                "do_sample": spec.generation.do_sample,
                "top_p": spec.generation.top_p,
                "top_k": spec.generation.top_k if spec.generation.top_k > 0 else None,
                "temperature": spec.generation.temperature,
                "length_penalty": spec.generation.length_penalty,
                "num_beams": spec.generation.num_beams,
                "repetition_penalty": spec.generation.repetition_penalty,
                "max_mel_tokens": spec.generation.max_mel_tokens,
            }

            self.update_progress(task_id, 0.02, "waiting for inference slot...")
            async with self.infer_lock:
                previous_progress = tts.gr_progress
                tts.gr_progress = JobProgress(self, task_id)
                try:
                    result = await asyncio.to_thread(tts.infer, **infer_kwargs)
                finally:
                    tts.gr_progress = previous_progress

            if result is None or not output_path.exists():
                raise RuntimeError("Inference failed to produce audio")
            elapsed = time.perf_counter() - started
            self._set_status(
                task_id,
                status_value="succeeded",
                progress=1.0,
                message="completed",
                output_path=output_path,
                duration_seconds=get_wav_duration(output_path),
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self._set_status(
                task_id,
                status_value="failed",
                progress=1.0,
                message="failed",
                error=str(exc),
                elapsed_seconds=elapsed,
            )


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


def task_response(request: Request, task: TaskRecord) -> TaskRecord:
    audio_url = str(request.url_for("get_task_audio", task_id=task.task_id))
    return task.model_copy(update={"audio_url": audio_url})


def submit_response(request: Request, task: TaskRecord) -> JobSubmitResponse:
    return JobSubmitResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        message=task.message,
        task_url=str(request.url_for("get_task", task_id=task.task_id)),
        events_url=str(request.url_for("get_task_events", task_id=task.task_id)),
        audio_url=str(request.url_for("get_task_audio", task_id=task.task_id)),
        queue_position=task.queue_position,
    )


def sse_message(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()
    service = IndexTTSService(settings)
    require_api_key = create_auth_dependency(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service
        await service.start()
        if not settings.lazy_load:
            await service.ensure_loaded()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(
        title="IndexTTS2 API Server",
        version="2.0.0",
        description="Async FastAPI service wrapper for IndexTTS2 zero-shot speech synthesis.",
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
            "queue_size": service.queue.qsize() if service.queue is not None else 0,
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
            "queue_size": service.queue.qsize() if service.queue is not None else 0,
            "max_queue_size": settings.max_queue_size,
            "tasks": len(service.jobs),
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
            "max_queue_size": settings.max_queue_size,
        }

    @app.post("/v1/voices", response_model=VoiceAsset, dependencies=[Depends(require_api_key)])
    async def create_voice(audio: UploadFile = File(...)) -> VoiceAsset:
        return await service.create_voice_from_upload(audio)

    @app.post("/v1/voices/json", response_model=VoiceAsset, dependencies=[Depends(require_api_key)])
    async def create_voice_json(body: VoiceJsonRequest) -> VoiceAsset:
        return service.create_voice_from_base64(body.audio_base64, body.filename, body.content_type)

    @app.get("/v1/voices", response_model=list[VoiceAsset], dependencies=[Depends(require_api_key)])
    async def list_voices() -> list[VoiceAsset]:
        return service.list_voices()

    @app.get("/v1/voices/{voice_id}", response_model=VoiceAsset, dependencies=[Depends(require_api_key)])
    async def get_voice(voice_id: str) -> VoiceAsset:
        return service.get_voice(voice_id)

    @app.delete("/v1/voices/{voice_id}", response_model=VoiceAsset, dependencies=[Depends(require_api_key)])
    async def delete_voice(voice_id: str) -> VoiceAsset:
        return service.delete_voice(voice_id)

    @app.post(
        "/v1/tts/jobs",
        response_model=JobSubmitResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key)],
    )
    async def create_tts_job_multipart(
        request: Request,
        text: str = Form(...),
        prompt_voice_id: str | None = Form(None),
        prompt_audio: UploadFile | None = File(None),
        prompt_audio_path: str | None = Form(None),
        emotion_mode: str = Form("speaker"),
        emo_voice_id: str | None = Form(None),
        emo_audio: UploadFile | None = File(None),
        emo_audio_path: str | None = Form(None),
        emo_alpha: float = Form(1.0),
        emo_vector: str | None = Form(None),
        emo_text: str | None = Form(None),
        use_random: bool = Form(False),
        normalize_emo_vector: bool = Form(True),
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
    ) -> JobSubmitResponse:
        if prompt_voice_id:
            prompt_path = service.resolve_voice_path(prompt_voice_id)
        elif prompt_audio is not None:
            prompt_asset = await service.create_voice_from_upload(prompt_audio)
            prompt_voice_id = prompt_asset.voice_id
            prompt_path = service.resolve_voice_path(prompt_voice_id)
        else:
            prompt_path = service.resolve_local_audio_path(prompt_audio_path, "prompt_audio_path")
        if prompt_path is None:
            raise HTTPException(status_code=400, detail="prompt_voice_id or prompt_audio is required")

        if emo_voice_id:
            emotion_path = service.resolve_voice_path(emo_voice_id)
        elif emo_audio is not None:
            emotion_asset = await service.create_voice_from_upload(emo_audio)
            emo_voice_id = emotion_asset.voice_id
            emotion_path = service.resolve_voice_path(emo_voice_id)
        else:
            emotion_path = service.resolve_local_audio_path(emo_audio_path, "emo_audio_path")

        spec = JobSpec(
            text=text,
            prompt_audio_path=str(prompt_path),
            prompt_voice_id=prompt_voice_id,
            emotion_mode=emotion_mode,
            emo_audio_path=str(emotion_path) if emotion_path else None,
            emo_voice_id=emo_voice_id,
            emo_alpha=emo_alpha,
            emo_vector=parse_emo_vector(emo_vector),
            emo_text=emo_text,
            use_random=use_random,
            normalize_emo_vector=normalize_emo_vector,
            generation=GenerationParams(
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
            ),
        )
        task = await service.enqueue(spec)
        return submit_response(request, task)

    @app.post(
        "/v1/tts/jobs/json",
        response_model=JobSubmitResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key)],
    )
    async def create_tts_job_json(request: Request, body: TTSJsonRequest) -> JobSubmitResponse:
        if body.prompt_voice_id:
            prompt_path = service.resolve_voice_path(body.prompt_voice_id)
        elif body.prompt_audio_base64:
            prompt_asset = service.create_voice_from_base64(body.prompt_audio_base64, body.prompt_audio_filename)
            body.prompt_voice_id = prompt_asset.voice_id
            prompt_path = service.resolve_voice_path(body.prompt_voice_id)
        else:
            prompt_path = service.resolve_local_audio_path(body.prompt_audio_path, "prompt_audio_path")
        if prompt_path is None:
            raise HTTPException(status_code=400, detail="prompt_voice_id or prompt_audio_base64 is required")

        emotion_path = None
        if body.emo_voice_id:
            emotion_path = service.resolve_voice_path(body.emo_voice_id)
        elif body.emo_audio_base64:
            emotion_asset = service.create_voice_from_base64(body.emo_audio_base64, body.emo_audio_filename)
            body.emo_voice_id = emotion_asset.voice_id
            emotion_path = service.resolve_voice_path(body.emo_voice_id)
        elif body.emo_audio_path:
            emotion_path = service.resolve_local_audio_path(body.emo_audio_path, "emo_audio_path")

        spec = JobSpec(
            text=body.text,
            prompt_audio_path=str(prompt_path),
            prompt_voice_id=body.prompt_voice_id,
            emotion_mode=body.emotion_mode,
            emo_audio_path=str(emotion_path) if emotion_path else None,
            emo_voice_id=body.emo_voice_id,
            emo_alpha=body.emo_alpha,
            emo_vector=body.emo_vector,
            emo_text=body.emo_text,
            use_random=body.use_random,
            normalize_emo_vector=body.normalize_emo_vector,
            generation=body,
        )
        task = await service.enqueue(spec)
        return submit_response(request, task)

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

    @app.get("/v1/tasks", response_model=list[TaskRecord], dependencies=[Depends(require_api_key)])
    async def list_tasks(request: Request, limit: int = 50) -> list[TaskRecord]:
        return [task_response(request, task) for task in service.list_tasks(max(1, min(limit, 200)))]

    @app.get("/v1/tasks/{task_id}", response_model=TaskRecord, name="get_task", dependencies=[Depends(require_api_key)])
    async def get_task(request: Request, task_id: str) -> TaskRecord:
        return task_response(request, service.get_task(task_id))

    @app.get("/v1/tasks/{task_id}/events", name="get_task_events", dependencies=[Depends(require_api_key)])
    async def get_task_events(request: Request, task_id: str) -> StreamingResponse:
        service.get_task(task_id)

        async def event_stream():
            last_payload = None
            while True:
                if await request.is_disconnected():
                    break
                task = task_response(request, service.get_task(task_id))
                payload = task.model_dump()
                if payload != last_payload:
                    yield sse_message("task", payload)
                    last_payload = payload
                if task.status in TERMINAL_STATUSES:
                    yield sse_message("done", payload)
                    break
                await asyncio.sleep(1.0)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/v1/tasks/{task_id}/audio", name="get_task_audio", dependencies=[Depends(require_api_key)])
    async def get_task_audio(task_id: str) -> FileResponse:
        task = service.get_task(task_id)
        if task.status != "succeeded" or not task.output_path:
            raise HTTPException(status_code=409, detail=f"Task is not ready: {task.status}")
        output_path = Path(task.output_path)
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Task audio file not found")
        return FileResponse(output_path, media_type="audio/wav", filename=f"{task.task_id}.wav")

    @app.delete("/v1/tasks/{task_id}", dependencies=[Depends(require_api_key)])
    async def delete_task(task_id: str) -> dict[str, Any]:
        task = service.get_task(task_id)
        if task.status == "running":
            raise HTTPException(status_code=409, detail="Running tasks cannot be deleted")
        with service.job_lock:
            if task.status == "queued":
                service._set_status(task_id, status_value="cancelled", progress=1.0, message="cancelled")
                try:
                    service.queued_ids.remove(task_id)
                except ValueError:
                    pass
            service.jobs.pop(task_id, None)
            service.job_specs.pop(task_id, None)
        if task.output_path:
            Path(task.output_path).unlink(missing_ok=True)
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
    parser.add_argument("--max_queue_size", type=int, default=_env_int("INDEXTTS_MAX_QUEUE_SIZE", 100))
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
        max_queue_size=args.max_queue_size,
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
        os.environ["INDEXTTS_MAX_QUEUE_SIZE"] = str(settings.max_queue_size)
        uvicorn.run("indextts.api_server:create_app", factory=True, host=settings.host, port=settings.port, reload=True)
    else:
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
