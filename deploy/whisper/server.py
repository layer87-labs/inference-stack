"""Whisper ASR HTTP server — OpenAI-compatible /v1/audio/transcriptions.

Exposes:
  POST /v1/audio/transcriptions  — transcribe an audio file
  POST /v1/audio/translations    — translate an audio file to English
  GET  /v1/models                — list the loaded model
  GET  /health                   — liveness/readiness (never queues behind inference)

Request (multipart/form-data), OpenAI-compatible subset:
  file             audio file (mp3, wav, m4a, ogg, flac, webm, mp4, ...)
  model            accepted for compatibility, ignored (one model is loaded)
  language         ISO-639-1 code; empty = auto-detect (default: WHISPER_LANGUAGE)
  prompt           optional initial prompt
  response_format  json (default) | text | verbose_json | srt | vtt
  temperature      float, default 0

Inference runs in a worker thread and is serialised by a semaphore, so the
event loop (and with it /health) stays responsive while a transcription runs.
"""

import asyncio
import os
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model")
MODEL_NAME = os.environ.get("MODEL_NAME", "whisper-large-v3-turbo")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
CPU_THREADS = int(os.environ.get("CPU_THREADS", "0"))  # 0 = CTranslate2 default
DEFAULT_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "") or None
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "1"))
VAD_FILTER = os.environ.get("VAD_FILTER", "true").lower() == "true"
PORT = int(os.environ.get("PORT", "9000"))

FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}

_model = None
_sem: asyncio.Semaphore | None = None


def _load_model():
    from faster_whisper import WhisperModel

    print(f"Loading whisper model: {MODEL_PATH} (compute_type={COMPUTE_TYPE}, cpu_threads={CPU_THREADS})", flush=True)
    t0 = time.monotonic()
    model = WhisperModel(MODEL_PATH, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)
    print(f"Model loaded in {time.monotonic() - t0:.1f}s", flush=True)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _sem
    _sem = asyncio.Semaphore(MAX_CONCURRENT)
    _model = await asyncio.to_thread(_load_model)
    yield


app = FastAPI(title="whisper", lifespan=lifespan)


@app.get("/health")
async def health():
    if _model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "inference-stack"}],
    }


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _transcribe(path: str, task: str, language: str | None, prompt: str | None, temperature: float):
    segments, info = _model.transcribe(
        path,
        task=task,
        language=language,
        initial_prompt=prompt or None,
        temperature=temperature,
        beam_size=BEAM_SIZE,
        vad_filter=VAD_FILTER,
    )
    # transcribe() is lazy: decoding happens while iterating.
    return list(segments), info


async def _handle(file: UploadFile, task: str, language: str | None, prompt: str | None,
                  response_format: str, temperature: float):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    if response_format not in FORMATS:
        raise HTTPException(status_code=400, detail=f"response_format must be one of {sorted(FORMATS)}")
    language = (language or DEFAULT_LANGUAGE) or None

    limit = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename or "")[1] or ".audio") as tmp:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(status_code=413, detail=f"file exceeds {MAX_UPLOAD_MB} MB")
            tmp.write(chunk)
        tmp.flush()
        if size == 0:
            raise HTTPException(status_code=400, detail="empty file")

        async with _sem:
            try:
                segments, info = await asyncio.to_thread(_transcribe, tmp.name, task, language, prompt, temperature)
            except Exception as exc:  # undecodable audio etc.
                raise HTTPException(status_code=400, detail=f"could not process audio: {exc}") from exc

    text = "".join(s.text for s in segments).strip()
    if response_format == "json":
        return {"text": text}
    if response_format == "text":
        return PlainTextResponse(text + "\n")
    if response_format == "verbose_json":
        return {
            "task": task,
            "language": info.language,
            "duration": info.duration,
            "text": text,
            "segments": [
                {"id": i, "start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob,
                 "no_speech_prob": s.no_speech_prob}
                for i, s in enumerate(segments)
            ],
        }
    if response_format == "srt":
        out = [f"{i}\n{_ts(s.start, ',')} --> {_ts(s.end, ',')}\n{s.text.strip()}\n" for i, s in enumerate(segments, 1)]
        return PlainTextResponse("\n".join(out), media_type="application/x-subrip")
    out = ["WEBVTT\n"] + [f"{_ts(s.start, '.')} --> {_ts(s.end, '.')}\n{s.text.strip()}\n" for s in segments]
    return PlainTextResponse("\n".join(out), media_type="text/vtt")


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    return await _handle(file, "transcribe", language, prompt, response_format, temperature)


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    return await _handle(file, "translate", None, prompt, response_format, temperature)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
