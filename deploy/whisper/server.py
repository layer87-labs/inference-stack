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
  prompt           optional initial prompt (domain vocabulary, spelling hints);
                   appended to WHISPER_INITIAL_PROMPT, see combine_prompt()
  hotwords         optional hint phrases; replaces WHISPER_HOTWORDS
  response_format  json (default) | text | verbose_json | srt | vtt
  temperature      float; when omitted the server uses faster-whisper's temperature
                   fallback (0.0 .. 1.0, see TEMPERATURES). An explicit value
                   (including 0) is used as given, without fallback.

Long audio (longer than WHISPER_BATCH_THRESHOLD_S) goes through faster-whisper's
BatchedInferencePipeline, short audio through the regular transcribe(). The
sequential decoder can fall into repetition loops on long recordings because
every window is conditioned on the previous text; the batched pipeline decodes
VAD-cut chunks independently and is also faster on long audio. The batched
pipeline always uses VAD and only the first temperature (no fallback; it has no
previous-text conditioning to loop on).

Inference runs in a worker thread and is serialised by a semaphore, so the
event loop (and with it /health) stays responsive while a transcription runs.

Every request writes one JSON log line ("whisper_request {...}") with timings,
audio duration, real-time factor and parameters. Prompt, hotwords and
transcript are never logged, only whether and how long the prompt/hotwords
were.

At startup the model is loaded and one short synthetic clip is run through the
pipeline (VAD and decoder) so the first real request does not pay the
initialisation cost; /health reports ready only after that.
"""

import asyncio
import json
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
# Audio longer than this (seconds) is transcribed with BatchedInferencePipeline.
BATCH_THRESHOLD_S = float(os.environ.get("WHISPER_BATCH_THRESHOLD_S", "35"))
BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))
# Off by default: conditioning every window on the previous text is what lets
# a repetition loop carry on.
CONDITION_ON_PREVIOUS_TEXT = os.environ.get("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false").lower() == "true"
WARMUP = os.environ.get("WHISPER_WARMUP", "true").lower() == "true"
SAMPLE_RATE = 16000
# faster-whisper's own fallback schedule; thresholds (compression ratio 2.4,
# log-prob -1.0, no-speech 0.6) stay at its defaults.
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Server-side defaults; many STT clients send neither prompt nor hotwords.
DEFAULT_PROMPT = os.environ.get("WHISPER_INITIAL_PROMPT", "").strip()
DEFAULT_HOTWORDS = os.environ.get("WHISPER_HOTWORDS", "").strip()
# Upper bound on input size per field. faster-whisper itself keeps only the last
# 223 tokens (max_length 448 // 2 - 1) of prompt and hotwords; this cap only
# protects against absurd inputs and also keeps the tail.
MAX_PROMPT_CHARS = 2000

FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}

_model = None
_batched = None
_ready = False
_sem: asyncio.Semaphore | None = None


def _load_model():
    from faster_whisper import WhisperModel

    print(f"Loading whisper model: {MODEL_PATH} (compute_type={COMPUTE_TYPE}, cpu_threads={CPU_THREADS})", flush=True)
    t0 = time.monotonic()
    model = WhisperModel(MODEL_PATH, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)
    print(f"Model loaded in {time.monotonic() - t0:.1f}s", flush=True)
    return model


def _load_batched(model):
    from faster_whisper import BatchedInferencePipeline

    return BatchedInferencePipeline(model=model)


def _warmup_audio():
    """Two seconds of a quiet tone plus noise (deterministic)."""
    import numpy as np

    n = 2 * SAMPLE_RATE
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    noise = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    return (0.05 * np.sin(2 * np.pi * 220 * t) + 0.02 * noise).astype(np.float32)


def _warmup():
    """Run a short clip through the pipeline and discard the result.

    Pass 1 uses the configured VAD setting (initialises the VAD runtime), pass 2
    runs with VAD off so the encoder and decoder are exercised even when the
    VAD drops the synthetic audio. Failures are logged, never raised.
    """
    t0 = time.monotonic()
    try:
        audio = _warmup_audio()
        for vad in dict.fromkeys((VAD_FILTER, False)):
            _transcribe(audio, 2.0, "transcribe", DEFAULT_LANGUAGE, None, None, None, vad=vad)
        print(f"Warmup done in {time.monotonic() - t0:.1f}s", flush=True)
    except Exception as exc:
        print(f"Warmup failed ({type(exc).__name__}: {exc}); continuing", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _batched, _ready, _sem
    _sem = asyncio.Semaphore(MAX_CONCURRENT)
    _model = await asyncio.to_thread(_load_model)
    _batched = await asyncio.to_thread(_load_batched, _model)
    if WARMUP:
        await asyncio.to_thread(_warmup)
    _ready = True
    yield


app = FastAPI(title="whisper", lifespan=lifespan)


@app.get("/health")
async def health():
    if _model is None or not _ready:
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


def combine_prompt(default: str, request: str | None) -> tuple[str | None, str]:
    """Merge the configured default prompt with a request prompt.

    The request prompt is appended after the default. Whisper conditions most
    strongly on the end of the prompt, and faster-whisper cuts an over-long
    prompt from the front (it keeps the last 223 tokens), so the more specific
    request text survives and the generic default is what gets cut first.
    Returns (prompt or None, source) with source in none|default|request|both.
    """
    request = (request or "").strip()
    parts = [p for p in (default, request) if p]
    if not parts:
        return None, "none"
    source = "both" if default and request else ("default" if default else "request")
    return " ".join(parts)[-MAX_PROMPT_CHARS:], source


def resolve_hotwords(default: str, request: str | None) -> str | None:
    """A request's hotwords replace the default (no merging)."""
    return ((request or "").strip() or default)[:MAX_PROMPT_CHARS] or None


def _decode(path: str):
    """Decode to 16 kHz mono float32 (PyAV/FFmpeg). Returns (audio, duration_s)."""
    from faster_whisper.audio import decode_audio

    audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
    return audio, len(audio) / SAMPLE_RATE


def _transcribe(audio, duration_s: float, task: str, language: str | None, prompt: str | None,
                hotwords: str | None, temperature: float | None, vad: bool | None = None):
    """Transcribe decoded audio; returns (segments, info, timing, meta).

    temperature None = faster-whisper's fallback schedule (TEMPERATURES); a
    number is used as given.
    """
    vad = VAD_FILTER if vad is None else vad
    batched = duration_s > BATCH_THRESHOLD_S and _batched is not None
    kwargs = dict(
        task=task,
        language=language,
        initial_prompt=prompt,
        hotwords=hotwords,
        temperature=TEMPERATURES if temperature is None else temperature,
        beam_size=BEAM_SIZE,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
    )
    t0 = time.monotonic()
    # transcribe() runs VAD and builds the features before it returns the lazy
    # generator.
    if batched:
        # The batched pipeline requires VAD (it cuts the chunks).
        vad = True
        segments, info = _batched.transcribe(audio, vad_filter=True, batch_size=BATCH_SIZE, **kwargs)
    else:
        segments, info = _model.transcribe(audio, vad_filter=vad, **kwargs)
    t1 = time.monotonic()
    # The actual decoding happens while iterating.
    segments = list(segments)
    t2 = time.monotonic()
    meta = {"path": "batched" if batched else "standard", "vad": vad,
            "temperature_mode": "fixed" if temperature is not None else "fallback"}
    return segments, info, {"prep_s": t1 - t0, "infer_s": t2 - t1}, meta


def _log_request(**fields):
    print("whisper_request " + json.dumps(fields, separators=(",", ":")), flush=True)


async def _handle(file: UploadFile, task: str, language: str | None, prompt: str | None,
                  hotwords: str | None, response_format: str, temperature: float | None):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    if response_format not in FORMATS:
        raise HTTPException(status_code=400, detail=f"response_format must be one of {sorted(FORMATS)}")
    language = (language or DEFAULT_LANGUAGE) or None
    prompt, prompt_source = combine_prompt(DEFAULT_PROMPT, prompt)
    hotwords = resolve_hotwords(DEFAULT_HOTWORDS, hotwords)
    t_start = time.monotonic()

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

        t_queued = time.monotonic()
        async with _sem:
            t_run = time.monotonic()
            try:
                t_dec = time.monotonic()
                audio, duration_s = await asyncio.to_thread(_decode, tmp.name)
                decode_s = time.monotonic() - t_dec
                segments, info, timing, meta = await asyncio.to_thread(
                    _transcribe, audio, duration_s, task, language, prompt, hotwords, temperature)
                timing["prep_s"] += decode_s
                del audio
            except Exception as exc:  # undecodable audio etc.
                _log_request(status="error", error=type(exc).__name__, task=task, file_bytes=size,
                             total_s=round(time.monotonic() - t_start, 3))
                raise HTTPException(status_code=400, detail=f"could not process audio: {exc}") from exc

    total_s = time.monotonic() - t_start
    work_s = timing["prep_s"] + timing["infer_s"]
    audio_s = info.duration or 0.0
    _log_request(
        status="ok",
        task=task,
        model=MODEL_NAME,
        language=info.language,
        language_forced=language is not None,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        beam_size=BEAM_SIZE,
        vad=meta["vad"],
        path=meta["path"],
        batch_size=BATCH_SIZE if meta["path"] == "batched" else None,
        temperature_mode=meta["temperature_mode"],
        temperature=temperature if temperature is not None else TEMPERATURES,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        prompt_chars=len(prompt or ""),
        prompt_source=prompt_source,
        hotwords_chars=len(hotwords or ""),
        response_format=response_format,
        file_bytes=size,
        audio_s=round(audio_s, 2),
        audio_after_vad_s=round(getattr(info, "duration_after_vad", audio_s) or 0.0, 2),
        upload_s=round(t_queued - t_start, 3),
        queue_s=round(t_run - t_queued, 3),
        prep_s=round(timing["prep_s"], 3),
        infer_s=round(timing["infer_s"], 3),
        total_s=round(total_s, 3),
        rtf=round(work_s / audio_s, 3) if audio_s else None,
    )

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
    hotwords: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float | None = Form(None),
):
    return await _handle(file, "transcribe", language, prompt, hotwords, response_format, temperature)


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    prompt: str | None = Form(None),
    hotwords: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float | None = Form(None),
):
    return await _handle(file, "translate", None, prompt, hotwords, response_format, temperature)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
