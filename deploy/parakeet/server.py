"""Parakeet ASR HTTP server — OpenAI-compatible /v1/audio/transcriptions.

A fast CPU path for short recordings (dictation): NVIDIA Parakeet-TDT-0.6B-v3
(int8 ONNX export) on ONNX Runtime via onnx-asr. Whisper stays the backend for
long or quality-critical audio; the router picks the backend by model id.

Exposes:
  POST /v1/audio/transcriptions  — transcribe an audio file
  GET  /v1/models                — list the loaded model
  GET  /health                   — liveness/readiness (never queues behind inference)

Request (multipart/form-data), OpenAI-compatible subset:
  file             audio file (wav, mp3, ogg/opus, webm, mp4/m4a, flac, ...),
                   decoded with PyAV (FFmpeg libraries bundled in the wheel)
  model            accepted for compatibility, ignored (one model is loaded)
  language         accepted and echoed in verbose_json; Parakeet-v3 detects the
                   language itself (25 European languages), it cannot be forced
  prompt           accepted and ignored: the model has no prompt or hotword input
  response_format  json (default) | text | verbose_json | srt | vtt
  temperature      accepted and ignored (greedy TDT decoding)

verbose_json carries `segments` (split at sentence punctuation) and `words`.
Both have start/end from the model's token timestamps (80 ms frames). The
confidence of a word is exp(mean log-probability of its tokens) taken from the
decoder's joint network: a relative score in 0..1, not a calibrated
probability.

Audio longer than CHUNK_S is cut into chunks at the quietest 100 ms within the
last CHUNK_SEARCH_S seconds before each chunk boundary (energy, no VAD model)
and the chunks are transcribed one after another. Audio longer than
MAX_AUDIO_S is rejected with 413: this backend is the short-dictation path.

Admission happens before the upload is read: an ASGI middleware answers 429
(with Retry-After) when MAX_CONCURRENT + MAX_QUEUE requests are already in
the server, and 413 when the body exceeds MAX_UPLOAD_MB, by Content-Length or
by counting the bytes as they arrive. So at most MAX_CONCURRENT + MAX_QUEUE
uploads are ever spooled to /tmp, each at most MAX_UPLOAD_MB, and the upload
is decoded from that single spool file without a second copy. Inference runs
in a worker thread, at most MAX_CONCURRENT at a time. Decoding and inference
stop with 504 after REQUEST_TIMEOUT_S (checked while decoding and between
chunks).

Errors use the OpenAI shape: {"error": {"message", "type", "code"}}.

Every request writes one JSON log line ("parakeet_request {...}") with sizes,
timings and real-time factor. Audio and transcript are never logged.

At startup the model is loaded and a short synthetic clip is transcribed so the
first real request does not pay the initialisation cost; /health reports ready
only after that.
"""

import asyncio
import json
import math
import os
import re
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model")
MODEL_NAME = os.environ.get("MODEL_NAME", "parakeet-tdt-0.6b-v3")
# onnx-asr model type of the files in MODEL_PATH.
MODEL_TYPE = os.environ.get("MODEL_TYPE", "nemo-parakeet-tdt-0.6b-v3")
QUANTIZATION = os.environ.get("QUANTIZATION", "int8") or None
CPU_THREADS = int(os.environ.get("CPU_THREADS", "0"))  # 0 = ONNX Runtime default
DEFAULT_LANGUAGE = os.environ.get("PARAKEET_LANGUAGE", "") or None
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
MAX_AUDIO_S = float(os.environ.get("MAX_AUDIO_S", "120"))
CHUNK_S = float(os.environ.get("CHUNK_S", "30"))
CHUNK_SEARCH_S = float(os.environ.get("CHUNK_SEARCH_S", "5"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "4"))
RETRY_AFTER_S = os.environ.get("RETRY_AFTER_S", "2")
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "60"))
WARMUP = os.environ.get("PARAKEET_WARMUP", "true").lower() == "true"
PORT = int(os.environ.get("PORT", "9000"))

SAMPLE_RATE = 16000
# Shorter than this is treated as no audio at all.
MIN_AUDIO_S = 0.1
# Duration of one encoder frame (10 ms hop x subsampling 8): the resolution of
# token timestamps and the end of the last token.
FRAME_S = 0.08
FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}
TRANSCRIBE_PATH = "/v1/audio/transcriptions"
# Room for the multipart framing and the small form fields next to the file.
MULTIPART_SLACK = 64 * 1024
SENTENCE_END = re.compile(r"[.!?…]$")

_model = None
_ready = False
_sem: asyncio.Semaphore | None = None
_inflight = 0  # admitted requests: uploading, waiting or running


class AudioTooLong(Exception):
    pass


class Deadline(Exception):
    pass


class ApiError(Exception):
    """An error answered in the OpenAI error shape."""

    def __init__(self, status: int, message: str, code: str, headers: dict | None = None):
        super().__init__(message)
        self.status, self.message, self.code, self.headers = status, message, code, headers


def error_body(status: int, message: str, code: str) -> dict:
    kind = "server_error" if status >= 500 else ("rate_limit_error" if status == 429 else "invalid_request_error")
    return {"error": {"message": message, "type": kind, "code": code}}


def _load_model():
    import onnx_asr
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if CPU_THREADS > 0:
        opts.intra_op_num_threads = CPU_THREADS
        opts.inter_op_num_threads = 1
    print(f"Loading parakeet model: {MODEL_PATH} (type={MODEL_TYPE}, quantization={QUANTIZATION}, "
          f"cpu_threads={CPU_THREADS})", flush=True)
    t0 = time.monotonic()
    model = onnx_asr.load_model(MODEL_TYPE, MODEL_PATH, quantization=QUANTIZATION, sess_options=opts,
                                providers=["CPUExecutionProvider"]).with_timestamps()
    print(f"Model loaded in {time.monotonic() - t0:.1f}s", flush=True)
    return model


def _warmup_audio():
    """Two seconds of a quiet tone plus noise (deterministic)."""
    n = 2 * SAMPLE_RATE
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    noise = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    return (0.05 * np.sin(2 * np.pi * 220 * t) + 0.02 * noise).astype(np.float32)


def _warmup():
    t0 = time.monotonic()
    try:
        _model.recognize(_warmup_audio())
        print(f"Warmup done in {time.monotonic() - t0:.1f}s", flush=True)
    except Exception as exc:
        print(f"Warmup failed ({type(exc).__name__}); continuing", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _ready, _sem
    _sem = asyncio.Semaphore(MAX_CONCURRENT)
    _model = await asyncio.to_thread(_load_model)
    if WARMUP:
        await asyncio.to_thread(_warmup)
    _ready = True
    yield


app = FastAPI(title="parakeet", lifespan=lifespan)


@app.exception_handler(ApiError)
async def _api_error(request, exc: ApiError):
    return JSONResponse(error_body(exc.status, exc.message, exc.code), status_code=exc.status, headers=exc.headers)


@app.exception_handler(HTTPException)
async def _http_error(request, exc: HTTPException):
    return JSONResponse(error_body(exc.status_code, str(exc.detail), "http_error"), status_code=exc.status_code,
                        headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def _validation_error(request, exc: RequestValidationError):
    return JSONResponse(error_body(400, "invalid request: missing or malformed form fields", "invalid_request"),
                        status_code=400)


class AdmissionMiddleware:
    """Admit or reject transcription requests before their body is parsed.

    Starlette spools a multipart upload to disk before the endpoint runs, so
    the endpoint is too late to protect /tmp. This middleware counts requests
    in the server (429 beyond MAX_CONCURRENT + MAX_QUEUE) and caps the body
    (413 by Content-Length, or once the bytes received exceed the limit).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        global _inflight
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != TRANSCRIBE_PATH:
            await self.app(scope, receive, send)
            return
        limit = MAX_UPLOAD_MB * 1024 * 1024 + MULTIPART_SLACK
        headers = dict(scope.get("headers") or [])
        if _inflight >= MAX_CONCURRENT + MAX_QUEUE:
            _log_request(status="error", error="overloaded", http_status=429)
            await _send_json(send, 429, error_body(429, "server busy, retry later", "overloaded"),
                             [(b"retry-after", RETRY_AFTER_S.encode())])
            return
        try:
            declared = int(headers.get(b"content-length", b"-1"))
        except ValueError:
            declared = -1
        if declared > limit:
            await self._too_large(send, declared)
            return

        received = 0
        too_large = False

        async def limited_receive():
            nonlocal received, too_large
            if too_large:
                return {"type": "http.request", "body": b"", "more_body": False}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    too_large = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        started = False

        async def guarded_send(message):
            # Once the body was cut off, whatever the app answers (a parse
            # error) is replaced by the 413.
            nonlocal started
            if too_large:
                if not started:
                    started = True
                    await self._too_large(send, received)
                return
            await send(message)

        _inflight += 1
        try:
            await self.app(scope, limited_receive, guarded_send)
        finally:
            _inflight -= 1

    @staticmethod
    async def _too_large(send, size: int):
        _log_request(status="error", error="upload_too_large", http_status=413, file_bytes=size)
        await _send_json(send, 413, error_body(413, f"file exceeds {MAX_UPLOAD_MB} MB", "upload_too_large"))


async def _send_json(send, status: int, body: dict, extra: list | None = None):
    data = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]
                + (extra or [])})
    await send({"type": "http.response.body", "body": data})


app.add_middleware(AdmissionMiddleware)


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


def decode(source, max_s: float, deadline: float | None = None) -> np.ndarray:
    """Decode any FFmpeg-readable file (path or file object) to 16 kHz mono float32.

    Raises AudioTooLong as soon as more than max_s seconds are decoded, so an
    over-long upload is not decoded to the end, and Deadline once
    time.monotonic() passes deadline.
    """
    import av

    limit = int(max_s * SAMPLE_RATE)
    parts: list[np.ndarray] = []
    total = 0
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    with av.open(source, mode="r") as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("no audio stream")
        frames = container.decode(stream)
        for frame in frames:
            for out in resampler.resample(frame):
                parts.append(out.to_ndarray().reshape(-1))
                total += out.samples
            if total > limit:
                raise AudioTooLong()
            if deadline is not None and time.monotonic() > deadline:
                raise Deadline()
        for out in resampler.resample(None):
            parts.append(out.to_ndarray().reshape(-1))
            total += out.samples
    if total > limit:
        raise AudioTooLong()
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def split_points(audio: np.ndarray, chunk_s: float, search_s: float, sr: int = SAMPLE_RATE) -> list[int]:
    """Sample offsets where to cut audio into chunks of at most chunk_s.

    Each cut lies at the quietest 100 ms window within the last search_s
    seconds before the chunk limit. Returns the start offsets, beginning with 0.
    """
    n = len(audio)
    max_len = int(chunk_s * sr)
    search = max(1, min(int(search_s * sr), max_len // 2))
    win = sr // 10
    starts = [0]
    while n - starts[-1] > max_len:
        lo = starts[-1] + max_len - search
        region = audio[lo:starts[-1] + max_len]
        k = len(region) // win
        if k == 0:
            cut = starts[-1] + max_len
        else:
            energy = (region[:k * win].reshape(k, win) ** 2).mean(axis=1)
            cut = lo + int(np.argmin(energy)) * win + win // 2
        starts.append(cut)
    return starts


def words_from_tokens(tokens: list[str], timestamps: list[float], logprobs: list[float] | None,
                      offset: float = 0.0) -> list[dict]:
    """Group SentencePiece tokens into words ("▁" starts a word)."""
    words: list[dict] = []
    cur = None
    for i, tok in enumerate(tokens):
        start = timestamps[i] + offset
        lp = logprobs[i] if logprobs is not None else None
        if cur is None or tok.startswith("▁") or tok.startswith(" "):
            if cur is not None:
                words.append(cur)
            cur = {"word": tok.lstrip("▁ "), "start": start, "end": start + FRAME_S, "lps": []}
        else:
            cur["word"] += tok
            cur["end"] = start + FRAME_S
        if lp is not None:
            cur["lps"].append(lp)
    if cur is not None:
        words.append(cur)
    out = []
    for w in words:
        if not w["word"]:
            continue
        item = {"word": w["word"], "start": round(w["start"], 2), "end": round(w["end"], 2)}
        if w["lps"]:
            item["confidence"] = round(math.exp(sum(w["lps"]) / len(w["lps"])), 3)
            item["logprob"] = round(sum(w["lps"]) / len(w["lps"]), 4)
        out.append(item)
    return out


def segments_from_words(words: list[dict]) -> list[dict]:
    """Group words into segments ending at sentence punctuation."""
    segments: list[dict] = []
    cur: list[dict] = []
    for w in words:
        cur.append(w)
        if SENTENCE_END.search(w["word"]):
            segments.append(_segment(len(segments), cur))
            cur = []
    if cur:
        segments.append(_segment(len(segments), cur))
    return segments


def _segment(idx: int, words: list[dict]) -> dict:
    seg = {"id": idx, "start": words[0]["start"], "end": words[-1]["end"],
           "text": " ".join(w["word"] for w in words)}
    lps = [w["logprob"] for w in words if "logprob" in w]
    if lps:
        seg["avg_logprob"] = round(sum(lps) / len(lps), 4)
    return seg


def transcribe(audio: np.ndarray, deadline: float | None = None) -> tuple[str, list[dict], int]:
    """Transcribe decoded audio; returns (text, words, number of chunks).

    Raises Deadline before a chunk when time.monotonic() has passed deadline.
    """
    starts = split_points(audio, CHUNK_S, CHUNK_SEARCH_S)
    bounds = starts[1:] + [len(audio)]
    texts: list[str] = []
    words: list[dict] = []
    for lo, hi in zip(starts, bounds):
        if deadline is not None and time.monotonic() > deadline:
            raise Deadline()
        res = _model.recognize(audio[lo:hi])
        text = (res.text or "").strip()
        if text:
            texts.append(text)
        if res.tokens and res.timestamps:
            words.extend(words_from_tokens(res.tokens, res.timestamps, res.logprobs, offset=lo / SAMPLE_RATE))
    return " ".join(texts), words, len(starts)


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _log_request(**fields):
    print("parakeet_request " + json.dumps(fields, separators=(",", ":")), flush=True)


def _fail(status: int, message: str, t_start: float, size: int, code: str, headers: dict | None = None):
    _log_request(status="error", error=code, http_status=status, file_bytes=size,
                 total_s=round(time.monotonic() - t_start, 3))
    raise ApiError(status, message, code, headers)


@app.post(TRANSCRIBE_PATH)
async def transcriptions(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float | None = Form(None),
):
    # The upload is already spooled (and size-checked by AdmissionMiddleware).
    t_start = time.monotonic()
    if _model is None or not _ready:
        _fail(503, "model not loaded yet", t_start, 0, "not_ready")
    if response_format not in FORMATS:
        _fail(400, f"response_format must be one of {sorted(FORMATS)}", t_start, 0, "invalid_response_format")
    size = file.size or 0
    if size == 0:
        _fail(400, "empty file", t_start, 0, "empty_file")

    async with _sem:
        t_run = time.monotonic()
        deadline = t_run + REQUEST_TIMEOUT_S
        try:
            file.file.seek(0)
            # Decode straight from the spooled upload: no second copy on disk.
            audio = await asyncio.to_thread(decode, file.file, MAX_AUDIO_S, deadline)
        except AudioTooLong:
            _fail(413, f"audio longer than {MAX_AUDIO_S:g} s; use a long-form transcription model",
                  t_start, size, "audio_too_long")
        except Deadline:
            _fail(504, f"decoding took longer than {REQUEST_TIMEOUT_S:g} s", t_start, size, "timeout")
        except Exception as exc:
            _fail(400, "could not decode audio", t_start, size, "invalid_audio:" + type(exc).__name__)
        t_dec = time.monotonic()
        audio_s = len(audio) / SAMPLE_RATE
        if audio_s < MIN_AUDIO_S:
            _fail(400, "no audio in file", t_start, size, "no_audio")
        try:
            text, words, chunks = await asyncio.to_thread(transcribe, audio, deadline)
        except Deadline:
            _fail(504, f"transcription took longer than {REQUEST_TIMEOUT_S:g} s", t_start, size, "timeout")
        except Exception as exc:
            _fail(500, "transcription failed", t_start, size, "inference_error:" + type(exc).__name__)
        t_done = time.monotonic()
        del audio

    _log_request(
        status="ok",
        model=MODEL_NAME,
        cpu_threads=CPU_THREADS,
        response_format=response_format,
        prompt_ignored=bool((prompt or "").strip()),
        file_bytes=size,
        audio_s=round(audio_s, 2),
        chunks=chunks,
        words=len(words),
        queue_s=round(t_run - t_start, 3),
        decode_s=round(t_dec - t_run, 3),
        infer_s=round(t_done - t_dec, 3),
        total_s=round(time.monotonic() - t_start, 3),
        rtf=round((t_done - t_run) / audio_s, 3),
    )

    if response_format == "json":
        return {"text": text}
    if response_format == "text":
        return PlainTextResponse(text + "\n")
    segments = segments_from_words(words)
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": (language or DEFAULT_LANGUAGE) or None,
            "duration": round(audio_s, 2),
            "text": text,
            "segments": segments,
            "words": words,
        }
    if response_format == "srt":
        out = [f"{i}\n{_ts(s['start'], ',')} --> {_ts(s['end'], ',')}\n{s['text']}\n"
               for i, s in enumerate(segments, 1)]
        return PlainTextResponse("\n".join(out), media_type="application/x-subrip")
    out = ["WEBVTT\n"] + [f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}\n{s['text']}\n" for s in segments]
    return PlainTextResponse("\n".join(out), media_type="text/vtt")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
