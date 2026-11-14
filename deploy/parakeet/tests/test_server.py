"""Tests for deploy/parakeet/server.py with a fake model (no model download).

Decoding is real (PyAV), so the format tests exercise the same code as the image.
Run: pip install fastapi httpx python-multipart pytest numpy av && pytest deploy/parakeet
"""

import io
import os
import sys
import threading
import time
import wave
from types import SimpleNamespace

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import server  # noqa: E402

SECRET_TEXT = "Geheimer Inhalt."

# Tokens as Parakeet emits them: "▁" starts a word.
TOKENS = ["▁Hallo", "▁Wel", "t", ".", "▁Zwei", "ter", "▁Satz", "."]
TIMES = [0.0, 0.4, 0.56, 0.64, 1.2, 1.36, 1.6, 1.84]
LOGPROBS = [-0.1, -0.2, -0.4, 0.0, -1.0, -1.0, -0.05, 0.0]


class FakeModel:
    def __init__(self):
        self.calls = []
        self.gate = None  # threading.Event: block recognize() until set
        self.delay = 0.0

    def recognize(self, audio):
        self.calls.append(len(audio))
        if self.gate is not None:
            self.gate.wait(10)
        time.sleep(self.delay)
        return SimpleNamespace(text=SECRET_TEXT, tokens=TOKENS, timestamps=TIMES, logprobs=LOGPROBS)


def wav_bytes(seconds: float, rate: int = 44100, channels: int = 2) -> bytes:
    n = int(seconds * rate)
    t = np.arange(n) / rate
    tone = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    data = np.repeat(tone[:, None], channels, axis=1).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)
    return buf.getvalue()


def encoded_bytes(container: str, codec: str, seconds: float = 1.0, rate: int = 48000) -> bytes:
    """Encode a tone with PyAV, e.g. webm/opus as browsers record it."""
    buf = io.BytesIO()
    with av.open(buf, mode="w", format=container) as out:
        stream = out.add_stream(codec, rate=rate, layout="mono")
        n = int(seconds * rate)
        samples = (0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / rate)).astype(np.float32)
        step = 960
        for i in range(0, n, step):
            frame = av.AudioFrame.from_ndarray(samples[None, i:i + step], format="flt", layout="mono")
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(server, "_load_model", lambda: model)
    monkeypatch.setattr(server, "WARMUP", False)
    monkeypatch.setattr(server, "_inflight", 0)
    monkeypatch.setattr(server, "MAX_CONCURRENT", 1)
    monkeypatch.setattr(server, "MAX_QUEUE", 1)
    monkeypatch.setattr(server, "MAX_AUDIO_S", 120.0)
    monkeypatch.setattr(server, "CHUNK_S", 30.0)
    monkeypatch.setattr(server, "REQUEST_TIMEOUT_S", 60.0)
    # The context manager runs the lifespan (fake model, semaphore) and keeps
    # one event loop for all requests, so concurrent requests really queue.
    with TestClient(server.app) as c:
        c.model = model
        yield c


def err(r):
    return r.json()["error"]


def post(client, payload: bytes, name: str = "a.wav", **data):
    return client.post("/v1/audio/transcriptions", files={"file": (name, io.BytesIO(payload), "application/octet-stream")},
                       data=data)


def test_json_response(client):
    r = post(client, wav_bytes(2.0))
    assert r.status_code == 200
    assert r.json() == {"text": SECRET_TEXT}
    # 2 s of 44.1 kHz stereo arrive as 16 kHz mono.
    assert client.model.calls == [32000]


@pytest.mark.parametrize("name,container,codec", [
    ("a.webm", "webm", "libopus"),
    ("a.ogg", "ogg", "libopus"),
    ("a.m4a", "mp4", "aac"),
    ("a.mp3", "mp3", "libmp3lame"),
])
def test_browser_formats_decode(client, name, container, codec):
    if codec not in av.codecs_available:
        pytest.skip(f"{codec} not in this PyAV build")
    r = post(client, encoded_bytes(container, codec), name=name)
    assert r.status_code == 200, r.text
    # 1 s, give or take the codec's padding.
    assert len(client.model.calls) == 1
    assert 15000 <= client.model.calls[0] <= 17500


def test_verbose_json_words_segments(client):
    r = post(client, wav_bytes(2.0), response_format="verbose_json", language="de")
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "de"
    assert body["duration"] == 2.0
    assert body["words"] == [
        {"word": "Hallo", "start": 0.0, "end": 0.08, "confidence": 0.905, "logprob": -0.1},
        {"word": "Welt.", "start": 0.4, "end": 0.72, "confidence": 0.819, "logprob": -0.2},
        {"word": "Zweiter", "start": 1.2, "end": 1.44, "confidence": 0.368, "logprob": -1.0},
        {"word": "Satz.", "start": 1.6, "end": 1.92, "confidence": 0.975, "logprob": -0.025},
    ]
    assert body["segments"] == [
        {"id": 0, "start": 0.0, "end": 0.72, "text": "Hallo Welt.", "avg_logprob": -0.15},
        {"id": 1, "start": 1.2, "end": 1.92, "text": "Zweiter Satz.", "avg_logprob": -0.5125},
    ]


def test_srt(client):
    r = post(client, wav_bytes(2.0), response_format="srt")
    assert r.text == ("1\n00:00:00,000 --> 00:00:00,720\nHallo Welt.\n\n"
                      "2\n00:00:01,200 --> 00:00:01,920\nZweiter Satz.\n")


def test_unknown_format(client):
    r = post(client, wav_bytes(1.0), response_format="xml")
    assert r.status_code == 400
    assert err(r)["code"] == "invalid_response_format"


def test_missing_file_field(client):
    r = client.post("/v1/audio/transcriptions", data={"model": "x"})
    assert r.status_code == 400
    assert err(r) == {"message": "invalid request: missing or malformed form fields",
                      "type": "invalid_request_error", "code": "invalid_request"}


def test_empty_file(client):
    r = post(client, b"")
    assert r.status_code == 400
    assert err(r) == {"message": "empty file", "type": "invalid_request_error", "code": "empty_file"}


def test_corrupt_file(client):
    r = post(client, b"this is not audio at all" * 100)
    assert r.status_code == 400
    assert err(r)["message"] == "could not decode audio"
    assert client.model.calls == []


def test_silence_too_short(client):
    r = post(client, wav_bytes(0.05))
    assert r.status_code == 400
    assert err(r)["code"] == "no_audio"


def test_upload_too_large_by_content_length(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_MB", 1)
    r = post(client, b"\0" * (2 * 1024 * 1024))
    assert r.status_code == 413
    assert err(r) == {"message": "file exceeds 1 MB", "type": "invalid_request_error", "code": "upload_too_large"}
    assert client.model.calls == []


def test_upload_too_large_without_content_length(client, monkeypatch):
    """Chunked upload: the limit is enforced while the bytes arrive."""
    monkeypatch.setattr(server, "MAX_UPLOAD_MB", 1)
    sent = []

    def body():
        yield b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n"
        for _ in range(64):  # 4 MiB in 64 KiB pieces
            sent.append(1)
            yield b"\0" * 65536
        yield b"\r\n--b--\r\n"

    r = client.post("/v1/audio/transcriptions", content=body(),
                    headers={"Content-Type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413
    assert err(r)["code"] == "upload_too_large"
    assert client.model.calls == []


def test_audio_too_long(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_AUDIO_S", 3.0)
    r = post(client, wav_bytes(4.0, rate=16000, channels=1))
    assert r.status_code == 413
    assert client.model.calls == []


def test_long_audio_is_chunked(client, monkeypatch):
    monkeypatch.setattr(server, "CHUNK_S", 2.0)
    monkeypatch.setattr(server, "CHUNK_SEARCH_S", 0.5)
    r = post(client, wav_bytes(5.0, rate=16000, channels=1), response_format="verbose_json")
    assert r.status_code == 200
    assert len(client.model.calls) == 3
    assert sum(client.model.calls) == 80000
    # The second chunk's words are shifted by its start offset.
    assert r.json()["words"][4]["start"] > 1.4


def test_backpressure_returns_429(client):
    """One request runs, one waits (MAX_CONCURRENT 1, MAX_QUEUE 1), the third is rejected."""
    client.model.gate = threading.Event()
    payload = wav_bytes(1.0, rate=16000, channels=1)
    results = []
    threads = [threading.Thread(target=lambda: results.append(post(client, payload).status_code)) for _ in range(2)]
    for t in threads:
        t.start()
    for _ in range(100):  # until both are admitted
        if server._inflight == 2:
            break
        time.sleep(0.02)
    assert server._inflight == 2
    r = post(client, payload)
    assert r.status_code == 429
    assert r.headers["Retry-After"] == server.RETRY_AFTER_S
    assert err(r)["type"] == "rate_limit_error"
    client.model.gate.set()
    for t in threads:
        t.join(10)
    assert sorted(results) == [200, 200]
    assert server._inflight == 0
    # Capacity is free again.
    assert post(client, payload).status_code == 200


def test_timeout_between_chunks_releases_slot(client, monkeypatch):
    monkeypatch.setattr(server, "CHUNK_S", 1.0)
    monkeypatch.setattr(server, "CHUNK_SEARCH_S", 0.2)
    monkeypatch.setattr(server, "REQUEST_TIMEOUT_S", 0.2)
    client.model.delay = 0.3
    r = post(client, wav_bytes(5.0, rate=16000, channels=1))
    assert r.status_code == 504
    assert err(r)["code"] == "timeout"
    assert len(client.model.calls) == 1  # stopped after the first chunk
    client.model.delay = 0.0
    monkeypatch.setattr(server, "REQUEST_TIMEOUT_S", 60.0)
    assert post(client, wav_bytes(1.0)).status_code == 200


def test_health_and_models(client, monkeypatch):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/models").json()["data"][0]["id"] == server.MODEL_NAME
    monkeypatch.setattr(server, "_ready", False)
    assert client.get("/health").status_code == 503


def test_log_has_no_transcript(client, capsys):
    post(client, wav_bytes(1.0), prompt="geheimes-glossar")
    out = capsys.readouterr().out
    assert "parakeet_request" in out
    assert "Geheimer" not in out
    assert "geheimes-glossar" not in out
    assert '"prompt_ignored":true' in out


def test_split_points_cuts_in_silence():
    sr = 16000
    audio = np.ones(10 * sr, dtype=np.float32)
    audio[int(3.5 * sr):int(3.7 * sr)] = 0.0  # pause inside the search window
    starts = server.split_points(audio, chunk_s=4.0, search_s=1.0, sr=sr)
    assert starts[0] == 0
    assert 3.5 * sr <= starts[1] <= 3.7 * sr
    assert all(b - a <= 4 * sr for a, b in zip(starts, starts[1:] + [len(audio)]))


def test_split_points_short_audio():
    assert server.split_points(np.zeros(16000, dtype=np.float32), 30.0, 5.0) == [0]
