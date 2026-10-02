"""Tests for deploy/whisper/server.py with a stubbed model (no model download).

Run: pip install fastapi httpx python-multipart pytest && pytest deploy/whisper
"""

import io
import json
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import server  # noqa: E402

SECRET_PROMPT = "geheimes-fachwort-im-prompt"
SECRET_HOTWORD = "geheimes-hotword"
SECRET_TEXT = "geheimer transkriptinhalt"


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        seg = SimpleNamespace(text=f" {SECRET_TEXT}", start=0.0, end=2.0, avg_logprob=-0.1, no_speech_prob=0.0)
        info = SimpleNamespace(language="de", duration=10.0, duration_after_vad=8.0)
        return iter([seg]), info


@pytest.fixture
def client(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(server, "_model", model)
    monkeypatch.setattr(server, "DEFAULT_PROMPT", "")
    monkeypatch.setattr(server, "DEFAULT_HOTWORDS", "")
    monkeypatch.setattr(server, "DEFAULT_LANGUAGE", None)
    import asyncio

    monkeypatch.setattr(server, "_sem", asyncio.Semaphore(1))
    c = TestClient(server.app)
    c.model = model
    return c


def post(client, **data):
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", io.BytesIO(b"RIFFxxxx"), "audio/wav")},
        data=data,
    )


def test_combine_prompt():
    assert server.combine_prompt("", None) == (None, "none")
    assert server.combine_prompt("", "  ") == (None, "none")
    assert server.combine_prompt("A", None) == ("A", "default")
    assert server.combine_prompt("", "B") == ("B", "request")
    assert server.combine_prompt("A", "B") == ("A B", "both")


def test_combine_prompt_keeps_tail():
    prompt, _ = server.combine_prompt("d" * 3000, "REQUEST")
    assert len(prompt) == server.MAX_PROMPT_CHARS
    assert prompt.endswith("REQUEST")


def test_resolve_hotwords():
    assert server.resolve_hotwords("", None) is None
    assert server.resolve_hotwords("default", None) == "default"
    assert server.resolve_hotwords("default", "req") == "req"
    assert server.resolve_hotwords("default", "  ") == "default"


def test_no_prompt_by_default(client):
    assert post(client).json() == {"text": SECRET_TEXT}
    kw = client.model.calls[0]
    assert kw["initial_prompt"] is None and kw["hotwords"] is None
    assert kw["beam_size"] == server.BEAM_SIZE and kw["vad_filter"] == server.VAD_FILTER


def test_request_fields_are_forwarded(client):
    post(client, prompt="Alpha", hotwords="Beta Gamma", language="de", temperature="0.2")
    kw = client.model.calls[0]
    assert kw["initial_prompt"] == "Alpha"
    assert kw["hotwords"] == "Beta Gamma"
    assert kw["language"] == "de"
    assert kw["temperature"] == 0.2


def test_defaults_combined_with_request(client, monkeypatch):
    monkeypatch.setattr(server, "DEFAULT_PROMPT", "Standard.")
    monkeypatch.setattr(server, "DEFAULT_HOTWORDS", "StdWort")
    post(client)
    kw = client.model.calls[-1]
    assert kw["initial_prompt"] == "Standard." and kw["hotwords"] == "StdWort"
    post(client, prompt="Anfrage.", hotwords="Eigenes")
    kw = client.model.calls[-1]
    assert kw["initial_prompt"] == "Standard. Anfrage."
    assert kw["hotwords"] == "Eigenes"


def test_translations_forward_hotwords(client):
    client.post(
        "/v1/audio/translations",
        files={"file": ("a.wav", io.BytesIO(b"RIFFxxxx"), "audio/wav")},
        data={"hotwords": "X"},
    )
    assert client.model.calls[0]["hotwords"] == "X"
    assert client.model.calls[0]["task"] == "translate"


def test_log_line_has_timings_and_no_content(client, capsys, monkeypatch):
    monkeypatch.setattr(server, "DEFAULT_PROMPT", "Standard.")
    post(client, prompt=SECRET_PROMPT, hotwords=SECRET_HOTWORD)
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.startswith("whisper_request ")]
    assert len(lines) == 1
    rec = json.loads(lines[0].split(" ", 1)[1])
    assert rec["status"] == "ok" and rec["audio_s"] == 10.0 and rec["audio_after_vad_s"] == 8.0
    assert rec["prompt_source"] == "both"
    assert rec["prompt_chars"] == len("Standard. " + SECRET_PROMPT)
    assert rec["hotwords_chars"] == len(SECRET_HOTWORD)
    assert rec["file_bytes"] == 8 and rec["language"] == "de"
    for key in ("prep_s", "infer_s", "queue_s", "total_s", "rtf", "beam_size", "vad", "model"):
        assert key in rec
    for secret in (SECRET_PROMPT, SECRET_HOTWORD, SECRET_TEXT, "Standard."):
        assert secret not in out


def test_error_log_without_content(client, capsys):
    def boom(path, **kw):
        raise RuntimeError("decode failed")

    client.model.transcribe = boom
    r = post(client, prompt=SECRET_PROMPT)
    assert r.status_code == 400
    out = capsys.readouterr().out
    assert '"status":"error"' in out and SECRET_PROMPT not in out
