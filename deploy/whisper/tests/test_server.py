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

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, **kwargs})
        seg = SimpleNamespace(text=f" {SECRET_TEXT}", start=0.0, end=2.0, avg_logprob=-0.1, no_speech_prob=0.0)
        info = SimpleNamespace(language="de", duration=10.0, duration_after_vad=8.0)
        return iter([seg]), info


@pytest.fixture
def client(monkeypatch):
    model = FakeModel()
    batched = FakeModel()
    audio = {"duration": 10.0}
    monkeypatch.setattr(server, "_model", model)
    monkeypatch.setattr(server, "_batched", batched)
    monkeypatch.setattr(server, "_ready", True)
    monkeypatch.setattr(server, "_decode", lambda path: ("audio", audio["duration"]))
    monkeypatch.setattr(server, "BATCH_THRESHOLD_S", 35.0)
    monkeypatch.setattr(server, "BATCH_SIZE", 8)
    monkeypatch.setattr(server, "CONDITION_ON_PREVIOUS_TEXT", False)
    monkeypatch.setattr(server, "DEFAULT_PROMPT", "")
    monkeypatch.setattr(server, "DEFAULT_HOTWORDS", "")
    monkeypatch.setattr(server, "DEFAULT_LANGUAGE", None)
    import asyncio

    monkeypatch.setattr(server, "_sem", asyncio.Semaphore(1))
    c = TestClient(server.app)
    c.model = model
    c.batched = batched
    c.audio = audio
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
    def boom(audio, **kw):
        raise RuntimeError("decode failed")

    client.model.transcribe = boom
    r = post(client, prompt=SECRET_PROMPT)
    assert r.status_code == 400
    out = capsys.readouterr().out
    assert '"status":"error"' in out and SECRET_PROMPT not in out


def test_default_uses_temperature_fallback(client):
    post(client)
    kw = client.model.calls[0]
    assert kw["temperature"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert kw["condition_on_previous_text"] is False
    assert "compression_ratio_threshold" not in kw  # faster-whisper defaults apply


def test_explicit_temperature_wins_including_zero(client):
    post(client, temperature="0")
    assert client.model.calls[-1]["temperature"] == 0.0
    post(client, temperature="0.4")
    assert client.model.calls[-1]["temperature"] == 0.4


def test_condition_on_previous_text_switch(client, monkeypatch):
    monkeypatch.setattr(server, "CONDITION_ON_PREVIOUS_TEXT", True)
    post(client)
    assert client.model.calls[0]["condition_on_previous_text"] is True


def test_short_audio_uses_standard_path(client):
    client.audio["duration"] = 35.0  # threshold is exclusive
    post(client)
    assert len(client.model.calls) == 1 and client.batched.calls == []
    assert client.model.calls[0]["audio"] == "audio"


def test_long_audio_uses_batched_path(client):
    client.audio["duration"] = 88.0
    post(client, prompt="Alpha", hotwords="Beta", language="de")
    assert client.model.calls == []
    kw = client.batched.calls[0]
    assert kw["batch_size"] == 8 and kw["vad_filter"] is True
    assert kw["initial_prompt"] == "Alpha" and kw["hotwords"] == "Beta" and kw["language"] == "de"
    assert kw["condition_on_previous_text"] is False


def test_batched_path_forces_vad_even_if_disabled(client, monkeypatch):
    monkeypatch.setattr(server, "VAD_FILTER", False)
    client.audio["duration"] = 60.0
    post(client)
    assert client.batched.calls[0]["vad_filter"] is True


def test_threshold_configurable(client, monkeypatch):
    monkeypatch.setattr(server, "BATCH_THRESHOLD_S", 5.0)
    post(client)
    assert len(client.batched.calls) == 1


def test_response_formats_unchanged_on_batched_path(client):
    client.audio["duration"] = 90.0
    assert post(client, response_format="json").json() == {"text": SECRET_TEXT}
    assert post(client, response_format="text").text == SECRET_TEXT + "\n"
    body = post(client, response_format="verbose_json").json()
    assert body["segments"][0]["text"] == f" {SECRET_TEXT}" and body["language"] == "de"
    assert "00:00:00,000 --> 00:00:02,000" in post(client, response_format="srt").text
    assert post(client, response_format="vtt").text.startswith("WEBVTT")


def test_log_has_path_and_temperature_mode(client, capsys):
    post(client)
    client.audio["duration"] = 80.0
    post(client, temperature="0")
    recs = [json.loads(l.split(" ", 1)[1]) for l in capsys.readouterr().out.splitlines()
            if l.startswith("whisper_request ")]
    assert (recs[0]["path"], recs[0]["temperature_mode"]) == ("standard", "fallback")
    assert (recs[1]["path"], recs[1]["temperature_mode"]) == ("batched", "fixed")
    assert recs[1]["batch_size"] == 8 and recs[1]["temperature"] == 0.0
    assert recs[0]["batch_size"] is None


def test_health_not_ready_until_warmup_done(client, monkeypatch):
    assert client.get("/health").json() == {"status": "ok"}
    monkeypatch.setattr(server, "_ready", False)
    r = client.get("/health")
    assert r.status_code == 503 and r.json() == {"status": "loading"}


def test_warmup_runs_standard_path_with_and_without_vad(client, monkeypatch):
    monkeypatch.setattr(server, "_warmup_audio", lambda: "tone")
    monkeypatch.setattr(server, "VAD_FILTER", True)
    server._warmup()
    assert [c["vad_filter"] for c in client.model.calls] == [True, False]
    assert all(c["audio"] == "tone" for c in client.model.calls)
    assert client.batched.calls == []


def test_warmup_failure_is_only_logged(client, monkeypatch, capsys):
    monkeypatch.setattr(server, "_warmup_audio", lambda: "tone")

    def boom(audio, **kw):
        raise RuntimeError("vad init failed")

    client.model.transcribe = boom
    server._warmup()  # must not raise
    assert "Warmup failed" in capsys.readouterr().out


def test_lifespan_ready_only_after_warmup(monkeypatch):
    order = []
    monkeypatch.setattr(server, "_ready", False)
    monkeypatch.setattr(server, "_model", None)
    monkeypatch.setattr(server, "_load_model", lambda: order.append("load") or FakeModel())
    monkeypatch.setattr(server, "_load_batched", lambda m: order.append("batched") or FakeModel())
    monkeypatch.setattr(server, "WARMUP", True)

    def fake_warmup():
        order.append("warmup")
        assert server._ready is False

    monkeypatch.setattr(server, "_warmup", fake_warmup)
    with TestClient(server.app) as c:
        assert order == ["load", "batched", "warmup"]
        assert c.get("/health").json() == {"status": "ok"}


def test_warmup_can_be_disabled(monkeypatch):
    monkeypatch.setattr(server, "_load_model", lambda: FakeModel())
    monkeypatch.setattr(server, "_load_batched", lambda m: FakeModel())
    monkeypatch.setattr(server, "WARMUP", False)
    monkeypatch.setattr(server, "_warmup", lambda: pytest.fail("warmup must not run"))
    with TestClient(server.app) as c:
        assert c.get("/health").status_code == 200
