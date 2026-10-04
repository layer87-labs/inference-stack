# inference-stack

[![Go Tests](https://github.com/layer87-labs/inference-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/layer87-labs/inference-stack/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Unified CPU-based inference stack — serving embeddings, reranking and transcription behind a single OpenAI-compatible API.

## Components

| Component            | Runtime                                                                | Model                   | Purpose                                 |
| -------------------- | ---------------------------------------------------------------------- | ----------------------- | --------------------------------------- |
| **inference-router** | Go                                                                     | —                       | Unified OpenAI-compatible reverse proxy |
| **embedding**        | [TEI](https://github.com/huggingface/text-embeddings-inference) (Rust) | BAAI/bge-m3             | Dense + sparse embeddings               |
| **reranker**         | [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) (Python)    | BAAI/bge-reranker-v2-m3 | Cross-encoder reranking                 |
| **whisper**          | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Python)   | large-v3-turbo          | Audio transcription (ASR)               |

All backends are disabled by default and enabled selectively via Helm values or env vars.

## Quick Start

### Local Development

```bash
# Run unit tests
make test

# Full end-to-end test (starts mock backends + router)
make test-local
```

### Kubernetes (Helm)

```bash
helm install inference-stack ./deploy/helm \
  --namespace ai \
  --set embedding.enabled=true \
  --set reranker.enabled=true \
  --set router.enabled=true
```

## API

All endpoints are OpenAI-compatible:

```bash
# Embeddings
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-m3","input":"Hello world"}'

# Reranking
curl http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Go?","documents":["Go is a language","Python is a language"]}'

# Transcription
curl http://localhost:8080/v1/audio/transcriptions \
  -F model=whisper-large-v3-turbo \
  -F file=@audio.mp3

# Model list (merged from all enabled backends)
curl http://localhost:8080/v1/models
```

`/v1/models` also reports the request limits of the embedding and reranker
backends (`limits.max_batch_tokens`, `limits.max_client_batch_size`,
`limits.max_input_length`), read from the running backend, so clients can size
batches without trial and error. The model id it advertises
(`BAAI/bge-m3`) can be sent back as `model` on `/v1/embeddings`; the router
strips it before forwarding, because TEI only accepts its internal model path
or no model at all.

## Architecture

### Init Container Pattern (Embedding + Reranker)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  model-init     │────▶│  /model (PVC)    │────▶│  main container │
│  (bakes model)  │     │  shared volume   │     │  (reads model)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

1. **Init container** copies baked model to PVC via `rsync --ignore-existing`
2. **Main container** reads model from PVC (mounted read-only)
3. PVC persists model across restarts (idempotent — no re-copy on restart)

### Whisper (Standalone)

Model baked into image at build time. No init container or PVC needed. The
image runs `deploy/whisper/server.py` (faster-whisper behind a small FastAPI
app) which implements `POST /v1/audio/transcriptions` and
`/v1/audio/translations` natively, so the router forwards requests unchanged.
`/health` is answered independently of inference and stays responsive while a
transcription is running.

### Router (Go Proxy)

Pure reverse proxy — routes requests to backends based on path prefix. Zero model logic.

## Container Images

All images are published to `ghcr.io/layer87-labs/`:

| Image                 | Description                            |
| --------------------- | -------------------------------------- |
| `inference-router`    | Go reverse proxy                       |
| `tei-base`            | Hardened TEI base (patchelf, non-root) |
| `tei-runtime`         | TEI runtime (model via volume)         |
| `tei-model-init`      | BGE-M3 model baked, copies to volume   |
| `reranker-model-init` | BGE-reranker-v2-m3 baked               |
| `reranker-server`     | FlagEmbedding HTTP server              |
| `whisper`             | Whisper ASR server, model baked in     |

All images run as non-root with no privilege escalation.

### Verify image signatures

Images built by the release workflows are signed with
[cosign](https://github.com/sigstore/cosign) keyless via GitHub OIDC. Verify by
digest against the exact workflow identity:

```bash
IMAGE=ghcr.io/layer87-labs/inference-router
DIGEST=$(docker buildx imagetools inspect "$IMAGE:<version>" --format '{{json .Manifest}}' | jq -r .digest)

cosign verify \
  --certificate-identity 'https://github.com/layer87-labs/inference-stack/.github/workflows/release.yml@refs/heads/main' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  "$IMAGE@$DIGEST"
```

`release.yml` signs `inference-router`, `tei-base` and `tei-runtime`.
`release-models.yml` signs `tei-model-init`, `tei-reranker-model-init` and
`whisper`; for those, use the identity
`https://github.com/layer87-labs/inference-stack/.github/workflows/release-models.yml@refs/heads/main`
(for manual runs, the ref is the branch the workflow was started from).
Images published before signing was added are unsigned.

## Build

```bash
# Go binary
make build

# All container images
make docker

# Push to registry
make push REGISTRY=ghcr.io/layer87-labs VERSION=0.1.0
```

## Metrics

Prometheus metrics on `:9090/metrics`:

- `inference_router_requests_total` — total requests by backend/path/status
- `inference_router_request_duration_seconds` — latency histogram
- `inference_router_active_requests` — in-flight requests
- `inference_router_backend_up` — backend health (0/1)
- `inference_router_upstream_errors_total` — upstream error counts

## CPU Tuning

### Embedding (TEI + BGE-M3)

| `--max-batch-tokens` | RSS (fp32) | Recommendation       |
| -------------------- | ---------- | -------------------- |
| 4096                 | 3-6Gi      | **CPU default**      |
| 8192                 | 6-10Gi     | May OOM on 8Gi limit |
| 16384                | 10-16Gi    | GPU-only             |

### Reranker (FlagEmbedding + BGE-reranker-v2-m3)

| `MAX_LENGTH` | RSS (fp16) | Recommendation  |
| ------------ | ---------- | --------------- |
| 512          | 2-4Gi      | **CPU default** |
| 1024         | 4-6Gi      | Larger context  |

### Whisper (faster-whisper + large-v3-turbo, int8)

Measured on CPU: ~2.0 GiB RSS peak; roughly real-time speed on 2 threads (a
3.7 min recording took 3.3 min), faster with more threads.

- `whisper.cpuThreads` must match the CPU limit (CTranslate2 sizes itself by
  the node's cores, not by the cgroup limit).
- Transcriptions run one at a time; further requests queue in the server. Keep
  the router's `WHISPER_TIMEOUT` (default `300s`) above queue time + runtime,
  e.g. via `router.extraEnv`.
- Upload limit: `whisper.maxUploadMB` (default 100). Formats: anything FFmpeg
  decodes (wav, mp3, m4a, ogg, flac, webm, mp4, ...).
- `response_format`: `json` (default), `text`, `verbose_json`, `srt`, `vtt`.
- Set `whisper.language` (e.g. `de`) to skip language auto-detection; a
  request's own `language` field wins.
- Vocabulary hints, two mechanisms (both optional, both off by default):
  - `prompt` (form field, OpenAI-compatible) becomes faster-whisper's
    `initial_prompt`. Set a server-side default with `whisper.initialPrompt`
    (`WHISPER_INITIAL_PROMPT`) for clients that send no prompt (e.g. an STT
    integration that only sends `model` and `language`). A request prompt is
    **appended** to the default (`<default> <request>`): Whisper weighs the end
    of the prompt most, and an over-long prompt is cut from the front, so the
    request-specific text survives. Whisper reads at most 223 tokens
    (`max_length // 2 - 1`); faster-whisper keeps the last 223 and drops the
    rest. The server additionally caps the combined text at 2000 characters
    (tail kept).
  - `hotwords` (form field, extension; free text such as
    `Kubernetes Grafana Postgres`) is passed to faster-whisper's `hotwords`.
    Server default: `whisper.hotwords` (`WHISPER_HOTWORDS`). A request's
    `hotwords` **replaces** the default (no merging). faster-whisper puts the
    hotwords in front of the previous-text prompt; it ignores them only when
    `prefix` is set, which this server never sets, so `prompt` and `hotwords`
    work together (checked in `faster_whisper/transcribe.py`, `get_prompt`,
    version 1.2.1). Both count against the same 223-token window of their own.
  - Neutral example for `whisper.initialPrompt`:
    `Release notes for Kubernetes, PostgreSQL and Grafana.`
  - Hints raise the odds of a spelling; they do not guarantee it. Keep them
    short and spell terms the way they should appear.
- Repetition loops on long audio: the sequential decoder conditions every
  window on the previous text and can get stuck repeating itself. The server
  therefore (1) keeps `condition_on_previous_text` off by default
  (`whisper.conditionOnPreviousText`), (2) sends audio longer than
  `whisper.batchThresholdSeconds` (default 35 s, `WHISPER_BATCH_THRESHOLD_S`)
  through faster-whisper's `BatchedInferencePipeline` (VAD-cut chunks decoded
  independently, `whisper.batchSize` = 8, `WHISPER_BATCH_SIZE`), and (3) uses
  faster-whisper's temperature fallback (0.0, 0.2, ... 1.0 with its default
  compression-ratio, log-prob and no-speech thresholds) on the sequential
  path. The batched path always uses VAD and only the first temperature (the
  library does not fall back there). A request's explicit `temperature` form
  field (including `0`) is used as given, without fallback; clients that always
  send `temperature=0` therefore get no fallback on the sequential path.
  Measured locally (FLEURS de, int8, 4 threads, beam 1, VAD): three 55-88 s
  clips WER 3.1 % instead of 42 % (one clip looped to 256 instead of 120
  words) and 56 s of audio in 17.8 s instead of 27.2 s; 40 single clips
  5.6 % vs 5.2 % (noise); no speed difference on short clips.
- Warmup: after loading the model the server runs a 2 s synthetic clip through
  VAD and decoder (`WHISPER_WARMUP`, default `true`) so the first real request
  does not pay the initialisation cost. `/health` returns 503 until model and
  warmup are done; a failed warmup is logged and does not block startup.
- Model source and pins: the image bakes `large-v3-turbo` from
  `dropbox-dash/faster-whisper-large-v3-turbo` at a fixed commit
  (`WHISPER_MODEL_REVISION` in `deploy/Containerfile.whisper`) and verifies
  `model.bin` against its SHA-256; the build fails if either changes. This is
  the repository faster-whisper's `large-v3-turbo` alias
  (`mobiuslabsgmbh/faster-whisper-large-v3-turbo`) redirects to (HTTP 307);
  `Systran/faster-whisper-large-v3-turbo` answers HTTP 401. Model license: MIT.
  Python dependencies are pinned in `deploy/whisper/requirements.txt` (`av`
  must stay on 16.x, newer majors break faster-whisper 1.2.1) and the base
  image by digest.
- Per-request log line (JSON after the prefix `whisper_request`), e.g.
  `whisper_request {"status":"ok","path":"standard","audio_s":9.0,"prep_s":0.4,"infer_s":7.6,"total_s":8.1,"rtf":0.889,...}`:
  audio duration (`audio_s`, `audio_after_vad_s`), upload, queue wait, decode +
  VAD (`prep_s`) and inference (`infer_s`) time, real-time factor
  (`rtf = (prep_s + infer_s) / audio_s`), model, language, `beam_size`, `vad`, `path` (`standard|batched`),
  `temperature_mode` (`fallback|fixed`),
  file size, and `prompt_chars` / `hotwords_chars` / `prompt_source`
  (`none|default|request|both`). Prompt, hotwords and transcript text are
  never logged.
- `/v1/audio/translations` (to English) is exposed, but large-v3-turbo is not
  trained for translation; use a non-turbo model for that.

### Scheduling (`affinity`, `topologySpreadConstraints`)

`embedding` and `whisper` accept optional `affinity` and
`topologySpreadConstraints` (empty by default, so the rendered manifests do not
change). With `embedding.replicaCount > 1` the replicas can land on one node,
and then the PodDisruptionBudget does not protect against a node failure.
Example, soft spreading that never blocks scheduling:

```yaml
embedding:
  replicaCount: 2
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          podAffinityTerm:
            topologyKey: kubernetes.io/hostname
            labelSelector:
              matchLabels:
                app.kubernetes.io/component: embedding
  topologySpreadConstraints:
    - maxSkew: 1
      topologyKey: kubernetes.io/hostname
      whenUnsatisfiable: ScheduleAnyway
      labelSelector:
        matchLabels:
          app.kubernetes.io/component: embedding
```

## Environment Variables

| Variable               | Default | Description                   |
| ---------------------- | ------- | ----------------------------- |
| `ROUTER_ADDR`          | `:8080` | Main listen address           |
| `ROUTER_METRICS_ADDR`  | `:9090` | Prometheus metrics address    |
| `ROUTER_READ_TIMEOUT`  | `120s`  | HTTP read timeout             |
| `ROUTER_WRITE_TIMEOUT` | `300s`  | HTTP write timeout            |
| `EMBEDDING_ENABLED`    | `false` | Enable embedding backend      |
| `EMBEDDING_URL`        | —       | Base URL for TEI embedding    |
| `EMBEDDING_TIMEOUT`    | `60s`   | Per-request timeout           |
| `RERANKER_ENABLED`     | `false` | Enable reranker backend       |
| `RERANKER_URL`         | —       | Base URL for reranker         |
| `RERANKER_TIMEOUT`     | `30s`   | Per-request timeout           |
| `WHISPER_ENABLED`      | `false` | Enable Whisper backend        |
| `WHISPER_URL`          | —       | Base URL for Whisper          |
| `WHISPER_TIMEOUT`      | `300s`  | Per-request timeout           |
| `LOG_LEVEL`            | `info`  | `debug`/`info`/`warn`/`error` |
| `LOG_FORMAT`           | `json`  | `json`/`console`              |

## Security

- All containers run as non-root (uid 1000 or 65532)
- `allowPrivilegeEscalation: false`
- `capabilities: drop: [ALL]`
- ONNX Runtime exec-stack flag cleared via patchelf
- No outbound network calls at runtime (`HF_HUB_OFFLINE=1`)
- Router image uses distroless base (no shell)

## License

[Apache License 2.0](LICENSE)
