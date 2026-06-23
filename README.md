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

Model baked into image at build time. No init container or PVC needed.

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
| `whisper`             | Whisper ASR with model baked in        |

All images run as non-root with no privilege escalation.

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
