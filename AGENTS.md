# AGENTS.md — Inference Stack

This file provides context for AI coding agents working in this repository.

---

## Project Overview

`inference-stack` is a Go HTTP reverse-proxy router that unifies CPU-based AI inference backends behind a single OpenAI-compatible API:

| Backend | Runtime | Model | Path prefix | Env vars |
|---|---|---|---|---|
| Embedding | TEI (Rust) | BAAI/bge-m3 | `/v1/embeddings`, `/embed`, `/embed_sparse` | `EMBEDDING_ENABLED`, `EMBEDDING_URL` |
| Reranker | FlagEmbedding (Python) | BAAI/bge-reranker-v2-m3 | `/v1/rerank`, `/rerank` | `RERANKER_ENABLED`, `RERANKER_URL` |
| Whisper ASR | faster-whisper (Python) | large-v3-turbo | `/v1/audio/transcriptions`, `/v1/audio/translations` | `WHISPER_ENABLED`, `WHISPER_URL` |

`/v1/models` returns a merged list from all enabled backends (with static fallback).

---

## Repository Layout

```
cmd/
  router/          main.go — HTTP server (port 8080) + metrics server (port 9090)
  mockbackend/     main.go — test mock for all three backends (MOCK_TYPE + MOCK_PORT)
internal/
  build/           version.go — build info injected via ldflags
  config/          config.go — env-based Config struct, Load(), validate()
  proxy/           proxy.go  — Router (ServeHTTP, dispatch, handleModels)
                   proxy_test.go — integration tests (httptest, no external deps)
  health/          health.go — background health probes, /healthz + /readyz handler
  metrics/         metrics.go — promauto Prometheus counters/histograms/gauges
scripts/
  test-local.sh    end-to-end test script (curl-based, starts mock backends + router)
deploy/
  Containerfile.router          Go router binary
  Containerfile.tei-base        TEI hardened base (patchelf + nonroot)
  Containerfile.tei-runtime     TEI embedding runtime (FROM tei-base)
  Containerfile.tei-model-init  Embedding model init container (bakes BGE-M3)
  Containerfile.reranker-model-init  Reranker model init container
  Containerfile.reranker-server      FlagEmbedding FastAPI server (/v1/rerank)
  Containerfile.whisper         Whisper ASR with model baked in
  reranker/
    server.py                   FlagEmbedding FastAPI server
  helm/                         Helm chart (Chart.yaml, values.yaml, templates/)
```

---

## Container Image Versioning

All images use **CalVer** (`YYYYMM.DD.PATCH`).

Registry: `ghcr.io/layer87-labs/<image>:<tag>`

---

## Architecture Patterns

### Init Container Pattern (Embedding + Reranker)

1. **Init container** (`*-model-init`): Bakes the model at build time, copies to a shared `/model` volume via `rsync -rlgoDv --ignore-existing` at pod start.
2. **Main container** (`tei-runtime` / `reranker-server`): Reads model from `/model` volume (mounted `readOnly: true`). Runs fully offline (`HF_HUB_OFFLINE=1`).

Key details:
- Pod `securityContext.fsGroup: 1000` — Kubernetes chowns the PVC to UID 1000 before init runs.
- `rsync -rlgoDv` instead of `-a` — avoids `-t` (timestamp preservation) which fails on some CSI volumes.
- PVC persistence keeps model across restarts (idempotent rsync skips existing files).

### Whisper (Standalone)

Model is baked directly into the image at build time. No init container, no PVC needed.

### Router (Go Proxy)

Pure reverse proxy — no model, no init container. Routes requests to backends based on path prefix.

---

## Build & Test Commands

```bash
make build          # compiles router → build/bin/inference-router
                    #           mock  → build/bin/mock-backend
make test           # go test -race ./...
make test-local     # make build + bash scripts/test-local.sh (full e2e)
make tidy           # go mod tidy
make lint           # golangci-lint run ./...
```

### Docker builds

```bash
make docker                           # build all images
make docker/router                    # inference-router
make docker/tei-base                  # tei-base
make docker/tei-runtime               # tei-runtime
make docker/tei-model-init-embedding  # tei-model-init
make docker/reranker-model-init       # reranker-model-init
make docker/reranker-server           # reranker-server
make docker/whisper                   # whisper
```

---

## Tech Stack & Constraints

- **Go 1.24+** — no generics workarounds needed
- **`net/http/httptest`** for all tests — no external test frameworks (no testify, no gomock)
- **`go.uber.org/zap`** for structured logging
- **`github.com/prometheus/client_golang`** for metrics
- **No new dependencies** — do not add entries to `go.mod` without discussion

---

## Key Design Decisions

### Config
- `config.Load()` reads exclusively from env vars with sane defaults.
- `config.validate()` enforces: if `*_ENABLED=true`, the matching `*_URL` must be set; at least one backend must be enabled.
- **Do not call `config.Load()` in tests** — construct `*config.Config` directly to avoid env pollution and satisfy the validation constraint.

### Proxy
- One `*httputil.ReverseProxy` per enabled backend, keyed by backend name.
- `dispatch()` checks `Backend.Enabled` first → 503 if disabled.
- Connection failures propagate through `ErrorHandler` → 502 with `{"error":"upstream_error"}`.
- `handleModels()` fetches `/v1/models` from each enabled backend with a 5-second timeout; falls back to the static `Backend.Models` slice on any error.

### Health
- `health.Checker` probes `<BaseURL>/health` every 15 seconds in the background.
- `ServeHTTP` (handles both `/healthz` and `/readyz`) always returns **HTTP 200** — even when `status:"degraded"` — so Kubernetes keeps routing traffic.

### Metrics
- All Prometheus metrics are **package-level `promauto` globals** in `internal/metrics/`.
- They are registered once at program start; do not re-register them in tests.
- Metric names follow the pattern `inference_router_*`.

---

## Writing Tests

Use `net/http/httptest` exclusively. Pattern for a new routing test:

```go
func TestRouting_MyBackend(t *testing.T) {
    // 1. Spin up a mock backend
    backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{"object":"list","data":[]}`))
    }))
    defer backend.Close()

    // 2. Build config manually (do NOT call config.Load())
    cfg := &config.Config{
        ReadTimeout: 5 * time.Second,
        WriteTimeout: 5 * time.Second,
        Embedding: config.Backend{Name: "embedding", BaseURL: backend.URL, Enabled: true, Timeout: 5 * time.Second},
        Reranker:  config.Backend{Name: "reranker"},
        Whisper:   config.Backend{Name: "whisper"},
    }

    // 3. Wire up the router
    log, _ := zap.NewDevelopment()
    checker := health.New(cfg.EnabledBackends(), log)
    router, _ := proxy.New(cfg, checker, log)

    // 4. Fire a request
    req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{}`))
    w := httptest.NewRecorder()
    router.ServeHTTP(w, req)

    if w.Code != http.StatusOK {
        t.Errorf("status = %d, want 200", w.Code)
    }
}
```

---

## Mock Backend

`cmd/mockbackend` is a standalone binary controlled by two env vars:

| Env | Values | Default |
|---|---|---|
| `MOCK_TYPE` | `embedding` \| `reranker` \| `whisper` | — (required) |
| `MOCK_PORT` | any free port | `9001` |

It exposes `/health`, `/v1/models`, and the backend-specific inference endpoints with static but valid OpenAI-compatible responses. Used by `scripts/test-local.sh` and for manual testing.

---

## Common Pitfalls

- `config.Load()` returns an error if no backend is enabled — never call it in tests.
- `promauto` metrics panic on double-registration. If you add a new metric, add it to `internal/metrics/metrics.go` as a package-level var, not inside a function.
- `health.Checker` must be initialised with `health.New(cfg.EnabledBackends(), log)` — passing `cfg.ActiveBackends()` will include disabled backends and skew the health response.
- The `statusWriter` wrapper in `proxy.go` must set `status = 200` lazily (in `Write`) to handle the case where `WriteHeader` is never called explicitly.
- TEI does **not** support reranking — the reranker uses FlagEmbedding, not TEI.
- Some CSI volumes don't support timestamp preservation — always use `rsync -rlgoDv`, never `rsync -a`.
- Pod `fsGroup: 1000` is required for init containers to write to PVCs (CSI provisions volumes as root).

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `ROUTER_ADDR` | `:8080` | Main listen address |
| `ROUTER_METRICS_ADDR` | `:9090` | Prometheus metrics address |
| `ROUTER_READ_TIMEOUT` | `120s` | HTTP read timeout |
| `ROUTER_WRITE_TIMEOUT` | `300s` | HTTP write timeout |
| `ROUTER_IDLE_TIMEOUT` | `60s` | HTTP idle timeout |
| `ROUTER_MAX_REQUEST_SIZE` | `104857600` | Max request body (100 MB) |
| `LOG_LEVEL` | `info` | `debug` \| `info` \| `warn` \| `error` |
| `LOG_FORMAT` | `json` | `json` \| `console` |
| `EMBEDDING_ENABLED` | `false` | Enable embedding backend |
| `EMBEDDING_URL` | — | Base URL for TEI embedding |
| `EMBEDDING_TIMEOUT` | `60s` | Per-request timeout |
| `RERANKER_ENABLED` | `false` | Enable reranker backend |
| `RERANKER_URL` | — | Base URL for FlagEmbedding reranker |
| `RERANKER_TIMEOUT` | `30s` | Per-request timeout |
| `WHISPER_ENABLED` | `false` | Enable Whisper backend |
| `WHISPER_URL` | — | Base URL for Whisper |
| `WHISPER_TIMEOUT` | `300s` | Per-request timeout |
