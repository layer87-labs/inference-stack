# Configuration

The router is configured exclusively through environment variables (`internal/config/config.go`). With the
Helm chart the backend variables are set for you; add or override variables with `router.extraEnv`.

At least one backend must be enabled, and every enabled backend needs its `*_URL`, otherwise the router exits
at startup with a config error.

| Variable | Default | Description |
| --- | --- | --- |
| `ROUTER_ADDR` | `:8080` | Main listen address |
| `ROUTER_METRICS_ADDR` | `:9090` | Prometheus metrics address (`/metrics`) |
| `ROUTER_READ_TIMEOUT` | `120s` | HTTP read timeout |
| `ROUTER_WRITE_TIMEOUT` | `300s` | HTTP write timeout |
| `ROUTER_IDLE_TIMEOUT` | `60s` | HTTP idle timeout |
| `ROUTER_MAX_REQUEST_SIZE` | `104857600` | Maximum request body in bytes (100 MB) |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warn` or `error` |
| `LOG_FORMAT` | `json` | `json` or `console` |
| `EMBEDDING_ENABLED` | `false` | Enable the embedding backend |
| `EMBEDDING_URL` | none | Base URL of the TEI embedding server |
| `EMBEDDING_TIMEOUT` | `60s` | Per-request timeout |
| `EMBEDDING_MAX_CONCURRENT` | `20` | Requests forwarded at once; more get `503` + `Retry-After` |
| `RERANKER_ENABLED` | `false` | Enable the reranker backend |
| `RERANKER_URL` | none | Base URL of the TEI reranker server |
| `RERANKER_TIMEOUT` | `30s` | Per-request timeout |
| `RERANKER_MAX_CONCURRENT` | `20` | Requests forwarded at once; more get `503` + `Retry-After` |
| `WHISPER_ENABLED` | `false` | Enable the Whisper backend |
| `WHISPER_URL` | none | Base URL of the Whisper server |
| `WHISPER_TIMEOUT` | `300s` | Per-request timeout |
| `WHISPER_MAX_CONCURRENT` | `8` | Requests forwarded at once; more get `503` + `Retry-After` |

A value of `0` or below for a `*_MAX_CONCURRENT` variable disables the bound for that backend.

## Backpressure

When a backend is at its concurrency bound the router answers immediately with
`503 Service Unavailable`, a `Retry-After: 3` header and `{"error":"backend_overloaded", ...}` instead of
queuing the request. Clients should back off and retry. Each rejection increments
`inference_router_rejected_total{backend}`.

## Health endpoints

| Path | Port | Behaviour |
| --- | --- | --- |
| `/healthz`, `/readyz` | 8080 | Backends are probed (`GET <url>/health`) every 15 seconds. The answer is always HTTP 200; the JSON body says `ok` or `degraded` and lists each backend as `ok`, `unhealthy` or `disabled`. A failing backend does not take the router out of rotation. |
| `/health` | 8080 | Router liveness only: `{"status":"ok"}` |
| `/metrics` | 9090 | Prometheus metrics |

## Metrics

| Metric | Type | Labels |
| --- | --- | --- |
| `inference_router_requests_total` | counter | `backend`, `path`, `status` |
| `inference_router_request_duration_seconds` | histogram | `backend`, `path` |
| `inference_router_request_size_bytes` | histogram | `backend` |
| `inference_router_response_size_bytes` | histogram | `backend` |
| `inference_router_active_requests` | gauge | `backend` |
| `inference_router_backend_up` | gauge | `backend` |
| `inference_router_upstream_errors_total` | counter | `backend`, `error_type` |
| `inference_router_rejected_total` | counter | `backend` |
