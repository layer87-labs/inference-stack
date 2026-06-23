#!/usr/bin/env bash
# =============================================================================
# test-local.sh — Inference Router local test runner
#
# Starts mock backends, builds and runs the router, fires curl tests.
# Cleans up all processes on exit (Ctrl+C or error).
#
# Usage:
#   ./scripts/test-local.sh              # run all tests
#   ./scripts/test-local.sh --only embed # run only embedding tests
#   SKIP_BUILD=1 ./scripts/test-local.sh # skip go build (use existing binary)
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
ROUTER_PORT=${ROUTER_PORT:-8080}
METRICS_PORT=${METRICS_PORT:-9090}
MOCK_EMBED_PORT=${MOCK_EMBED_PORT:-9101}
MOCK_RERANK_PORT=${MOCK_RERANK_PORT:-9102}
MOCK_WHISPER_PORT=${MOCK_WHISPER_PORT:-9103}
BINARY=${BINARY:-./build/bin/inference-router}
MOCK_BINARY=${MOCK_BINARY:-./build/bin/mock-backend}
ONLY=${1:-all}

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; FAILED=$((FAILED+1)); }
info() { echo -e "${BLUE}▶${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }

FAILED=0
PIDS=()

# ── Cleanup ───────────────────────────────────────────────────────────────────
cleanup() {
  echo ""
  info "Shutting down..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  info "Done."
}
trap cleanup EXIT INT TERM

# ── Helpers ───────────────────────────────────────────────────────────────────
wait_for_port() {
  local port=$1 name=$2 attempts=30
  for i in $(seq 1 $attempts); do
    if curl -sf "http://localhost:$port/health" >/dev/null 2>&1 || \
       curl -sf "http://localhost:$port/healthz" >/dev/null 2>&1; then
      ok "$name is up on :$port"
      return 0
    fi
    sleep 0.3
  done
  echo -e "${RED}TIMEOUT: $name did not start on :$port${NC}"
  exit 1
}

assert_status() {
  local desc=$1 expected=$2 actual=$3
  if [ "$actual" -eq "$expected" ]; then
    ok "$desc → HTTP $actual"
  else
    fail "$desc → expected $expected, got $actual"
  fi
}

assert_json_field() {
  local desc=$1 field=$2 expected=$3 body=$4
  local actual
  actual=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d$field)" 2>/dev/null || echo "PARSE_ERROR")
  if [ "$actual" = "$expected" ]; then
    ok "$desc → $field = $actual"
  else
    fail "$desc → expected $field=$expected, got $actual"
  fi
}

# ── Build ─────────────────────────────────────────────────────────────────────
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  info "Building binaries..."
  mkdir -p build/bin
  go build -o "$BINARY" ./cmd/router
  go build -o "$MOCK_BINARY" ./cmd/mockbackend
  ok "Build complete"
fi

# ── Start Mock Backends ───────────────────────────────────────────────────────
info "Starting mock backends..."

if [ -x "$MOCK_BINARY" ]; then
  MOCK_TYPE=embedding MOCK_PORT=$MOCK_EMBED_PORT "$MOCK_BINARY" &
  PIDS+=($!)

  MOCK_TYPE=reranker MOCK_PORT=$MOCK_RERANK_PORT "$MOCK_BINARY" &
  PIDS+=($!)

  MOCK_TYPE=whisper MOCK_PORT=$MOCK_WHISPER_PORT "$MOCK_BINARY" &
  PIDS+=($!)

  wait_for_port $MOCK_EMBED_PORT  "mock-embedding"
  wait_for_port $MOCK_RERANK_PORT "mock-reranker"
  wait_for_port $MOCK_WHISPER_PORT "mock-whisper"
else
  warn "No mock binary found — using real backends from env if set"
fi

# ── Start Router ──────────────────────────────────────────────────────────────
info "Starting inference-router..."

EMBEDDING_ENABLED=true \
EMBEDDING_URL="http://localhost:$MOCK_EMBED_PORT" \
RERANKER_ENABLED=true \
RERANKER_URL="http://localhost:$MOCK_RERANK_PORT" \
WHISPER_ENABLED=true \
WHISPER_URL="http://localhost:$MOCK_WHISPER_PORT" \
ROUTER_ADDR=":$ROUTER_PORT" \
ROUTER_METRICS_ADDR=":$METRICS_PORT" \
LOG_FORMAT=console \
LOG_LEVEL=warn \
"$BINARY" &
PIDS+=($!)

wait_for_port $ROUTER_PORT "inference-router"

echo ""
info "Running tests..."
echo "─────────────────────────────────────────────────────────────"

BASE="http://localhost:$ROUTER_PORT"

# ── Health checks ─────────────────────────────────────────────────────────────
if [ "$ONLY" = "all" ] || [ "$ONLY" = "health" ]; then
  echo ""
  echo "Health:"
  STATUS=$(curl -so /dev/null -w "%{http_code}" "$BASE/healthz")
  assert_status "GET /healthz" 200 "$STATUS"

  STATUS=$(curl -so /dev/null -w "%{http_code}" "$BASE/readyz")
  assert_status "GET /readyz" 200 "$STATUS"

  BODY=$(curl -sf "$BASE/healthz")
  assert_json_field "healthz.status" "['status']" "ok" "$BODY"
fi

# ── Model list ────────────────────────────────────────────────────────────────
if [ "$ONLY" = "all" ] || [ "$ONLY" = "models" ]; then
  echo ""
  echo "Models:"
  STATUS=$(curl -so /dev/null -w "%{http_code}" "$BASE/v1/models")
  assert_status "GET /v1/models" 200 "$STATUS"

  BODY=$(curl -sf "$BASE/v1/models")
  assert_json_field "models.object" "['object']" "list" "$BODY"
  COUNT=$(echo "$BODY" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))" 2>/dev/null || echo 0)
  if [ "$COUNT" -gt 0 ]; then
    ok "GET /v1/models → $COUNT models returned"
  else
    fail "GET /v1/models → 0 models returned"
  fi
fi

# ── Embeddings ────────────────────────────────────────────────────────────────
if [ "$ONLY" = "all" ] || [ "$ONLY" = "embed" ]; then
  echo ""
  echo "Embeddings:"
  BODY=$(curl -sf -X POST "$BASE/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d '{"model":"BAAI/bge-m3","input":"Hello world"}' \
    -w "\n%{http_code}" 2>/dev/null)
  STATUS=$(echo "$BODY" | tail -1)
  JSON=$(echo "$BODY" | head -n -1)
  assert_status "POST /v1/embeddings (string input)" 200 "$STATUS"
  assert_json_field "embeddings.object" "['object']" "list" "$JSON"

  # Batch input
  STATUS=$(curl -so /dev/null -w "%{http_code}" -X POST "$BASE/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d '{"model":"BAAI/bge-m3","input":["Hello","World"]}')
  assert_status "POST /v1/embeddings (batch input)" 200 "$STATUS"
fi

# ── Reranking ─────────────────────────────────────────────────────────────────
if [ "$ONLY" = "all" ] || [ "$ONLY" = "rerank" ]; then
  echo ""
  echo "Reranking:"
  BODY=$(curl -sf -X POST "$BASE/v1/rerank" \
    -H "Content-Type: application/json" \
    -d '{"model":"BAAI/bge-reranker-v2-m3","query":"What is Go?","documents":["Go is a programming language","Python is also a language","The weather is nice"]}' \
    -w "\n%{http_code}" 2>/dev/null)
  STATUS=$(echo "$BODY" | tail -1)
  JSON=$(echo "$BODY" | head -n -1)
  assert_status "POST /v1/rerank" 200 "$STATUS"

  # Also test /rerank (TEI native path)
  STATUS=$(curl -so /dev/null -w "%{http_code}" -X POST "$BASE/rerank" \
    -H "Content-Type: application/json" \
    -d '{"query":"test","texts":["doc1","doc2"]}')
  assert_status "POST /rerank (TEI native)" 200 "$STATUS"
fi

# ── Transcription ─────────────────────────────────────────────────────────────
if [ "$ONLY" = "all" ] || [ "$ONLY" = "whisper" ]; then
  echo ""
  echo "Transcription:"
  TMPWAV=$(mktemp /tmp/test-XXXXXX.wav)
  printf 'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00' > "$TMPWAV"

  STATUS=$(curl -so /dev/null -w "%{http_code}" -X POST "$BASE/v1/audio/transcriptions" \
    -F "model=whisper-large-v3-turbo" \
    -F "file=@$TMPWAV" 2>/dev/null)
  assert_status "POST /v1/audio/transcriptions" 200 "$STATUS"
  rm -f "$TMPWAV"
fi

# ── Error cases ───────────────────────────────────────────────────────────────
if [ "$ONLY" = "all" ] || [ "$ONLY" = "errors" ]; then
  echo ""
  echo "Error cases:"
  STATUS=$(curl -so /dev/null -w "%{http_code}" "$BASE/v1/nonexistent")
  assert_status "GET /v1/nonexistent → 404" 404 "$STATUS"
fi

# ── Metrics ───────────────────────────────────────────────────────────────────
if [ "$ONLY" = "all" ] || [ "$ONLY" = "metrics" ]; then
  echo ""
  echo "Metrics:"
  STATUS=$(curl -so /dev/null -w "%{http_code}" "http://localhost:$METRICS_PORT/metrics")
  assert_status "GET :$METRICS_PORT/metrics" 200 "$STATUS"

  BODY=$(curl -sf "http://localhost:$METRICS_PORT/metrics")
  if echo "$BODY" | grep -q "inference_router_requests_total"; then
    ok "Prometheus metric inference_router_requests_total present"
  else
    fail "inference_router_requests_total missing from /metrics"
  fi

  if echo "$BODY" | grep -q "inference_router_backend_up"; then
    ok "Prometheus metric inference_router_backend_up present"
  else
    fail "inference_router_backend_up missing from /metrics"
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────────────"
if [ "$FAILED" -eq 0 ]; then
  echo -e "${GREEN}All tests passed.${NC}"
else
  echo -e "${RED}$FAILED test(s) failed.${NC}"
  exit 1
fi
