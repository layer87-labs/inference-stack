REGISTRY        ?= ghcr.io/layer87-labs
GO_VERSION      ?= $(shell go env GOVERSION | sed 's/^go//')
VERSION         ?= $(shell git describe --tags --exact-match 2>/dev/null || echo "dev")
COMMIT_HASH     ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE      ?= $(shell date +%FT%T%z)
MODEL_DATE      ?= $(shell date +%Y.%m)
GOFLAGS         ?= -trimpath
LDFLAGS         ?= -s -w
LDSYMS          ?= \
  -X github.com/layer87-labs/inference-stack/internal/build.Version=$(VERSION) \
  -X github.com/layer87-labs/inference-stack/internal/build.CommitHash=$(COMMIT_HASH) \
  -X github.com/layer87-labs/inference-stack/internal/build.BuildDate=$(BUILD_DATE)

# TEI and Whisper versions — read from Containerfile ARGs by default.
# Override on the command line if needed: make docker/tei-runtime TEI_VERSION=cpu-1.9.4
TEI_VERSION     ?= $(shell grep '^ARG TEI_VERSION=' deploy/Containerfile.tei-base | sed 's/ARG TEI_VERSION=//')
WHISPER_VERSION ?= $(shell grep '^ARG WHISPER_VERSION=' deploy/Containerfile.whisper | sed 's/ARG WHISPER_VERSION=//')
WHISPER_MODEL   ?= $(shell grep '^ARG WHISPER_MODEL=' deploy/Containerfile.whisper | sed 's/ARG WHISPER_MODEL=//')

.PHONY: build build/router build/mockbackend tidy lint test test-local \
        docker docker/router \
        docker/tei-base docker/tei-runtime \
        docker/tei-model-init-embedding \
        docker/tei-reranker-model-init \
        docker/whisper \
        push push/router push/tei push/reranker push/whisper \
        versions

## ── Go ────────────────────────────────────────────────────────────────────────

build: build/router build/mockbackend

build/router:
	CGO_ENABLED=0 go build $(GOFLAGS) \
	  -ldflags="$(LDFLAGS) $(LDSYMS)" \
	  -o build/bin/inference-router ./cmd/router

build/mockbackend:
	CGO_ENABLED=0 go build $(GOFLAGS) \
	  -ldflags="$(LDFLAGS) $(LDSYMS)" \
	  -o build/bin/mock-backend ./cmd/mockbackend

tidy:
	go mod tidy

lint:
	golangci-lint run ./...

test:
	go test -race ./...

test-local: build
	bash scripts/test-local.sh

test-containers:
	bash scripts/test-containers.sh

## ── Docker / Track A: CalVer ─────────────────────────────────────────────────

# inference-router — CalVer tag
docker/router:
	docker build \
	  -f deploy/Containerfile.router \
	  -t $(REGISTRY)/inference-router:$(VERSION) \
	  .

# tei-base — CalVer tag, embeds TEI_VERSION as build arg
docker/tei-base:
	docker build \
	  -f deploy/Containerfile.tei-base \
	  --build-arg TEI_VERSION=$(TEI_VERSION) \
	  -t $(REGISTRY)/tei-base:$(VERSION) \
	  .

## ── Docker / Track B: TEI runtime ───────────────────────────────────────────

# tei-runtime — tag = CalVer
# Requires tei-base to be available at $(REGISTRY)/tei-base:$(VERSION)
docker/tei-runtime:
	docker build \
	  -f deploy/Containerfile.tei-runtime \
	  --build-arg REGISTRY=$(REGISTRY) \
	  --build-arg BASE_TAG=$(VERSION) \
	  --build-arg TEI_VERSION=$(TEI_VERSION) \
	  -t $(REGISTRY)/tei-runtime:$(VERSION) \
	  .

## ── Docker / Track C: Model init ────────────────────────────────────────────

# bge-m3 init container — tag = CalVer
docker/tei-model-init-embedding:
	docker build \
	  -f deploy/Containerfile.tei-model-init \
	  --build-arg MODEL_ID=BAAI/bge-m3 \
	  --build-arg MODEL_DATE=$(MODEL_DATE) \
	  -t $(REGISTRY)/tei-model-init:$(VERSION) \
	  .

## ── Docker / Track E: Reranker (TEI/ONNX) ──────────────────────────────────

# tei-reranker-model-init — ONNX-exported bge-reranker-v2-m3
# Uses optimum-cli to export SafeTensors → ONNX (required by TEI ORT backend).
docker/tei-reranker-model-init:
	docker build \
	  -f deploy/Containerfile.tei-reranker-model-init \
	  --build-arg MODEL_ID=BAAI/bge-reranker-v2-m3 \
	  --build-arg MODEL_DATE=$(MODEL_DATE) \
	  -t $(REGISTRY)/tei-reranker-model-init:$(VERSION) \
	  .

## ── Docker / Track D: Whisper ───────────────────────────────────────────────

# whisper — tag = CalVer
docker/whisper:
	docker build \
	  -f deploy/Containerfile.whisper \
	  --build-arg WHISPER_VERSION=$(WHISPER_VERSION) \
	  --build-arg WHISPER_MODEL=$(WHISPER_MODEL) \
	  -t $(REGISTRY)/whisper:$(VERSION) \
	  .

## ── Build all ─────────────────────────────────────────────────────────────────

docker: docker/router docker/tei-base docker/tei-runtime \
        docker/tei-model-init-embedding \
        docker/tei-reranker-model-init \
        docker/whisper

## ── Push ─────────────────────────────────────────────────────────────────────

push/router:
	docker push $(REGISTRY)/inference-router:$(VERSION)

push/tei:
	docker push $(REGISTRY)/tei-base:$(VERSION)
	docker push $(REGISTRY)/tei-runtime:$(VERSION)
	docker push $(REGISTRY)/tei-model-init:$(VERSION)

push/reranker:
	docker push $(REGISTRY)/tei-reranker-model-init:$(VERSION)

push/whisper:
	docker push $(REGISTRY)/whisper:$(VERSION)

push: push/router push/tei push/reranker push/whisper

## ── Info ──────────────────────────────────────────────────────────────────────

versions:
	@echo "All images use CalVer: $(VERSION)"
	@echo ""
	@echo "  inference-router    → $(REGISTRY)/inference-router:$(VERSION)"
	@echo "  tei-base            → $(REGISTRY)/tei-base:$(VERSION)"
	@echo "  tei-runtime         → $(REGISTRY)/tei-runtime:$(VERSION)"
	@echo "  tei-model-init      → $(REGISTRY)/tei-model-init:$(VERSION)"
  @echo "  tei-reranker-model-init → $(REGISTRY)/tei-reranker-model-init:$(VERSION)"
	@echo "  whisper             → $(REGISTRY)/whisper:$(VERSION)"
