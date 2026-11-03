REGISTRY        ?= ghcr.io/layer87-labs
GO_VERSION      ?= $(shell go env GOVERSION | sed 's/^go//')
# Release tags are per component (router/202611.3.1); the image tag is the bare CalVer.
VERSION         ?= $(shell git describe --tags --exact-match 2>/dev/null | sed 's|^.*/||' | grep . || echo "dev")
# Every component has its own version in CI (see .relctl.yaml). Locally they all default to VERSION.
ROUTER_VERSION          ?= $(VERSION)
TEI_IMAGE_VERSION       ?= $(VERSION)
EMBEDDING_MODEL_VERSION ?= $(VERSION)
RERANKER_MODEL_VERSION  ?= $(VERSION)
WHISPER_IMAGE_VERSION   ?= $(VERSION)
COMMIT_HASH     ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE      ?= $(shell date +%FT%T%z)
GOFLAGS         ?= -trimpath
LDFLAGS         ?= -s -w
LDSYMS          ?= \
  -X github.com/layer87-labs/inference-stack/internal/build.Version=$(ROUTER_VERSION) \
  -X github.com/layer87-labs/inference-stack/internal/build.CommitHash=$(COMMIT_HASH) \
  -X github.com/layer87-labs/inference-stack/internal/build.BuildDate=$(BUILD_DATE)

# TEI and Whisper versions — read from Containerfile ARGs by default.
# Override on the command line if needed: make docker/tei-runtime TEI_VERSION=cpu-1.9.4
TEI_VERSION     ?= $(shell grep '^ARG TEI_VERSION=' deploy/Containerfile.tei-base | sed 's/ARG TEI_VERSION=//')
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

## ── Docker / Track A: router and TEI (CalVer per component) ─────────────────────────────────────────────────

# inference-router — CalVer tag
docker/router:
	docker build \
	  -f deploy/Containerfile.router \
	  -t $(REGISTRY)/inference-router:$(ROUTER_VERSION) \
	  .

# tei-base — CalVer tag, embeds TEI_VERSION as build arg
docker/tei-base:
	docker build \
	  -f deploy/Containerfile.tei-base \
	  --build-arg TEI_VERSION=$(TEI_VERSION) \
	  -t $(REGISTRY)/tei-base:$(TEI_IMAGE_VERSION) \
	  .

## ── Docker / Track B: TEI runtime ───────────────────────────────────────────

# tei-runtime — tag = CalVer
# Requires tei-base to be available at $(REGISTRY)/tei-base:$(TEI_IMAGE_VERSION)
docker/tei-runtime:
	docker build \
	  -f deploy/Containerfile.tei-runtime \
	  --build-arg REGISTRY=$(REGISTRY) \
	  --build-arg BASE_TAG=$(TEI_IMAGE_VERSION) \
	  --build-arg TEI_VERSION=$(TEI_VERSION) \
	  -t $(REGISTRY)/tei-runtime:$(TEI_IMAGE_VERSION) \
	  .

## ── Docker / Track C: Model init ────────────────────────────────────────────

# bge-m3 init container — tag = bge-m3-<embedding-model version>
docker/tei-model-init-embedding:
	docker build \
	  -f deploy/Containerfile.tei-model-init \
	  --build-arg MODEL_ID=BAAI/bge-m3 \
	  --build-arg MODEL_DATE=$(EMBEDDING_MODEL_VERSION) \
	  -t $(REGISTRY)/tei-model-init:bge-m3-$(EMBEDDING_MODEL_VERSION) \
	  .

## ── Docker / Track E: Reranker (TEI/ONNX) ──────────────────────────────────

# tei-reranker-model-init — reranker as ONNX for TEI's ORT backend.
# Uses the model's own onnx/model.onnx if it ships one, else optimum-cli exports it.
# Pick the model with RERANKER_MODEL_ID, e.g. RERANKER_MODEL_ID=BAAI/bge-reranker-v2-m3
# RERANKER_SLUG is the model part of the image tag (<slug>-<version>), as in the release workflow.
RERANKER_MODEL_ID ?= cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RERANKER_SLUG     ?= mmarco-mminilmv2-l12-h384-v1
docker/tei-reranker-model-init:
	docker build \
	  -f deploy/Containerfile.tei-reranker-model-init \
	  --build-arg MODEL_ID=$(RERANKER_MODEL_ID) \
	  --build-arg MODEL_DATE=$(RERANKER_MODEL_VERSION) \
	  -t $(REGISTRY)/tei-reranker-model-init:$(RERANKER_SLUG)-$(RERANKER_MODEL_VERSION) \
	  .

## ── Docker / Track D: Whisper ───────────────────────────────────────────────

# whisper — tag = <whisper model>-<whisper version>
docker/whisper:
	docker build \
	  -f deploy/Containerfile.whisper \
	  --build-arg WHISPER_MODEL=$(WHISPER_MODEL) \
	  -t $(REGISTRY)/whisper:$(WHISPER_MODEL)-$(WHISPER_IMAGE_VERSION) \
	  .

## ── Build all ─────────────────────────────────────────────────────────────────

docker: docker/router docker/tei-base docker/tei-runtime \
        docker/tei-model-init-embedding \
        docker/tei-reranker-model-init \
        docker/whisper

## ── Push ─────────────────────────────────────────────────────────────────────

push/router:
	docker push $(REGISTRY)/inference-router:$(ROUTER_VERSION)

push/tei:
	docker push $(REGISTRY)/tei-base:$(TEI_IMAGE_VERSION)
	docker push $(REGISTRY)/tei-runtime:$(TEI_IMAGE_VERSION)
	docker push $(REGISTRY)/tei-model-init:bge-m3-$(EMBEDDING_MODEL_VERSION)

push/reranker:
	docker push $(REGISTRY)/tei-reranker-model-init:$(RERANKER_SLUG)-$(RERANKER_MODEL_VERSION)

push/whisper:
	docker push $(REGISTRY)/whisper:$(WHISPER_MODEL)-$(WHISPER_IMAGE_VERSION)

push: push/router push/tei push/reranker push/whisper

## ── Info ──────────────────────────────────────────────────────────────────────

versions:
	@echo "Each component has its own CalVer (default: $(VERSION)):"
	@echo ""
	@echo "  inference-router        → $(REGISTRY)/inference-router:$(ROUTER_VERSION)"
	@echo "  tei-base                → $(REGISTRY)/tei-base:$(TEI_IMAGE_VERSION)"
	@echo "  tei-runtime             → $(REGISTRY)/tei-runtime:$(TEI_IMAGE_VERSION)"
	@echo "  tei-model-init          → $(REGISTRY)/tei-model-init:bge-m3-$(EMBEDDING_MODEL_VERSION)"
	@echo "  tei-reranker-model-init → $(REGISTRY)/tei-reranker-model-init:$(RERANKER_SLUG)-$(RERANKER_MODEL_VERSION)"
	@echo "  whisper                 → $(REGISTRY)/whisper:$(WHISPER_MODEL)-$(WHISPER_IMAGE_VERSION)"
