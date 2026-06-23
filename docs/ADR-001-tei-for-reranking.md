# ADR-001: TEI (Rust/ONNX) for Reranking Instead of Python/FlagEmbedding

**Status:** Accepted  
**Date:** 2025-07-01  
**Author:** evombau  

---

## Context

The inference-stack provides an OpenAI-compatible reverse proxy that unifies embedding, reranking,
and speech-to-text backends behind a single endpoint. For reranking we initially shipped a
Python/FastAPI server (`reranker-server`) using the
[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) library with model
`BAAI/bge-reranker-v2-m3` (XLMRoberta-large, ~560M parameters).

When the stack was first deployed on CPU nodes in the INT cluster (RKE2, no GPU,
standard Kubernetes worker nodes), latency measurements showed the Python backend was taking
**37 seconds per `/v1/rerank` request** for a typical batch of 5–10 document pairs.
This made the reranker unusable for any latency-sensitive use case (RAG pipelines, search
re-ranking, etc.).

---

## Decision

Replace the Python/FlagEmbedding reranker server with
[Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference)
using the **ORT (ONNX Runtime) backend**.

TEI already powers the embedding backend in this stack (`tei-runtime` image, `BAAI/bge-m3`).
TEI automatically detects a `SequenceClassification` model architecture and switches to reranker
mode — no special flag is required.

The same `tei-runtime` image is reused for both embedding and reranking. The only new component
is `tei-reranker-model-init`, which extends the model init pattern (rsync of pre-baked weights)
with an additional ONNX export step, since TEI's ORT backend requires `onnx/model.onnx`.

---

## Consequences

### Performance

Measured on the INT cluster (CPU, bge-reranker-v2-m3, batch of 5 pairs):

| Backend          | Latency (warm)  | Notes                          |
|------------------|-----------------|--------------------------------|
| Python/FlagEmb.  | ~37 s           | PyTorch eager on CPU           |
| TEI/ONNX (ORT)   | **0.27–0.47 s** | ONNX Runtime, ORT backend      |

Speedup: **~100×**. This brings CPU-only reranking into a usable latency range for RAG workloads.

### Why Python/FlagEmbedding was slow

PyTorch performs tensor operations in eager mode on CPU without the fused kernel paths that
ONNX Runtime provides. BGE-reranker-v2-m3 uses XLMRoberta-large (~560M params), which is
compute-heavy for a synchronous request. FP16 does not help because x86 CPUs have no native
FP16 ALU (AVX-512 VNNI covers INT8/INT4, not FP16). FP32 and FP16 have nearly identical
throughput on CPU, meaning setting `USE_FP16=false` has no practical benefit either.

ONNX Runtime with the ORT CPU execution provider generates fused operations and uses AVX2/AVX-512
instructions natively, which is why the same model is ~100× faster.

### Architecture changes

1. **New init image `tei-reranker-model-init`** (`Containerfile.tei-reranker-model-init`):  
   Downloads `BAAI/bge-reranker-v2-m3` from Hugging Face and exports it to ONNX using
   `optimum-cli export onnx --task text-classification --opset 14`. The resulting
   `onnx/model.onnx` (~2.1 GB) is baked into the image and rsync-copied to the shared model
   volume at pod startup (same init container pattern as embedding).

2. **`reranker-server` image deprecated** — `Containerfile.reranker-server` and
   `deploy/reranker/server.py` are kept in the repository for reference but are no longer part
   of the default build. `docker make` targets `docker/reranker-model-init` and
   `docker/reranker-server` have been removed.

3. **Router path rewriting** — TEI exposes `POST /rerank`; the Python server exposed
   `POST /v1/rerank`. The router now rewrites `/v1/rerank` → `/rerank` transparently before
   forwarding to the reranker backend. Client-facing API is unchanged.

4. **Resource limits adjusted** — The ORT backend loads the full ONNX graph into memory during
   warmup. Default `--max-batch-tokens 16384` (GPU-tuned) causes OOM on CPU nodes with 8 Gi
   memory limit. Setting `--max-batch-tokens 2048` keeps RSS within the 8 Gi limit. Recommended
   deployment resources: `requests: 4Gi`, `limits: 8Gi`.

5. **Helm values changed** — `reranker.useFp16` and `reranker.maxLength` (Python-specific) are
   removed. `reranker.extraArgs` (same pattern as embedding) is used for `--max-batch-tokens`.

### Trade-offs / known limitations

- **ONNX export is slow at build time** (~15–20 min on CI). This is a one-time build cost;
  the resulting model image is cached and reused across deployments.
- **opset 14 warning** — `optimum-cli` with `--opset 14` emits a warning that XLMRoberta
  benefits from opset ≥ 18. The export and the model work correctly at opset 14; bumping to 18
  in a future image version would silence the warning.
- **Model volume size unchanged** at 4 Gi — the ONNX model is ~2.1 GB, which fits.
- **Python reranker code retained** — `deploy/reranker/server.py` and
  `deploy/Containerfile.reranker-server` are kept in the repository as reference implementations
  but are excluded from CI builds.

---

## Alternatives Considered

### Keep Python/FlagEmbedding, optimize

Options explored: FP16 (no benefit on CPU), batch size reduction, model quantization (INT8).
INT8 quantization would require a separate optimum quantization step and adds complexity.
TEI already supports the same model with better performance out of the box and aligns with the
existing embedding architecture.

### Use a different reranker model (smaller)

`bge-reranker-v2-m3` is the target model for the DFL data platform. Using a smaller model
would require a product decision outside the scope of this infrastructure component.

### GPU nodes

Not available in the current INT/PROD cluster configuration and would require significant
infrastructure changes.
