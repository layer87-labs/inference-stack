# CPU tuning and sizing

Measurements and settings for running the stack on plain CPU. The Helm values referenced here are in
[`deploy/helm/values.yaml`](../deploy/helm/values.yaml); see also [Configuration](configuration.md).

## Embedding (TEI + BGE-M3)

| `--max-batch-tokens` | RSS (fp32) | Recommendation       |
| -------------------- | ---------- | -------------------- |
| 4096                 | 3-6Gi      | **CPU default**      |
| 8192                 | 6-10Gi     | May OOM on 8Gi limit |
| 16384                | 10-16Gi    | GPU-only             |

## Reranker model choice

The reranker model is baked into the `tei-reranker-model-init` image and
selected by its tag (`reranker.modelInit.image.tag`). Both models run on the
same TEI runtime with the ONNX Runtime backend.

Measured with TEI `cpu-1.9.4`, limited to 4 CPUs, `--max-batch-tokens 2048`,
one `/rerank` request with one query and 20 documents of 800 characters each
(German/English mix), 4 runs per model. The CPU was an older AVX2 machine, so
only the ratios carry over.

| Model                                                  | Format                    | Parameters | Latency (20 docs) |
| ------------------------------------------------------ | ------------------------- | ---------- | ----------------- |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (default) | fp32 ONNX                 | 118M       | 0.76 to 0.85 s    |
| `BAAI/bge-reranker-v2-m3`                              | fp32 ONNX                 | 568M       | 8.3 to 11.9 s     |
| `BAAI/bge-reranker-v2-m3`                              | int8 ONNX                 | 568M       | 7.9 to 8.2 s      |
| `Alibaba-NLP/gte-multilingual-reranker-base`           | safetensors (candle path) | n/a        | 11.1 to 12.1 s    |

That is about 0.04 s per document pair for the default against roughly 0.4 to
0.5 s for `bge-reranker-v2-m3`, so reranking 40 candidates with the larger
model does not fit an interactive latency budget on 4 CPUs.

Quality, on a downstream 75-question German/English relevance set (hybrid
retrieval, rerank the top 20 candidates, 800 characters per document):

| Setup                                      | hit@1 | hit@5 | hit@10 |
| ------------------------------------------ | ----- | ----- | ------ |
| No rerank                                  | 77%   | 95%   | 96%    |
| `BAAI/bge-reranker-v2-m3`                  | 92%   | 96%   | 96%    |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 92%   | 96%   | 96%    |

This set is small and saturated (one question is 1.3 points), its documents
were short, and the embeddings used for candidate retrieval were a simple
stand-in, so it cannot separate the two models. Real-world German technical
text may favour the larger model; measure on your own data if quality matters
more than latency.

### Sizing the request for the default model

Latency is dominated by the number of passages and their length, and one
request already uses all CPUs, so concurrent requests queue. Measured with the
default model on a 4-vCPU pod (TEI `cpu-1.9.4`, `--max-batch-tokens 2048`),
75 queries, passages of title plus text cut to the given length (800
characters are about 190 tokens, 500 about 120), p95 latency of one `/rerank`
call:

| Passages | 800 chars | 600 chars | 500 chars | 400 chars |
| -------- | --------- | --------- | --------- | --------- |
| 40       | 3.2 s     | 2.5 s     | 2.1 s     | 1.8 s     |
| 30       | 2.4 s     | 1.9 s     | 1.7 s     | 1.4 s     |
| 20       | 1.7 s     | 1.3 s     | 1.1 s     | 1.0 s     |
| 15       | 1.3 s     | 1.0 s     | 0.9 s     | 0.7 s     |

With four such requests in flight at once, p95 is about 3 to 4 times higher
(for example 3.0 s for 15 passages of 500 characters, 6.4 s for 40 of 400).
If the budget is about 1.5 s per request, send at most 15 to 20 passages of
about 500 characters, and scale out (`reranker.replicaCount`) or add CPU for
concurrent callers. Reranking a short list also bounds what it can gain: it can
only reorder the passages it receives.

Memory: the working set stayed at about 1.4 GiB, including four concurrent
requests of 40 passages, so the chart defaults to a 1.5 GiB request and a 2 GiB
limit. Inputs longer than the model's 512-token limit are truncated by default
(`truncate: true`, or the TEI default when the field is omitted); with
`truncate: false` the request fails with HTTP 422.

The model repository also ships quantised ONNX variants (for example
`onnx/model_qint8_avx512_vnni.onnx`, `onnx/model_quint8_avx2.onnx`). They are
not used by this chart and were not measured; they are CPU-specific and
would need to be baked as `onnx/model.onnx` for TEI.

Licence: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` is Apache-2.0,
`BAAI/bge-reranker-v2-m3` is Apache-2.0 as well (as tagged on the Hugging Face hub); check the model card of
anything else you select.

To use the higher-quality model (more CPU or a GPU), build or pull the matching
init image and set:

```yaml
reranker:
  modelName: "BAAI/bge-reranker-v2-m3"
  modelInit:
    image:
      tag: "bge-reranker-v2-m3-<version>"   # e.g. bge-reranker-v2-m3-202611.3.1, see the published tags
```

The init step copies with `--ignore-existing`, so when you change the model on
an existing release, delete the reranker model PVC (or install with
`reranker.persistence.enabled=false`) so the old model files are not reused.
`BAAI/bge-reranker-v2-m3` needs `requests: 4Gi` / `limits: 8Gi` memory with
`--max-batch-tokens 2048`; the default model needs far less.

To build an init image yourself: `make docker/tei-reranker-model-init
RERANKER_MODEL_ID=<huggingface id>`. If the model repository ships
`onnx/model.onnx` it is used as-is; otherwise the model is exported with
Optimum. Only XLM-RoBERTa-style sequence-classification models that TEI
supports will work.

## Whisper (faster-whisper + large-v3-turbo, int8)

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

## Scheduling (`affinity`, `topologySpreadConstraints`)

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
