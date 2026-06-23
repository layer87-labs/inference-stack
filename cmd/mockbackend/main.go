// cmd/mockbackend/main.go — Mock backend for local testing of inference-router.
//
// Simulates TEI embedding, reranker, and Whisper backends with valid
// OpenAI-compatible responses. Controlled by environment variables:
//
//	MOCK_TYPE  — "embedding" | "reranker" | "whisper"  (required)
//	MOCK_PORT  — listen port (default 9001)
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"go.uber.org/zap"
)

func main() {
	log, _ := zap.NewDevelopment()
	defer log.Sync() //nolint:errcheck

	mockType := os.Getenv("MOCK_TYPE")
	if mockType == "" {
		log.Fatal("MOCK_TYPE must be set (embedding | reranker | whisper)")
	}

	port := os.Getenv("MOCK_PORT")
	if port == "" {
		port = "9001"
	}

	mux := http.NewServeMux()
	registerHealth(mux)

	switch mockType {
	case "embedding":
		registerEmbedding(mux, log)
	case "reranker":
		registerReranker(mux, log)
	case "whisper":
		registerWhisper(mux, log)
	default:
		log.Fatal("unknown MOCK_TYPE", zap.String("type", mockType))
	}

	addr := ":" + port
	log.Info("mock backend starting", zap.String("type", mockType), zap.String("addr", addr))
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal("server error", zap.Error(err))
	}
}

// ── Shared ────────────────────────────────────────────────────────────────────

func registerHealth(mux *http.ServeMux) {
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

// ── Embedding ─────────────────────────────────────────────────────────────────

func registerEmbedding(mux *http.ServeMux, log *zap.Logger) {
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": "BAAI/bge-m3", "object": "model", "created": time.Now().Unix(), "owned_by": "inference-stack"},
			},
		})
	})

	embedHandler := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		log.Debug("embedding request", zap.String("path", r.URL.Path))
		writeJSON(w, 200, map[string]any{
			"object": "list",
			"data": []map[string]any{
				{
					"object":    "embedding",
					"index":     0,
					"embedding": []float64{0.1, 0.2, 0.3, 0.4, 0.5},
				},
			},
			"model": "BAAI/bge-m3",
			"usage": map[string]int{"prompt_tokens": 5, "total_tokens": 5},
		})
	}
	mux.HandleFunc("/v1/embeddings", embedHandler)
	mux.HandleFunc("/embed", embedHandler)
	mux.HandleFunc("/embed_sparse", embedHandler)
}

// ── Reranker ──────────────────────────────────────────────────────────────────

func registerReranker(mux *http.ServeMux, log *zap.Logger) {
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": "BAAI/bge-reranker-v2-m3", "object": "model", "created": time.Now().Unix(), "owned_by": "inference-stack"},
			},
		})
	})

	rerankHandler := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		log.Debug("rerank request", zap.String("path", r.URL.Path))
		writeJSON(w, 200, map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"index": 0, "relevance_score": 0.95, "document": map[string]string{"text": "doc0"}},
				{"index": 1, "relevance_score": 0.42, "document": map[string]string{"text": "doc1"}},
			},
			"model": "BAAI/bge-reranker-v2-m3",
			"usage": map[string]int{"prompt_tokens": 20, "total_tokens": 20},
		})
	}
	mux.HandleFunc("/v1/rerank", rerankHandler)
	mux.HandleFunc("/rerank", rerankHandler)
}

// ── Whisper ───────────────────────────────────────────────────────────────────

func registerWhisper(mux *http.ServeMux, log *zap.Logger) {
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		now := time.Now().Unix()
		writeJSON(w, 200, map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": "whisper-large-v3-turbo", "object": "model", "created": now, "owned_by": "inference-stack"},
				{"id": "whisper-large-v3", "object": "model", "created": now, "owned_by": "inference-stack"},
				{"id": "whisper-medium", "object": "model", "created": now, "owned_by": "inference-stack"},
				{"id": "whisper-small", "object": "model", "created": now, "owned_by": "inference-stack"},
			},
		})
	})

	mux.HandleFunc("/v1/audio/transcriptions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		log.Debug("transcription request")
		_ = r.ParseMultipartForm(32 << 20)
		format := r.FormValue("response_format")
		if format == "text" {
			w.Header().Set("Content-Type", "text/plain")
			fmt.Fprint(w, "Hello from mock whisper.")
			return
		}
		writeJSON(w, 200, map[string]any{
			"text": "Hello from mock whisper.",
		})
	})
}
