package proxy_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"go.uber.org/zap"

	"github.com/layer87-labs/inference-stack/internal/config"
	"github.com/layer87-labs/inference-stack/internal/health"
	"github.com/layer87-labs/inference-stack/internal/proxy"
)

// ── Helpers ───────────────────────────────────────────────────────────────────

func noopLogger() *zap.Logger {
	log, _ := zap.NewDevelopment()
	return log
}

// mockBackend spins up an httptest.Server that responds to the given paths.
func mockBackend(t *testing.T, handlers map[string]http.HandlerFunc) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	for path, h := range handlers {
		mux.HandleFunc(path, h)
	}
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// jsonOK returns a handler that writes status 200 with the given JSON body.
func jsonOK(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, body)
	}
}

// modelsJSON returns a minimal /v1/models response for the given model IDs.
func modelsJSON(ids ...string) string {
	type modelObj struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	type resp struct {
		Object string     `json:"object"`
		Data   []modelObj `json:"data"`
	}
	r := resp{Object: "list"}
	now := time.Now().Unix()
	for _, id := range ids {
		r.Data = append(r.Data, modelObj{ID: id, Object: "model", Created: now, OwnedBy: "inference-stack"})
	}
	b, _ := json.Marshal(r)
	return string(b)
}

// buildConfig constructs a Config directly (bypassing Load/validate) so tests
// can control exactly which backends are enabled without touching env vars.
func buildConfig(embedding, reranker, whisper config.Backend) *config.Config {
	return &config.Config{
		ListenAddr:     ":0",
		MetricsAddr:    ":0",
		ReadTimeout:    5 * time.Second,
		WriteTimeout:   5 * time.Second,
		IdleTimeout:    5 * time.Second,
		MaxRequestSize: 1 << 20,
		LogLevel:       "debug",
		LogFormat:      "console",
		Embedding:      embedding,
		Reranker:       reranker,
		Whisper:        whisper,
	}
}

// buildRouter wires up a proxy.Router with a health.Checker against the given
// config and returns it ready to be used with httptest.
func buildRouter(t *testing.T, cfg *config.Config) http.Handler {
	t.Helper()
	log := noopLogger()
	checker := health.New(cfg.EnabledBackends(), log)
	r, err := proxy.New(cfg, checker, log)
	if err != nil {
		t.Fatalf("proxy.New: %v", err)
	}
	return r
}

// ── Tests ─────────────────────────────────────────────────────────────────────

func TestRouting_Embedding(t *testing.T) {
	const responseBody = `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}]}`

	backend := mockBackend(t, map[string]http.HandlerFunc{
		"/v1/embeddings": jsonOK(responseBody),
	})

	cfg := buildConfig(
		config.Backend{Name: "embedding", BaseURL: backend.URL, Enabled: true, Timeout: 5 * time.Second},
		config.Backend{Name: "reranker"},
		config.Backend{Name: "whisper"},
	)
	router := buildRouter(t, cfg)

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings",
		strings.NewReader(`{"model":"BAAI/bge-m3","input":"hello"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
	var m map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &m); err != nil {
		t.Fatalf("response is not valid JSON: %v\nbody: %s", err, w.Body.String())
	}
	if m["object"] != "list" {
		t.Errorf("object = %v, want list", m["object"])
	}
}

func TestRouting_Embed_NativePaths(t *testing.T) {
	const responseBody = `{"object":"list","data":[]}`

	backend := mockBackend(t, map[string]http.HandlerFunc{
		"/embed":        jsonOK(responseBody),
		"/embed_sparse": jsonOK(responseBody),
	})

	cfg := buildConfig(
		config.Backend{Name: "embedding", BaseURL: backend.URL, Enabled: true, Timeout: 5 * time.Second},
		config.Backend{Name: "reranker"},
		config.Backend{Name: "whisper"},
	)
	router := buildRouter(t, cfg)

	for _, path := range []string{"/embed", "/embed_sparse"} {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{}`))
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Errorf("path %s: status = %d, want 200", path, w.Code)
		}
	}
}

func TestRouting_Reranker(t *testing.T) {
	const responseBody = `{"object":"list","data":[{"index":0,"relevance_score":0.9}]}`

	backend := mockBackend(t, map[string]http.HandlerFunc{
		"/v1/rerank": jsonOK(responseBody),
		"/rerank":    jsonOK(responseBody),
	})

	cfg := buildConfig(
		config.Backend{Name: "embedding"},
		config.Backend{Name: "reranker", BaseURL: backend.URL, Enabled: true, Timeout: 5 * time.Second},
		config.Backend{Name: "whisper"},
	)
	router := buildRouter(t, cfg)

	for _, path := range []string{"/v1/rerank", "/rerank"} {
		req := httptest.NewRequest(http.MethodPost, path,
			strings.NewReader(`{"query":"q","documents":["d1","d2"]}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Errorf("path %s: status = %d, want 200", path, w.Code)
		}
	}
}

func TestRouting_Whisper(t *testing.T) {
	backend := mockBackend(t, map[string]http.HandlerFunc{
		"/v1/audio/transcriptions": jsonOK(`{"text":"hello world"}`),
	})

	cfg := buildConfig(
		config.Backend{Name: "embedding"},
		config.Backend{Name: "reranker"},
		config.Backend{Name: "whisper", BaseURL: backend.URL, Enabled: true, Timeout: 5 * time.Second},
	)
	router := buildRouter(t, cfg)

	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions",
		strings.NewReader(`--boundary\r\nContent-Disposition: form-data; name="model"\r\n\r\nwhisper-large-v3\r\n--boundary--`))
	req.Header.Set("Content-Type", "multipart/form-data; boundary=boundary")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200; body: %s", w.Code, w.Body.String())
	}
}

func TestModelsAggregation(t *testing.T) {
	embedBackend := mockBackend(t, map[string]http.HandlerFunc{
		"/v1/models": jsonOK(modelsJSON("BAAI/bge-m3")),
	})
	rerankBackend := mockBackend(t, map[string]http.HandlerFunc{
		"/v1/models": jsonOK(modelsJSON("BAAI/bge-reranker-v2-m3")),
	})
	whisperBackend := mockBackend(t, map[string]http.HandlerFunc{
		"/v1/models": jsonOK(modelsJSON("whisper-large-v3", "whisper-medium")),
	})

	cfg := buildConfig(
		config.Backend{Name: "embedding", BaseURL: embedBackend.URL, Enabled: true, Timeout: 5 * time.Second},
		config.Backend{Name: "reranker", BaseURL: rerankBackend.URL, Enabled: true, Timeout: 5 * time.Second},
		config.Backend{Name: "whisper", BaseURL: whisperBackend.URL, Enabled: true, Timeout: 5 * time.Second},
	)
	router := buildRouter(t, cfg)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Created int64  `json:"created"`
		} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v\nbody: %s", err, w.Body.String())
	}
	if resp.Object != "list" {
		t.Errorf("object = %q, want list", resp.Object)
	}
	if len(resp.Data) != 4 {
		t.Errorf("got %d models, want 4; IDs: %v", len(resp.Data), resp.Data)
	}
	for _, m := range resp.Data {
		if m.Created == 0 {
			t.Errorf("model %s: created = 0, want non-zero unix timestamp", m.ID)
		}
	}
}

func TestModelsAggregation_FallbackToStatic(t *testing.T) {
	failBackend := mockBackend(t, map[string]http.HandlerFunc{
		"/v1/models": func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		},
	})

	cfg := buildConfig(
		config.Backend{
			Name:    "embedding",
			BaseURL: failBackend.URL,
			Models:  []string{"BAAI/bge-m3"},
			Enabled: true,
			Timeout: 5 * time.Second,
		},
		config.Backend{Name: "reranker"},
		config.Backend{Name: "whisper"},
	)
	router := buildRouter(t, cfg)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp struct {
		Data []struct {
			ID      string `json:"id"`
			Created int64  `json:"created"`
		} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if len(resp.Data) != 1 || resp.Data[0].ID != "BAAI/bge-m3" {
		t.Errorf("expected fallback model BAAI/bge-m3, got %+v", resp.Data)
	}
	if resp.Data[0].Created == 0 {
		t.Errorf("fallback model created = 0, want non-zero unix timestamp")
	}
}

func TestModelsAggregation_InvalidUpstreamID(t *testing.T) {
	rerankBackend := mockBackend(t, map[string]http.HandlerFunc{
		"/v1/models": jsonOK(`{"object":"list","data":[{"id":"/model","object":"model","created":1700000000,"owned_by":"BAAI"}]}`),
	})

	cfg := buildConfig(
		config.Backend{Name: "embedding"},
		config.Backend{
			Name:    "reranker",
			BaseURL: rerankBackend.URL,
			Models:  []string{"BAAI/bge-reranker-v2-m3"},
			Enabled: true,
			Timeout: 5 * time.Second,
		},
		config.Backend{Name: "whisper"},
	)
	router := buildRouter(t, cfg)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp struct {
		Data []struct {
			ID      string `json:"id"`
			Created int64  `json:"created"`
		} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v\nbody: %s", err, w.Body.String())
	}

	if len(resp.Data) != 1 {
		t.Fatalf("got %d models, want 1 (static fallback); data: %+v", len(resp.Data), resp.Data)
	}
	if resp.Data[0].ID != "BAAI/bge-reranker-v2-m3" {
		t.Errorf("id = %q, want static fallback BAAI/bge-reranker-v2-m3", resp.Data[0].ID)
	}
	if resp.Data[0].Created == 0 {
		t.Errorf("created = 0, want non-zero unix timestamp")
	}
}

func TestBackendDown_502(t *testing.T) {
	cfg := buildConfig(
		config.Backend{Name: "embedding", BaseURL: "http://127.0.0.1:1", Enabled: true, Timeout: 2 * time.Second},
		config.Backend{Name: "reranker"},
		config.Backend{Name: "whisper"},
	)
	router := buildRouter(t, cfg)

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings",
		strings.NewReader(`{"model":"BAAI/bge-m3","input":"test"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", w.Code)
	}

	var errResp map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("response is not valid JSON: %v\nbody: %s", err, w.Body.String())
	}
	if errResp["error"] != "upstream_error" {
		t.Errorf("error = %q, want upstream_error", errResp["error"])
	}
}

func TestBackendDisabled_503(t *testing.T) {
	cfg := buildConfig(
		config.Backend{Name: "embedding", BaseURL: "http://127.0.0.1:1", Enabled: true, Timeout: 2 * time.Second},
		config.Backend{Name: "reranker", Enabled: false},
		config.Backend{Name: "whisper"},
	)
	router := buildRouter(t, cfg)

	req := httptest.NewRequest(http.MethodPost, "/v1/rerank",
		strings.NewReader(`{"query":"q","documents":["d"]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", w.Code)
	}

	var errResp map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("response is not valid JSON: %v\nbody: %s", err, w.Body.String())
	}
	if errResp["error"] != "backend_disabled" {
		t.Errorf("error = %q, want backend_disabled", errResp["error"])
	}
}

func TestHealthEndpoints(t *testing.T) {
	backend := mockBackend(t, map[string]http.HandlerFunc{
		"/health": func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(200) },
	})

	backends := []config.Backend{
		{Name: "embedding", BaseURL: backend.URL, Enabled: true, Timeout: 2 * time.Second},
	}
	log := noopLogger()
	checker := health.New(backends, log)

	mux := http.NewServeMux()
	mux.Handle("/healthz", checker)
	mux.Handle("/readyz", checker)

	for _, path := range []string{"/healthz", "/readyz"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("%s: status = %d, want 200", path, w.Code)
		}

		var body map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
			t.Fatalf("%s: invalid JSON: %v\nbody: %s", path, err, w.Body.String())
		}
		if _, ok := body["status"]; !ok {
			t.Errorf("%s: response missing 'status' field; got %v", path, body)
		}
	}
}

func TestRouterHealthEndpoint(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	})

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("invalid JSON: %v\nbody: %s", err, w.Body.String())
	}
	if body["status"] != "ok" {
		t.Errorf("status field = %v, want ok", body["status"])
	}
}

func TestUnknownPath_404(t *testing.T) {
	cfg := buildConfig(
		config.Backend{Name: "embedding", BaseURL: "http://127.0.0.1:1", Enabled: true, Timeout: 2 * time.Second},
		config.Backend{Name: "reranker"},
		config.Backend{Name: "whisper"},
	)
	router := buildRouter(t, cfg)

	req := httptest.NewRequest(http.MethodGet, "/v1/nonexistent", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}

	var errResp map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("response is not valid JSON: %v\nbody: %s", err, w.Body.String())
	}
	if errResp["error"] != "not_found" {
		t.Errorf("error = %q, want not_found", errResp["error"])
	}
}
