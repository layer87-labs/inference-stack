package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/layer87-labs/inference-stack/internal/config"
	"github.com/layer87-labs/inference-stack/internal/health"
	"github.com/layer87-labs/inference-stack/internal/metrics"
)

// Router is the main reverse-proxy handler. It dispatches incoming requests
// to the correct backend based on the request path and enabled backends.
type Router struct {
	cfg       *config.Config
	checker   *health.Checker
	log       *zap.Logger
	proxies   map[string]*httputil.ReverseProxy // keyed by backend name
	startedAt int64                             // unix timestamp, used as fallback "created" for /v1/models
}

// New creates a Router and pre-builds one ReverseProxy per enabled backend.
func New(cfg *config.Config, checker *health.Checker, log *zap.Logger) (*Router, error) {
	r := &Router{
		cfg:       cfg,
		checker:   checker,
		log:       log,
		proxies:   make(map[string]*httputil.ReverseProxy),
		startedAt: time.Now().Unix(),
	}

	for _, b := range cfg.EnabledBackends() {
		target, err := url.Parse(b.BaseURL)
		if err != nil {
			return nil, fmt.Errorf("invalid backend URL for %s: %w", b.Name, err)
		}

		backend := b // capture
		rp := &httputil.ReverseProxy{
			Director: func(req *http.Request) {
				req.URL.Scheme = target.Scheme
				req.URL.Host = target.Host
				req.Host = target.Host

				// Strip x-forwarded headers from untrusted upstream responses
				req.Header.Del("X-Forwarded-For")
				req.Header.Set("X-Forwarded-Host", req.Host)
				req.Header.Set("User-Agent", "inference-router/1.0")
			},
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout:   5 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,
				MaxIdleConns:          200,
				MaxIdleConnsPerHost:   100,
				IdleConnTimeout:       90 * time.Second,
				TLSHandshakeTimeout:   10 * time.Second,
				ExpectContinueTimeout: 1 * time.Second,
				ResponseHeaderTimeout: backend.Timeout,
			},
			ErrorHandler: func(w http.ResponseWriter, req *http.Request, err error) {
				errType := classifyError(err)
				metrics.UpstreamErrors.WithLabelValues(backend.Name, errType).Inc()
				log.Error("upstream error",
					zap.String("backend", backend.Name),
					zap.String("path", req.URL.Path),
					zap.String("error_type", errType),
					zap.Error(err),
				)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadGateway)
				json.NewEncoder(w).Encode(map[string]string{
					"error":   "upstream_error",
					"message": fmt.Sprintf("backend %s unavailable: %s", backend.Name, errType),
				})
			},
			ModifyResponse: func(resp *http.Response) error {
				// Track response size when Content-Length is known
				if cl := resp.Header.Get("Content-Length"); cl != "" {
					if n, err := strconv.ParseInt(cl, 10, 64); err == nil {
						metrics.ResponseSizeBytes.WithLabelValues(backend.Name).Observe(float64(n))
					}
				}
				return nil
			},
		}
		r.proxies[b.Name] = rp
	}

	return r, nil
}

// ServeHTTP is the main entry point. It routes based on path prefix.
func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	path := req.URL.Path

	switch {
	case path == "/v1/models" || path == "/v1/models/":
		r.handleModels(w, req)

	case strings.HasPrefix(path, "/v1/embeddings") || path == "/embed" || path == "/embed_sparse":
		r.dispatch(w, req, r.cfg.Embedding, "embedding")

	case strings.HasPrefix(path, "/rerank") || strings.HasPrefix(path, "/v1/rerank"):
		r.dispatch(w, req, r.cfg.Reranker, "reranker")

	case strings.HasPrefix(path, "/v1/audio/transcriptions") || strings.HasPrefix(path, "/v1/audio/translations"):
		r.dispatch(w, req, r.cfg.Whisper, "whisper")

	default:
		r.log.Debug("no route matched", zap.String("path", path))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{
			"error":   "not_found",
			"message": fmt.Sprintf("no backend handles path %s", path),
		})
	}
}

// dispatch proxies the request to the given backend with full observability.
func (r *Router) dispatch(w http.ResponseWriter, req *http.Request, b config.Backend, name string) {
	if !b.Enabled {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{
			"error":   "backend_disabled",
			"message": fmt.Sprintf("backend %s is not enabled", name),
		})
		return
	}

	rp, ok := r.proxies[name]
	if !ok {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	// Track request size
	if req.ContentLength > 0 {
		metrics.RequestSizeBytes.WithLabelValues(name).Observe(float64(req.ContentLength))
	}

	metrics.ActiveRequests.WithLabelValues(name).Inc()
	defer metrics.ActiveRequests.WithLabelValues(name).Dec()

	start := time.Now()
	wrappedW := &statusWriter{ResponseWriter: w}

	// Apply per-backend timeout
	ctx, cancel := context.WithTimeout(req.Context(), b.Timeout)
	defer cancel()

	rp.ServeHTTP(wrappedW, req.WithContext(ctx))

	duration := time.Since(start).Seconds()
	status := strconv.Itoa(wrappedW.status)
	path := req.URL.Path

	metrics.RequestsTotal.WithLabelValues(name, path, status).Inc()
	metrics.RequestDuration.WithLabelValues(name, path).Observe(duration)

	r.log.Info("proxied request",
		zap.String("backend", name),
		zap.String("method", req.Method),
		zap.String("path", path),
		zap.Int("status", wrappedW.status),
		zap.Float64("duration_s", duration),
		zap.Int64("bytes_in", req.ContentLength),
		zap.String("remote_addr", req.RemoteAddr),
	)
}

// modelObj mirrors the OpenAI /v1/models entry shape.
type modelObj struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type modelsResp struct {
	Object string     `json:"object"`
	Data   []modelObj `json:"data"`
}

// handleModels returns a merged model list from all enabled backends.
func (r *Router) handleModels(w http.ResponseWriter, req *http.Request) {
	var models []modelObj
	client := &http.Client{Timeout: 5 * time.Second}

	for _, b := range r.cfg.EnabledBackends() {
		resp, err := client.Get(b.BaseURL + "/v1/models")
		if err != nil {
			r.log.Warn("failed to fetch models from backend",
				zap.String("backend", b.Name), zap.Error(err))
			models = append(models, r.staticModels(b)...)
			continue
		}

		body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
		resp.Body.Close()
		if err != nil || resp.StatusCode != http.StatusOK {
			models = append(models, r.staticModels(b)...)
			continue
		}

		var upstreamResp modelsResp
		if err := json.Unmarshal(body, &upstreamResp); err != nil {
			models = append(models, r.staticModels(b)...)
			continue
		}

		valid := r.validateModels(b, upstreamResp.Data)
		if valid == nil {
			models = append(models, r.staticModels(b)...)
			continue
		}
		models = append(models, valid...)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(modelsResp{Object: "list", Data: models})
}

// validateModels filters out entries with an unusable "id".
func (r *Router) validateModels(b config.Backend, data []modelObj) []modelObj {
	var out []modelObj
	for _, m := range data {
		id := strings.TrimSpace(m.ID)
		if id == "" || strings.HasPrefix(id, "/") || strings.HasSuffix(id, "/") {
			r.log.Warn("dropping invalid model entry from backend /v1/models",
				zap.String("backend", b.Name), zap.String("id", m.ID))
			continue
		}
		if m.Created == 0 {
			m.Created = r.startedAt
		}
		if m.Object == "" {
			m.Object = "model"
		}
		m.ID = id
		out = append(out, m)
	}
	return out
}

// staticModels builds modelObj entries from the configured static model list.
func (r *Router) staticModels(b config.Backend) []modelObj {
	out := make([]modelObj, 0, len(b.Models))
	for _, m := range b.Models {
		out = append(out, modelObj{
			ID:      m,
			Object:  "model",
			Created: r.startedAt,
			OwnedBy: "inference-stack",
		})
	}
	return out
}

// statusWriter wraps ResponseWriter to capture the status code.
type statusWriter struct {
	http.ResponseWriter
	status int
}

func (sw *statusWriter) WriteHeader(code int) {
	sw.status = code
	sw.ResponseWriter.WriteHeader(code)
}

func (sw *statusWriter) Write(b []byte) (int, error) {
	if sw.status == 0 {
		sw.status = http.StatusOK
	}
	return sw.ResponseWriter.Write(b)
}

func classifyError(err error) string {
	if err == nil {
		return "none"
	}
	s := err.Error()
	switch {
	case strings.Contains(s, "context deadline exceeded") || strings.Contains(s, "timeout"):
		return "timeout"
	case strings.Contains(s, "connection refused"):
		return "connection_refused"
	case strings.Contains(s, "no such host"):
		return "dns_error"
	case strings.Contains(s, "context canceled"):
		return "canceled"
	default:
		return "unknown"
	}
}
