package health

import (
	"context"
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/layer87-labs/inference-stack/internal/config"
	"github.com/layer87-labs/inference-stack/internal/metrics"
)

type Status struct {
	Status   string            `json:"status"`
	Backends map[string]string `json:"backends"`
}

type Checker struct {
	backends []config.Backend
	log      *zap.Logger
	mu       sync.RWMutex
	states   map[string]bool
}

func New(backends []config.Backend, log *zap.Logger) *Checker {
	c := &Checker{
		backends: backends,
		log:      log,
		states:   make(map[string]bool),
	}
	for _, b := range backends {
		c.states[b.Name] = false
		metrics.BackendUp.WithLabelValues(b.Name).Set(0)
	}
	return c
}

// Start runs background health checks every interval.
func (c *Checker) Start(ctx context.Context, interval time.Duration) {
	c.check()
	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				c.check()
			}
		}
	}()
}

func (c *Checker) check() {
	client := &http.Client{Timeout: 5 * time.Second}
	for _, b := range c.backends {
		if !b.Enabled {
			continue
		}
		resp, err := client.Get(b.BaseURL + "/health")
		healthy := err == nil && resp != nil && resp.StatusCode < 400
		if resp != nil {
			resp.Body.Close()
		}

		c.mu.Lock()
		c.states[b.Name] = healthy
		c.mu.Unlock()

		val := 0.0
		if healthy {
			val = 1.0
		}
		metrics.BackendUp.WithLabelValues(b.Name).Set(val)

		if !healthy {
			c.log.Warn("backend health check failed",
				zap.String("backend", b.Name),
				zap.String("url", b.BaseURL),
				zap.Error(err),
			)
		}
	}
}

// IsHealthy returns true if the named backend is up.
func (c *Checker) IsHealthy(name string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.states[name]
}

// ServeHTTP handles /healthz and /readyz.
func (c *Checker) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	c.mu.RLock()
	states := make(map[string]bool, len(c.states))
	for k, v := range c.states {
		states[k] = v
	}
	c.mu.RUnlock()

	s := Status{
		Status:   "ok",
		Backends: make(map[string]string),
	}

	allOK := true
	for _, b := range c.backends {
		if !b.Enabled {
			s.Backends[b.Name] = "disabled"
			continue
		}
		if states[b.Name] {
			s.Backends[b.Name] = "ok"
		} else {
			s.Backends[b.Name] = "unhealthy"
			allOK = false
		}
	}

	if !allOK {
		s.Status = "degraded"
		// degraded but not fatal — return 200 so Kubernetes keeps routing
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s)
}
