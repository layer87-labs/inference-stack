package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Backend represents a single upstream model service.
type Backend struct {
	Name    string
	BaseURL string
	Models  []string // models this backend serves; used for /v1/models aggregation
	Enabled bool
	Timeout time.Duration
}

// Config holds all runtime configuration for the inference router.
type Config struct {
	// Server
	ListenAddr     string
	MetricsAddr    string
	ReadTimeout    time.Duration
	WriteTimeout   time.Duration
	IdleTimeout    time.Duration
	MaxRequestSize int64

	// Backends
	Embedding Backend
	Reranker  Backend
	Whisper   Backend

	// Observability
	LogLevel  string
	LogFormat string // "json" | "console"

	// Optional extra backends (future extension via env)
	ExtraBackends []Backend
}

// Load reads configuration from environment variables with sane defaults.
func Load() (*Config, error) {
	cfg := &Config{
		ListenAddr:     env("ROUTER_ADDR", ":8080"),
		MetricsAddr:    env("ROUTER_METRICS_ADDR", ":9090"),
		ReadTimeout:    envDuration("ROUTER_READ_TIMEOUT", 120*time.Second),
		WriteTimeout:   envDuration("ROUTER_WRITE_TIMEOUT", 300*time.Second),
		IdleTimeout:    envDuration("ROUTER_IDLE_TIMEOUT", 60*time.Second),
		MaxRequestSize: envInt64("ROUTER_MAX_REQUEST_SIZE", 100*1024*1024), // 100 MB
		LogLevel:       env("LOG_LEVEL", "info"),
		LogFormat:      env("LOG_FORMAT", "json"),

		Embedding: Backend{
			Name:    "embedding",
			BaseURL: env("EMBEDDING_URL", ""),
			Models:  []string{"BAAI/bge-m3"},
			Enabled: envBool("EMBEDDING_ENABLED", false),
			Timeout: envDuration("EMBEDDING_TIMEOUT", 60*time.Second),
		},
		Reranker: Backend{
			Name:    "reranker",
			BaseURL: env("RERANKER_URL", ""),
			Models:  []string{"BAAI/bge-reranker-v2-m3"},
			Enabled: envBool("RERANKER_ENABLED", false),
			Timeout: envDuration("RERANKER_TIMEOUT", 30*time.Second),
		},
		Whisper: Backend{
			Name:    "whisper",
			BaseURL: env("WHISPER_URL", ""),
			Models:  []string{"whisper-large-v3-turbo", "whisper-large-v3", "whisper-medium", "whisper-small"},
			Enabled: envBool("WHISPER_ENABLED", false),
			Timeout: envDuration("WHISPER_TIMEOUT", 300*time.Second),
		},
	}

	if err := cfg.validate(); err != nil {
		return nil, err
	}

	return cfg, nil
}

func (c *Config) validate() error {
	if c.Embedding.Enabled && c.Embedding.BaseURL == "" {
		return fmt.Errorf("EMBEDDING_ENABLED=true but EMBEDDING_URL is not set")
	}
	if c.Reranker.Enabled && c.Reranker.BaseURL == "" {
		return fmt.Errorf("RERANKER_ENABLED=true but RERANKER_URL is not set")
	}
	if c.Whisper.Enabled && c.Whisper.BaseURL == "" {
		return fmt.Errorf("WHISPER_ENABLED=true but WHISPER_URL is not set")
	}

	enabled := 0
	for _, b := range c.ActiveBackends() {
		if b.Enabled {
			enabled++
		}
	}
	if enabled == 0 {
		return fmt.Errorf("no backends enabled: set at least one of EMBEDDING_ENABLED, RERANKER_ENABLED, WHISPER_ENABLED")
	}

	return nil
}

// ActiveBackends returns all configured backends (enabled or not).
func (c *Config) ActiveBackends() []Backend {
	backends := []Backend{c.Embedding, c.Reranker, c.Whisper}
	backends = append(backends, c.ExtraBackends...)
	return backends
}

// EnabledBackends returns only backends that are enabled.
func (c *Config) EnabledBackends() []Backend {
	var out []Backend
	for _, b := range c.ActiveBackends() {
		if b.Enabled {
			out = append(out, b)
		}
	}
	return out
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envBool(key string, def bool) bool {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return def
	}
	return b
}

func envInt64(key string, def int64) int64 {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return def
	}
	return n
}

func envDuration(key string, def time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return def
	}
	return d
}
