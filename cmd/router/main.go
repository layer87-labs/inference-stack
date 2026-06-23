package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/layer87-labs/inference-stack/internal/config"
	"github.com/layer87-labs/inference-stack/internal/health"
	"github.com/layer87-labs/inference-stack/internal/proxy"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		_, _ = os.Stderr.WriteString("config error: " + err.Error() + "\n")
		os.Exit(1)
	}

	log := buildLogger(cfg.LogLevel, cfg.LogFormat)
	defer log.Sync() //nolint:errcheck

	log.Info("inference-router starting",
		zap.String("addr", cfg.ListenAddr),
		zap.String("metrics_addr", cfg.MetricsAddr),
		zap.Bool("embedding_enabled", cfg.Embedding.Enabled),
		zap.Bool("reranker_enabled", cfg.Reranker.Enabled),
		zap.Bool("whisper_enabled", cfg.Whisper.Enabled),
	)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// Health checker — runs background probes, exposes BackendUp metrics
	checker := health.New(cfg.EnabledBackends(), log)
	checker.Start(ctx, 15*time.Second)

	// Main proxy router
	router, err := proxy.New(cfg, checker, log)
	if err != nil {
		log.Fatal("failed to build router", zap.Error(err))
	}

	mux := http.NewServeMux()
	mux.Handle("/v1/", router)
	mux.Handle("/rerank", router)
	mux.Handle("/embed", router)
	mux.Handle("/embed_sparse", router)
	mux.Handle("/healthz", checker)
	mux.Handle("/readyz", checker)

	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"service":"inference-router","status":"ok"}`))
			return
		}
		http.NotFound(w, r)
	})

	mainServer := &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      mux,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  cfg.IdleTimeout,
	}

	// Metrics server — separate port, not exposed externally
	metricsMux := http.NewServeMux()
	metricsMux.Handle("/metrics", promhttp.Handler())
	metricsMux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("ok"))
	})
	metricsServer := &http.Server{
		Addr:        cfg.MetricsAddr,
		Handler:     metricsMux,
		ReadTimeout: 5 * time.Second,
	}

	// Start both servers
	go func() {
		log.Info("metrics server listening", zap.String("addr", cfg.MetricsAddr))
		if err := metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("metrics server error", zap.Error(err))
		}
	}()

	go func() {
		log.Info("main server listening", zap.String("addr", cfg.ListenAddr))
		if err := mainServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal("main server error", zap.Error(err))
		}
	}()

	<-ctx.Done()
	log.Info("shutdown signal received")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := mainServer.Shutdown(shutdownCtx); err != nil {
		log.Error("main server shutdown error", zap.Error(err))
	}
	if err := metricsServer.Shutdown(shutdownCtx); err != nil {
		log.Error("metrics server shutdown error", zap.Error(err))
	}

	log.Info("inference-router stopped")
}

func buildLogger(level, format string) *zap.Logger {
	var lvl zapcore.Level
	if err := lvl.UnmarshalText([]byte(level)); err != nil {
		lvl = zapcore.InfoLevel
	}

	var cfg zap.Config
	if format == "console" {
		cfg = zap.NewDevelopmentConfig()
	} else {
		cfg = zap.NewProductionConfig()
	}
	cfg.Level = zap.NewAtomicLevelAt(lvl)

	log, err := cfg.Build()
	if err != nil {
		panic("failed to build logger: " + err.Error())
	}
	return log
}
