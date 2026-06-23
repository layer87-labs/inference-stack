package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	RequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "inference",
		Subsystem: "router",
		Name:      "requests_total",
		Help:      "Total number of requests routed, labeled by backend, path and status code.",
	}, []string{"backend", "path", "status"})

	RequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "inference",
		Subsystem: "router",
		Name:      "request_duration_seconds",
		Help:      "End-to-end request duration in seconds.",
		Buckets:   []float64{.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120, 300},
	}, []string{"backend", "path"})

	RequestSizeBytes = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "inference",
		Subsystem: "router",
		Name:      "request_size_bytes",
		Help:      "Size of incoming requests in bytes.",
		Buckets:   prometheus.ExponentialBuckets(1024, 4, 10), // 1KB → 1GB
	}, []string{"backend"})

	ResponseSizeBytes = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "inference",
		Subsystem: "router",
		Name:      "response_size_bytes",
		Help:      "Size of upstream responses in bytes.",
		Buckets:   prometheus.ExponentialBuckets(1024, 4, 10),
	}, []string{"backend"})

	UpstreamErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "inference",
		Subsystem: "router",
		Name:      "upstream_errors_total",
		Help:      "Total number of upstream errors (connection failures, timeouts).",
	}, []string{"backend", "error_type"})

	ActiveRequests = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "inference",
		Subsystem: "router",
		Name:      "active_requests",
		Help:      "Number of requests currently being proxied.",
	}, []string{"backend"})

	BackendUp = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "inference",
		Subsystem: "router",
		Name:      "backend_up",
		Help:      "1 if the backend is healthy, 0 otherwise.",
	}, []string{"backend"})
)
