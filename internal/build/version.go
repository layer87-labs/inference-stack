package build

// Build information injected at compile time via ldflags.
var (
	ServiceName  = "inference-stack"
	Version      = "dev"
	CommitHash   = "unknown"
	BuildDate    = "unknown"
	InstanceName = "" // Set at runtime from POD_NAME or HOSTNAME
)

// Info returns build information as a map.
func Info() map[string]string {
	return map[string]string{
		"service_name": ServiceName,
		"version":      Version,
		"commit_hash":  CommitHash,
		"build_date":   BuildDate,
		"instance":     InstanceName,
	}
}
