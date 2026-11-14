package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// Capabilities describe what the model(s) behind a backend can do. They are
// published as the "capabilities" field of /v1/models entries, so that a
// client can adapt (send a prompt or not, ask for word timestamps, cut audio
// to the longest accepted length) instead of guessing from the model name.
// The field is additive: clients that do not know it ignore it.
type Capabilities struct {
	Tasks     []string           `json:"tasks,omitempty"`     // "embed", "rerank", "transcribe"
	Languages []string           `json:"languages,omitempty"` // ISO 639-1; empty = not stated
	Audio     *AudioCapabilities `json:"audio,omitempty"`
}

// AudioCapabilities are the abilities of a transcription backend.
type AudioCapabilities struct {
	Prompt       bool     `json:"prompt"`                 // accepts a context prompt (`prompt` form field)
	PromptMax    int      `json:"prompt_max,omitempty"`   // characters; 0 = not stated
	Hotwords     bool     `json:"hotwords"`               // accepts a `hotwords` form field
	HotwordsMax  int      `json:"hotwords_max,omitempty"` // characters; 0 = not stated
	WordTimes    bool     `json:"word_times"`             // word timestamps in verbose_json
	SegmentTimes bool     `json:"segment_times"`          // segment timestamps in verbose_json
	Confidence   bool     `json:"confidence"`             // per-segment or per-word confidence
	AutoLanguage bool     `json:"auto_language"`          // detects the language itself
	ForceLang    bool     `json:"force_language"`         // honours the `language` field
	Diarization  bool     `json:"diarization"`            // separates speakers
	Streaming    bool     `json:"streaming"`              // takes audio while it is spoken
	MaxSeconds   int      `json:"max_seconds,omitempty"`  // longest audio per request; 0 = not stated
	MaxBytes     int      `json:"max_bytes,omitempty"`    // largest upload per request; 0 = not stated
	Formats      []string `json:"formats,omitempty"`      // accepted containers; empty = not stated
}

func (c *Capabilities) validate() error {
	if a := c.Audio; a != nil && (a.PromptMax < 0 || a.HotwordsMax < 0 || a.MaxSeconds < 0 || a.MaxBytes < 0) {
		return fmt.Errorf("negative limit")
	}
	return nil
}

// capabilitiesFromEnv returns the JSON in the environment variable key,
// or def when it is unset. A malformed value is an error: a typo must not
// silently publish the defaults.
func capabilitiesFromEnv(key string, def *Capabilities) (*Capabilities, error) {
	v := os.Getenv(key)
	if v == "" {
		return def, nil
	}
	var c Capabilities
	if err := json.Unmarshal([]byte(v), &c); err != nil {
		return nil, fmt.Errorf("%s: invalid capabilities JSON: %w", key, err)
	}
	if err := c.validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", key, err)
	}
	return &c, nil
}

// Defaults for the bundled backends, matching what the servers in deploy/ do.
func embeddingCapabilities() *Capabilities { return &Capabilities{Tasks: []string{"embed"}} }
func rerankerCapabilities() *Capabilities  { return &Capabilities{Tasks: []string{"rerank"}} }

func whisperCapabilities() *Capabilities {
	return &Capabilities{Tasks: []string{"transcribe"}, Audio: &AudioCapabilities{
		Prompt: true, PromptMax: 2000, Hotwords: true, HotwordsMax: 2000,
		WordTimes: true, SegmentTimes: true, Confidence: true, AutoLanguage: true, ForceLang: true,
		MaxBytes: 100 << 20,
	}}
}

func parakeetCapabilities() *Capabilities {
	return &Capabilities{Tasks: []string{"transcribe"}, Audio: &AudioCapabilities{
		// The prompt field is accepted and ignored: the model has no prompt input.
		WordTimes: true, SegmentTimes: true, Confidence: true, AutoLanguage: true,
		MaxSeconds: 120, MaxBytes: 25 << 20,
	}}
}
