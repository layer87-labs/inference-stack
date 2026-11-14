package config

import (
	"reflect"
	"strings"
	"testing"
)

func setEnabled(t *testing.T) {
	t.Helper()
	t.Setenv("WHISPER_ENABLED", "true")
	t.Setenv("WHISPER_URL", "http://whisper:8000")
	t.Setenv("PARAKEET_ENABLED", "true")
	t.Setenv("PARAKEET_URL", "http://parakeet:8000")
}

func TestLoad_CapabilityDefaults(t *testing.T) {
	setEnabled(t)
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	w, p := cfg.Whisper.Capabilities.Audio, cfg.Parakeet.Capabilities.Audio
	if !w.Prompt || !w.Hotwords || !w.WordTimes || !w.ForceLang || w.MaxBytes != 100<<20 {
		t.Errorf("whisper: %+v", w)
	}
	// The second backend differs: no prompt, no hotwords, a short audio limit.
	if p.Prompt || p.Hotwords || !p.WordTimes || p.ForceLang || p.MaxSeconds != 120 || p.MaxBytes != 25<<20 {
		t.Errorf("parakeet: %+v", p)
	}
	if !reflect.DeepEqual(cfg.Embedding.Capabilities.Tasks, []string{"embed"}) || !reflect.DeepEqual(cfg.Reranker.Capabilities.Tasks, []string{"rerank"}) {
		t.Errorf("tasks: %+v %+v", cfg.Embedding.Capabilities, cfg.Reranker.Capabilities)
	}
}

func TestLoad_CapabilityOverride(t *testing.T) {
	setEnabled(t)
	t.Setenv("WHISPER_CAPABILITIES", `{"tasks":["transcribe"],"languages":["de","en"],"audio":{"prompt":false,"hotwords":true,"max_seconds":300}}`)
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	c := cfg.Whisper.Capabilities
	if !reflect.DeepEqual(c.Languages, []string{"de", "en"}) || c.Audio.Prompt || !c.Audio.Hotwords || c.Audio.MaxSeconds != 300 {
		t.Errorf("%+v %+v", c, c.Audio)
	}
}

func TestLoad_BadCapabilitiesFail(t *testing.T) {
	setEnabled(t)
	for name, v := range map[string]string{"not json": `{"audio":`, "negative": `{"audio":{"max_bytes":-1}}`} {
		t.Setenv("PARAKEET_CAPABILITIES", v)
		if _, err := Load(); err == nil || !strings.Contains(err.Error(), "PARAKEET_CAPABILITIES") {
			t.Errorf("%s: got %v", name, err)
		}
	}
}
