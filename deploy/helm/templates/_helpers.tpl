{{/*
Expand chart name.
*/}}
{{- define "inference-stack.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Fully qualified app name.
*/}}
{{- define "inference-stack.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "inference-stack.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 }}
app.kubernetes.io/name: {{ include "inference-stack.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels for a given component.
Usage: include "inference-stack.selectorLabels" (dict "Release" .Release "Chart" .Chart "Values" .Values "component" "router")
*/}}
{{- define "inference-stack.selectorLabels" -}}
app.kubernetes.io/name: {{ include "inference-stack.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/*
Resolve image — prepends global.imageRegistry if set.
Usage: include "inference-stack.image" (dict "registry" .Values.global.imageRegistry "repo" .Values.router.image.repository "tag" .Values.router.image.tag "appVersion" .Chart.AppVersion)
*/}}
{{- define "inference-stack.image" -}}
{{- $tag := .tag | default .appVersion -}}
{{- if .registry -}}
{{- printf "%s/%s:%s" (trimSuffix "/" .registry) .repo $tag -}}
{{- else -}}
{{- printf "%s:%s" .repo $tag -}}
{{- end -}}
{{- end }}

{{/*
Image pull secrets — merges component-level with global, deduplicates.
Usage: include "inference-stack.imagePullSecrets" (dict "local" .Values.router.imagePullSecrets "global" .Values.global.imagePullSecrets)
*/}}
{{- define "inference-stack.imagePullSecrets" -}}
{{- $merged := concat (.local | default list) (.global | default list) | uniq -}}
{{- if $merged -}}
imagePullSecrets:
{{- range $merged }}
  - name: {{ .name }}
{{- end }}
{{- end -}}
{{- end }}

{{/*
Storage class — falls back to global.storageClass.
*/}}
{{- define "inference-stack.storageClass" -}}
{{- $sc := .local | default .global -}}
{{- if $sc -}}{{ $sc }}{{- end -}}
{{- end }}
