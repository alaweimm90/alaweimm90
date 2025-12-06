{{/*
Expand the name of the chart.
*/}}
{{- define "mezan-atlas.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "mezan-atlas.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "mezan-atlas.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mezan-atlas.labels" -}}
helm.sh/chart: {{ include "mezan-atlas.chart" . }}
{{ include "mezan-atlas.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: mezan
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mezan-atlas.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mezan-atlas.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "mezan-atlas.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "mezan-atlas.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "mezan-atlas.image" -}}
{{- with .Values.global.imageRegistry }}
{{- printf "%s/%s:%s" . $.Values.image.repository ($.Values.image.tag | default $.Chart.AppVersion) }}
{{- else }}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) }}
{{- end }}
{{- end }}

{{/*
Return the proper Docker Image Registry Secret Names
*/}}
{{- define "mezan-atlas.imagePullSecrets" -}}
{{- $pullSecrets := list }}
{{- if .Values.global.imagePullSecrets }}
{{- range .Values.global.imagePullSecrets }}
{{- $pullSecrets = append $pullSecrets . }}
{{- end }}
{{- end }}
{{- if .Values.imagePullSecrets }}
{{- range .Values.imagePullSecrets }}
{{- $pullSecrets = append $pullSecrets . }}
{{- end }}
{{- end }}
{{- if $pullSecrets }}
imagePullSecrets:
{{- range $pullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Return the secret name
*/}}
{{- define "mezan-atlas.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- include "mezan-atlas.fullname" . }}-secrets
{{- end }}
{{- end }}

{{/*
Return Redis host
*/}}
{{- define "mezan-atlas.redisHost" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" (include "mezan-atlas.fullname" .) }}
{{- else }}
{{- .Values.externalRedis.host }}
{{- end }}
{{- end }}

{{/*
Return PostgreSQL host
*/}}
{{- define "mezan-atlas.postgresqlHost" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "mezan-atlas.fullname" .) }}
{{- else }}
{{- .Values.externalPostgresql.host }}
{{- end }}
{{- end }}

{{/*
Check if any ingress is enabled
*/}}
{{- define "mezan-atlas.ingress.enabled" -}}
{{- if or .Values.ingress.enabled .Values.monitoring.grafana.ingress.enabled }}
true
{{- else }}
false
{{- end }}
{{- end }}