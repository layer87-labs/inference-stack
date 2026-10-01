#!/usr/bin/env bash
# Sign pushed container images by digest with keyless cosign (GitHub OIDC).
# Usage: scripts/sign-image.sh <repository:tag>...
# Requires: cosign, docker, a prior `docker push`, and `id-token: write`.
set -euo pipefail

for ref in "$@"; do
  repo="${ref%:*}"
  digest="$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "${ref}" \
    | grep "^${repo}@sha256:" | head -n1)"
  if [ -z "${digest}" ]; then
    echo "no pushed digest found for ${ref}" >&2
    exit 1
  fi
  cosign sign --yes "${digest}"
  echo "Signed ${digest}" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
done
