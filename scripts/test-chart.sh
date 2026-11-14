#!/usr/bin/env bash
# Render the Helm chart and check the optional Parakeet backend:
# off by default (nothing rendered, router told it is disabled), and with
# parakeet.enabled=true a Deployment, Service, NetworkPolicy and the router's
# PARAKEET_URL. Needs helm.
set -euo pipefail
chart="$(dirname "$0")/../deploy/helm"
fail() { echo "FAIL: $*" >&2; exit 1; }

off=$(helm template t "$chart")
grep -q 'name: t-inference-stack-parakeet' <<<"$off" && fail "parakeet rendered although disabled"
grep -A1 'name: PARAKEET_ENABLED' <<<"$off" | grep -q 'value: "false"' || fail "router lacks PARAKEET_ENABLED=false"

on=$(helm template t "$chart" --set parakeet.enabled=true)
for kind in Deployment Service NetworkPolicy; do
  grep -B3 'name: t-inference-stack-parakeet$' <<<"$on" | grep -q "kind: $kind" || fail "no parakeet $kind"
done
grep -q 'value: "http://t-inference-stack-parakeet:9000"' <<<"$on" || fail "router lacks PARAKEET_URL"
grep -q 'app.kubernetes.io/component: router' <<<"$(grep -A30 'kind: NetworkPolicy' <<<"$on")" || fail "networkpolicy does not select the router"

nonp=$(helm template t "$chart" --set parakeet.enabled=true --set parakeet.networkPolicy.enabled=false)
grep -q 'kind: NetworkPolicy' <<<"$nonp" && fail "networkpolicy rendered although disabled"
echo "chart OK"
