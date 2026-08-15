#!/usr/bin/env bash
# Generic remote round wrapper: run an orchestrator, then STOP OUR OWN POD.
#
# EV-0036 repair: completion-means-STOP must not depend on a laptop process
# that may be suspended. After the orchestrator finishes (round-log end
# boundary reached, all arm outputs durable on /workspace), this wrapper
# writes a ROUND_COMPLETE marker and stops the pod via the RunPod GraphQL
# API from the pod itself.
#
# Usage (launch via ssh):
#   RUNPOD_API_KEY=<key> POD_ID=<id> ROUND_MARKER=/workspace/currX.done \
#     nohup bash remote_orchestrator_with_stop.sh <orchestrator.sh> > out 2>&1 &
#
# Safety: stop fires ONLY after the wrapped orchestrator exits; outputs are
# on the persistent volume and survive the stop. A failed arm does not skip
# the stop (idle burn is worse; adjudication handles failures from artefacts).
set -u

ORCH="$1"
POD_ID="${POD_ID:?POD_ID required}"
MARKER="${ROUND_MARKER:-/workspace/round_complete.marker}"

bash "$ORCH"
code=$?

{
  echo "orchestrator=$ORCH exit=$code"
  echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$MARKER"

echo "ROUND_COMPLETE exit=$code; stopping pod $POD_ID"
curl -sS -m 60 -X POST "https://api.runpod.io/graphql" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY:?RUNPOD_API_KEY required}" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }\"}" \
  || echo "SELF_STOP_FAILED - watchdog/resume must stop the pod"
