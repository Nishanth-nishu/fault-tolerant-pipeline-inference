#!/usr/bin/env bash
# Two-node deployment (e.g. RCE). Run the first line on node B, the second on node A, then the client anywhere.
#   node B:  python -m pllm.worker --stage 1 --model gpt2 --host 0.0.0.0 --port 50052
#   node A:  python -m pllm.worker --stage 0 --model gpt2 --host 0.0.0.0 --port 50051 --downstream NODE_B:50052 --log-activations
#   client:  python -m pllm.cli generate --s0 NODE_A:50051 --s1 NODE_B:50052 --prompt "Distributed systems are"
# Worker epochs default to the start time in ms, so a restarted worker is always a new epoch; the coordinator
# detects it via Health/STALE_EPOCH and replays affected requests (restart is up to your process manager).
echo "see comments in $0"
