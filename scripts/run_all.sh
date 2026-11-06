#!/usr/bin/env bash
# Reproduce every number in docs/REPORT.md. Roughly 4-5 h on a 6P+8E-core desktop CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -v | tee results/test_report.txt                       # 31 tests (~4 min)
python -m pllm.balance gpt2 | tee results/balance_gpt2.txt              # cost-balanced split (-> 8)
python -m pllm.balance Qwen/Qwen2.5-0.5B | tee results/balance_qwen.txt # (-> 15)
python -m bench.contention                                              # DRAM-bandwidth contention check
python -m bench.accuracy --models gpt2 --out results/accuracy
python -m bench.accuracy --models Qwen/Qwen2.5-0.5B --wikitext-tokens 32768 --lambada-n 500 --gen-n 32 \
       --sweep-tokens 4096 --out results/accuracy
python -m bench.perf --model gpt2 --split 8 --repeats 4 --exp concurrency network load --out results/perf
python -m bench.perf --model gpt2 --split 8 --repeats 4 --exp network concurrency --links 0,0 2,0 5,20 \
       --net-systems pipe-d1-fp32 pipe-d2-fp32 pipe-auto-fp32 --systems baseline pipe-d1 pipe-d2 pipe-auto \
       --out results/perf_auto
python -m bench.perf --model Qwen/Qwen2.5-0.5B --split 15 --repeats 3 --per-client 1 --exp concurrency network \
       --concurrency 1 4 8 --prompt-lens 64 --systems baseline pipe-d1 pipe-d2 pipe-auto --links 0,0 5,20 \
       --net-systems pipe-d1-fp32 pipe-d2-fp32 pipe-d2-int8h pipe-auto-fp32 --out results/perf
python -m bench.faults --model gpt2 --trials 5 --out results/faults
python -m bench.memory --models gpt2 Qwen/Qwen2.5-0.5B --out results/memory.csv
python -m bench.plots --results results --out docs/figures
