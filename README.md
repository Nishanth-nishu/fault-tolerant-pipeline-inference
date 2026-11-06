# pllm — fault-tolerant two-stage pipeline inference for decoder-only LLMs

Distributed Systems project, Topic 38, Team 62 (Rajath Gadagkar, Nishanth R, Sai Kiran).

A pretrained causal LM (GPT-2, DistilGPT-2, or Llama-style Qwen2.5) is split at a Transformer-layer boundary
across **two independent worker processes** that talk **gRPC over TCP**. A coordinator handles tokenization,
admission control, iteration-level scheduling, token selection, failure detection and recovery. Each worker
keeps a **per-request KV cache** for the layers it owns.

| | |
|---|---|
| **Correctness** | Every request either reproduces the single-process greedy output **token-for-token**, or fails explicitly. Tokens are never stale, duplicated or out of order, including under SIGKILL of either or both workers, lost replies, overload and timeouts (31 automated tests) |
| **Systems** | bounded queues + explicit backpressure (`OVERLOADED`) and load shedding (`Overloaded`), RPC deadlines with propagation, bounded retries, worker **epochs** and stale-reply rejection, idempotent position-indexed KV writes |
| **Recovery** | full replay (scope baseline), **targeted** replay (stage-1 rebuild from stage-0's logged boundary activations; stage-0-only replay), and **hot-spare** workers that take the weight load off the recovery critical path |
| **Scheduling** | Orca-style *selective batching* (one GEMM per micro-batch, attention per sequence), Sarathi-style chunked prefill with decode-first stall-free batches, pipeline micro-batching, and an **adaptive depth policy** that A/B-tests overlap online |
| **Research extensions** | inter-stage activation compression (fp16/bf16, int8, **Hadamard-rotated int8/int4**), a split-point × codec study, a cost-balanced partitioner, and a measured explanation of when single-host pipelining helps or hurts |

See [docs/REPORT.md](docs/REPORT.md) for the evaluation and [docs/DESIGN.md](docs/DESIGN.md) for the protocol and design.

## Headline results

| what | result |
|---|---|
| Numerical fidelity | WikiText-2 PPL of the split model = HF reference to 4 decimals (GPT-2 37.4033, Qwen2.5-0.5B 16.9895); 64/64 greedy continuations identical |
| Fault tolerance | **60/60** SIGKILL trials (s0, s1, both × full/targeted × cold/hot-spare) and **25/25** lossy-reply trials (≤ 20 % replies lost): every request exact, 0 duplicates, 0 out-of-order, 0 failures |
| Recovery time | detection 45–130 ms; stall after a kill 9–19 s (cold restart) → **0.56–1.4 s with hot spares**; targeted replay ≈ 2× faster for stage-0 loss with 0 link bytes |
| Slow links | 5 ms / 20 Mb link: depth-2 overlap + int4h codec = **1.76×** throughput and **½ TTFT** vs. naive depth-1 fp32 |
| Activation codecs | Hadamard-rotated int8: +0.012 % PPL (GPT-2) and −0.002 % (Qwen) vs. +0.26 % / +1.88 % for plain int8 at the same size |
| Memory capacity | meta-device partial loading + tied-head ring placement: largest Qwen worker needs **0.60×** the single-process RSS |
| Bigger model | Qwen2.5-0.5B pipeline at throughput parity with the single process at low concurrency (1.04× / 0.96× at C=1/4) |
| Honest negative | on one shared-memory host a batched single process is 1.3–1.8× faster than the pipeline; we measured why (45–55 % DRAM-bandwidth contention between stages on disjoint cores) |

## Layout

```
pllm/
  model.py        Stage: layer-range executor (GPT-2 + Llama/Qwen2 families), position-indexed KV caches
  codec.py        activation codecs for the inter-stage link
  proto/          pipeline.proto (+ generated stubs)
  worker.py       gRPC stage server (bounded queue, epochs, downstream forwarding, netem, fault injection, hot standby)
  coordinator.py  Engine: admission, scheduler, commits, failure detection, recovery
  supervisor.py   process launcher: epochs, kill, restart, hot spares
  local.py        single-process baseline transport (same Engine, no partition/RPC)
  balance.py      cost-balanced split-point search
  (model.py also: load_partial = meta-device partial loading; --head-on-first ring placement)
  cli.py          demo / generate entry points
bench/            accuracy.py, perf.py, faults.py, memory.py, contention.py, plots.py
tests/            test_model.py (numerics), test_distributed.py (end-to-end over real processes)
results/          raw CSV / JSON measurements      docs/figures/  plots
```

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python -m pytest -v
```

```bash
python -m pllm.cli demo --model gpt2 --kill s1 --recovery targeted --hot-spare
```

Run on two machines (e.g. two RCE nodes):

```bash
python -m pllm.worker --stage 1 --model gpt2 --host 0.0.0.0 --port 50052
```

```bash
python -m pllm.worker --stage 0 --model gpt2 --host 0.0.0.0 --port 50051 --downstream NODE_B:50052 --log-activations
```

```bash
python -m pllm.cli generate --s0 NODE_A:50051 --s1 NODE_B:50052 --prompt "Distributed systems are"
```

Reproduce all results: `bash scripts/run_all.sh`.
