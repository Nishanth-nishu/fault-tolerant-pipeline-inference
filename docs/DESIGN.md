# Design notes

## Topology

```
            gRPC Forward(token ids)            gRPC Forward(hidden states, codec)
client ──► Coordinator ───────────────────► Stage 0 worker ─────────────────────────► Stage 1 worker
             ▲  admission queue (bounded)     embeddings + layers [0,k)                 layers [k,n) + final norm + LM head
             │  iteration-level scheduler     per-request KV cache                       per-request KV cache
             │  token selection / commits     optional boundary-activation log           returns top-k / full logits row
             └──────────────── ForwardReply (scores, per-stage stats, live epochs) ◄──────┘
Supervisor: spawns workers as OS processes, assigns epochs, kills (fault injection), restarts / promotes hot spares
```

Stage 0 awaits the downstream RPC *outside* its single compute thread, so while stage 1 computes micro-batch
*k*, stage 0 already computes micro-batch *k+1*. The coordinator keeps `max_inflight_batches` (default 2 = pipeline
depth) micro-batches outstanding; each request has at most one operation in flight, which preserves per-request
order while different requests occupy different stages.

## Unified operation model

For a request, `history = prompt + committed tokens` and `pos` = number of history tokens whose KV is present on
both stages. Every operation feeds `history[pos:pos+n]` at `cache_position=pos`; when it consumes the last history
token it asks for scores and exactly one token is committed. Consequences:

* prefill, chunked prefill and decode are the same code path;
* micro-batches can mix prefill chunks and decodes (Sarathi-Serve-style *stall-free* batches: decodes first,
  prefill chunks fill the remaining token budget);
* recovery is "move `pos` back and re-feed history with outputs suppressed".

## Selective batching

A micro-batch is flattened to `[T, H]`; QKV, output projection and MLP run once over all T tokens (one GEMM
each), and only attention is evaluated per sequence against that sequence's KV cache (Orca's selective
batching). No padding is ever materialised.

## Pipeline depth, micro-batching and the adaptive policy

`max_inflight_batches` (depth) bounds the micro-batches outstanding in the pipeline. With depth *d* > 1 the
scheduler caps a micro-batch at ⌈runnable / d⌉ sequences, so stage 0 can work on one micro-batch while stage 1
works on another (GPipe-style). Whether that helps depends on the hardware. If the stages share a memory bus
(one host), smaller micro-batches re-stream the weights more often and can lose to depth 1. If the link is
slow, overlap hides its latency. `depth_policy="auto"` A/B-tests depth 1 against depth *d* online:

* metric = sequence-steps per second (insensitive to the prefill/decode mix);
* arms alternate A, B, A, B over `auto_window_s` windows (`auto_samples` clean windows per arm);
* a window in which new requests were admitted is discarded (its prefill burst would bias the active arm);
* the winner is exploited for `auto_exploit_windows`, then re-measured (tracks load and link changes).

## Cost-balanced partitioning

`python -m pllm.balance <model>` times each block in isolation for a decode micro-batch (the last block's time
includes the final norm and LM head) and picks k = argmin max(Σ_{i<k} t_i, Σ_{i≥k} t_i). For GPT-2 the head
costs ≈ 4–5 blocks, so the balanced cut is layer 8, not 6.

## Ordering, idempotence, staleness

| Mechanism | Guarantees |
|---|---|
| `op_seq` per request, one op in flight | per-request order; reply must echo the outstanding `op_seq` |
| position-indexed KV writes, cache length truncated to `pos+n` | re-executing an op (lost reply, retry) is idempotent |
| `CacheMiss` when `cache_position > cache length` | a restarted worker can never attend to missing state |
| worker epochs (`expected_epochs` in every request; live epochs in every reply) | workers reject requests addressed to an older incarnation (`STALE_EPOCH`); the coordinator discards replies produced under an epoch view that changed while the batch was in flight |
| `max_retries` per request | after bounded retries the request **fails explicitly** |

## Flow control

* coordinator admission queue `max_waiting` → `Overloaded` raised to the caller (load shedding);
* `max_active` bounds sequences holding KV state (memory);
* each worker has a bounded inbound queue `--max-queue`; when full it replies `OVERLOADED` immediately and the
  coordinator backs off and re-queues (explicit backpressure, never unbounded buffering);
* every RPC has a deadline; stage 0 propagates a (smaller) deadline to its downstream hop.

## Failure handling and recovery

On an RPC error, timeout, `DOWNSTREAM_FAILED`, `STALE_EPOCH` or `CACHE_MISS` the coordinator probes both workers.
Dead workers are restarted by the supervisor with a new epoch; every active request whose KV lived on a changed
stage is marked for recovery. Three recovery paths:

| path | when | work |
|---|---|---|
| `full` (scope baseline) | always available | drop request state on both stages, replay `history[0:pos]` through the complete pipeline, outputs suppressed |
| `s1-from-log` (*targeted*) | only stage 1 lost, stage 0 logs boundary activations | stage 0 re-sends its logged activations; no stage-0 recompute (cf. Petals client-side input caching) |
| `s0-local` (*targeted*) | only stage 0 lost | replay `history[0:pos]` through stage 0 only (`forward_downstream=false`); stage 1 keeps its KV |

**Hot spares.** With `hot_spare=True` the supervisor keeps one pre-loaded standby process per stage (weights
resident, port not yet bound). Promotion = write `GO <epoch>` to its stdin; it binds the port and serves. This
removes interpreter start-up and weight loading from the recovery critical path (DéjàVu / FailSafe motivate
keeping recovery off the model-load path).

## Activation codecs (inter-stage link)

| codec | bytes / token (H=768) | notes |
|---|---|---|
| fp32 | 3072 | lossless reference |
| fp16 / bf16 | 1536 | fp16 has an overflow guard (GPT-2 residual outliers reach ~3000 but stay < 65504) |
| int8 | 770 | per-token absmax — hurt by outlier channels |
| int8h | 792 | 64-wide block Hadamard rotation + per-group int8 |
| int4h | 408 | Hadamard + per-group int4, nibble-packed (TAH-Quant-inspired) |
