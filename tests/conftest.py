import os
import socket
import sys

import asyncio

import pytest

# One event loop for the whole session: grpc.aio's completion-queue poller is bound to the first loop it sees,
# so creating/closing a loop per test (asyncio.run) can strand it.
LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)


def run(coro):
    return LOOP.run_until_complete(coro)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

PROMPTS = [
    "The capital of France is",
    "In 1492, Columbus",
    "def fibonacci(n):",
    "Machine learning is a field of",
    "The quick brown fox",
    "Once upon a time, in a land far away,",
]


def free_ports(n=2):
    socks, ports = [], []
    for _ in range(n):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        socks.append(s)
        ports.append(s.getsockname()[1])
    for s in socks:
        s.close()
    return ports


@pytest.fixture(scope="session")
def gpt2():
    from pllm.model import load_hf
    return load_hf("gpt2")


@pytest.fixture(scope="session")
def references(gpt2):
    """HF `generate` greedy outputs: the ground truth every distributed run must reproduce."""
    tok, hf = gpt2
    out = {}
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        g = hf.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
        out[p] = g[0, ids.shape[1]:].tolist()
    return out
