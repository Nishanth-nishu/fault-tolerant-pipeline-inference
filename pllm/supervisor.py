"""Process supervisor: launches the two stage workers as independent OS processes, assigns monotonically
increasing *worker epochs*, kills them (failure injection) and restarts them with a fresh epoch."""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

import grpc

from .proto import pipeline_pb2 as pb
from .proto import pipeline_pb2_grpc as rpc
from .worker import GRPC_OPTS


@dataclass
class WorkerSpec:
    stage: int
    port: int
    host: str = "127.0.0.1"
    extra_args: list[str] = field(default_factory=list)

    @property
    def addr(self):
        return f"{self.host}:{self.port}"


class Supervisor:
    def __init__(self, model: str, split: int | None, ports=(50051, 50052), threads: int | None = None,
                 log_dir: str | None = None, s0_args: list[str] | None = None, s1_args: list[str] | None = None,
                 hot_spare: bool = False, head_on_first: bool = False):
        self.model, self.split = model, split
        self.specs = {0: WorkerSpec(0, ports[0], extra_args=s0_args or []),
                      1: WorkerSpec(1, ports[1], extra_args=s1_args or [])}
        self.threads = threads
        self.procs: dict[int, subprocess.Popen] = {}
        self.epochs = {0: 0, 1: 0}
        self.log_dir = log_dir
        self.restarts = {0: 0, 1: 0}
        self.hot_spare = hot_spare
        self.head_on_first = head_on_first
        self.spares: dict[int, subprocess.Popen] = {}
        self.spare_logs: dict[int, str] = {}
        self.restart_ms: list[dict] = []

    def addr(self, stage: int) -> str:
        return self.specs[stage].addr

    def _cmd(self, stage: int) -> list[str]:
        sp = self.specs[stage]
        cmd = [sys.executable, "-m", "pllm.worker", "--stage", str(stage), "--model", self.model,
               "--port", str(sp.port), "--epoch", str(self.epochs[stage]), "--parent-pid", str(os.getpid())]
        if self.split is not None:
            cmd += ["--split", str(self.split)]
        if self.threads:
            cmd += ["--threads", str(self.threads)]
        if stage == 0:
            cmd += ["--downstream", self.specs[1].addr]
        if self.head_on_first:
            cmd += ["--head-on-first"]
        return cmd + sp.extra_args

    def _popen(self, stage: int, standby: bool):
        tag = "spare" if standby else f"e{self.epochs[stage]}"
        import tempfile
        log_dir = self.log_dir or os.path.join(tempfile.gettempdir(), "pllm-workers")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"s{stage}-{tag}-{os.getpid()}-{time.time_ns()}.log")
        out = open(path, "w")
        if standby:
            self.spare_logs[stage] = path
        env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1", HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"),
                   TRANSFORMERS_VERBOSITY="error")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = self._cmd(stage) + (["--standby"] if standby else [])
        return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=root,
                                stdin=subprocess.PIPE if standby else subprocess.DEVNULL)

    def spawn(self, stage: int):
        self.epochs[stage] += 1
        self.procs[stage] = self._popen(stage, standby=False)

    def spawn_spare(self, stage: int):
        if self.hot_spare and (stage not in self.spares or self.spares[stage].poll() is not None):
            self.spares[stage] = self._popen(stage, standby=True)

    async def wait_ready(self, stage: int, timeout: float = 120.0) -> pb.HealthReply:
        deadline = time.monotonic() + timeout
        async with grpc.aio.insecure_channel(self.addr(stage), options=GRPC_OPTS) as ch:
            stub = rpc.StageServiceStub(ch)
            while time.monotonic() < deadline:
                p = self.procs.get(stage)
                if p is not None and p.poll() is not None:
                    raise RuntimeError(f"worker s{stage} exited with code {p.returncode} during startup")
                try:
                    h = await stub.Health(pb.HealthRequest(), timeout=1.0)
                    if h.epoch == self.epochs[stage]:
                        return h
                except grpc.aio.AioRpcError:
                    pass
                await asyncio.sleep(0.1)
        raise TimeoutError(f"worker s{stage} not ready after {timeout}s")

    async def start(self):
        for s in (1, 0):
            self.spawn(s)
        await asyncio.gather(self.wait_ready(0), self.wait_ready(1))
        for s in (0, 1):
            self.spawn_spare(s)

    def alive(self, stage: int) -> bool:
        p = self.procs.get(stage)
        return p is not None and p.poll() is None

    def kill(self, stage: int):
        """Hard failure injection (SIGKILL / TerminateProcess): no graceful shutdown, caches are lost."""
        p = self.procs.get(stage)
        if p and p.poll() is None:
            if os.name == "nt":
                p.kill()
            else:
                os.kill(p.pid, signal.SIGKILL)
            p.wait(timeout=10)

    def spare_ready_snapshot(self, stage, proc) -> bool:
        try:
            with open(self.spare_logs.get(stage, "")) as f:
                return "STANDBY" in f.read()
        except OSError:
            return False

    def spare_ready(self, stage: int) -> bool:
        """True when the standby for `stage` has finished loading weights (printed STANDBY)."""
        p = self.spares.get(stage)
        if p is None or p.poll() is not None:
            return False
        try:
            with open(self.spare_logs[stage]) as f:
                return "STANDBY" in f.read()
        except OSError:
            return False

    async def wait_spares(self, timeout: float = 120.0):
        deadline = time.monotonic() + timeout
        while self.hot_spare and not all(self.spare_ready(s) for s in (0, 1)):
            if time.monotonic() > deadline:
                raise TimeoutError("hot spares not ready")
            await asyncio.sleep(0.2)

    async def restart(self, stage: int) -> int:
        """Replace a failed worker. With a hot spare the replacement already holds the weights, so recovery
        cost is port re-bind + KV replay instead of process start + model load."""
        t0 = time.perf_counter()
        self.kill(stage)
        self.restarts[stage] += 1
        spare = self.spares.pop(stage, None)
        used_spare = spare is not None and spare.poll() is None
        spare_was_loaded = used_spare and self.spare_ready_snapshot(stage, spare)
        if used_spare:
            self.epochs[stage] += 1
            spare.stdin.write(f"GO {self.epochs[stage]}\n".encode())
            spare.stdin.flush()
            self.procs[stage] = spare
        else:
            self.spawn(stage)
        await self.wait_ready(stage)
        self.restart_ms.append({"stage": stage, "ms": (time.perf_counter() - t0) * 1e3, "hot_spare": used_spare,
                                "spare_loaded": spare_was_loaded})
        self.spawn_spare(stage)
        return self.epochs[stage]

    def stop(self):
        for s in list(self.procs):
            self.kill(s)
        for p in self.spares.values():
            if p.poll() is None:
                p.kill()
        self.procs.clear()
        self.spares.clear()
