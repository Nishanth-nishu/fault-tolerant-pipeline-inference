"""Memory-bandwidth contention micro-benchmark: time each stage's decode step alone, then both concurrently on
disjoint pinned cores. Usage: python -m bench.contention [--s0-cpus 0,2,4 --s1-cpus 6,8,10 --split 8]"""
import argparse, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--s0-cpus", default="0,2,4")
ap.add_argument("--s1-cpus", default="6,8,10")
ap.add_argument("--split", type=int, default=8)
ap.add_argument("--layers", type=int, default=12)
ap.add_argument("--seconds", type=float, default=6)
a = ap.parse_args()
w = [sys.executable, "bench/contention_worker.py"]
s0 = [a.s0_cpus, "0", str(a.split)]
s1 = [a.s1_cpus, str(a.split), str(a.layers)]
run = lambda args, start: subprocess.Popen(w + args + [str(a.seconds), str(start)], stdout=subprocess.PIPE, text=True)
for name, procs in [("solo s0", [s0]), ("solo s1", [s1]), ("concurrent", [s0, s1])]:
    start = time.time() + 15
    ps = [run(p, start) for p in procs]
    print(name, " | ".join(p.communicate()[0].strip().splitlines()[-1] for p in ps), flush=True)
