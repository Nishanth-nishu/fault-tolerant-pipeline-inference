import sys, time, torch, psutil
sys.path.insert(0,".")
from pllm.model import Stage, SeqSlice, load_hf
cpus=[int(c) for c in sys.argv[1].split(",")]; lo,hi=int(sys.argv[2]),int(sys.argv[3]); dur=float(sys.argv[4])
psutil.Process().cpu_affinity(cpus); torch.set_num_threads(len(cpus))
tok, hf = load_hf("gpt2"); st = Stage(hf, lo, hi); H=768; bs=1
pre=[SeqSlice("b",0,64)]; dec=[SeqSlice("b",64,1)]
st.forward(torch.randint(0,1000,(64,)) if lo==0 else torch.randn(64,H), pre)
x = torch.randint(0,1000,(1,)) if lo==0 else torch.randn(1,H)
time.sleep(max(0, float(sys.argv[5]) - time.time()))
t=time.perf_counter(); n=0
while time.perf_counter()-t < dur: st.forward(x, dec); n+=1
print(sys.argv[1], f"{(time.perf_counter()-t)/n*1e3:.2f} ms/step")
