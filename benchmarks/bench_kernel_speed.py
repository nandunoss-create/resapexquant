import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from resapexquant import ApexQ, inject_resapexquant, calibrate, set_active, SC

SEED=2025; torch.manual_seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
D,NKV,T,BITS=128,2,512,3

def benchmark(fn, x, n=50, w=10):
    for _ in range(w): fn(x)
    if DEVICE=="cuda": torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(n): fn(x)
    if DEVICE=="cuda": torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n*1000

X_cal=torch.randn(4,48,NKV,D,device=DEVICE,dtype=torch.float16)
X_test=torch.randn(1,T,NKV,D,device=DEVICE,dtype=torch.float16)
q_gpu=ApexQ(D,BITS,device=DEVICE); q_gpu.calibrate(X_cal)
q_cpu=ApexQ(D,BITS,device="cpu")
q_cpu.Pi=q_gpu.Pi.cpu(); q_cpu.codebook=q_gpu.codebook.cpu()
q_cpu.sc_local=q_gpu.sc_local.cpu(); q_cpu.calibrated=True
t_gpu=benchmark(q_gpu.compress,X_test)
t_cpu=benchmark(q_cpu.compress,X_test.cpu(),n=20,w=3)
print(f"NumPy CPU  : {t_cpu:.2f}ms (1.0x)")
print(f"PyTorch GPU: {t_gpu:.2f}ms ({t_cpu/t_gpu:.1f}x speedup)")

MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
tok=AutoTokenizer.from_pretrained(MODEL_ID)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map="auto",torch_dtype=torch.float16).eval()
modules=inject_resapexquant(model,bits_shallow=8,bits_deep=3,sanctuary=8)
calibrate(model,tok,modules,["The transformer architecture relies on self-attention.","Memory efficiency is critical for mobile deployment.","Quantization reduces footprint while preserving quality."])
def tps(active,n=30):
    msgs=[{"role":"user","content":"Explain general relativity briefly."}]
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ids=tok(text,return_tensors="pt").input_ids.to(DEVICE)
    set_active(modules,active)
    if DEVICE=="cuda": torch.cuda.synchronize()
    t0=time.perf_counter()
    with torch.no_grad(): model.generate(ids,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id)
    if DEVICE=="cuda": torch.cuda.synchronize()
    set_active(modules,False); return n/(time.perf_counter()-t0)
fp16=tps(False); comp=tps(True)
print(f"FP16: {fp16:.1f} tok/s  |  ResApexQuant: {comp:.1f} tok/s  ({comp/fp16:+.1%})")
print(f"Sc={SC} -- Patent pending INPI France 2026")
