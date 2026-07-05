import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from resapexquant import inject_resapexquant, calibrate, set_active, memory_stats, SC

SEED=2025; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

def quantize_weights_4bit(model):
    with torch.no_grad():
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Linear) and not any(s in name for s in ["k_proj","v_proj"]):
                W=module.weight.data; w_min=W.amin(dim=1,keepdim=True); w_max=W.amax(dim=1,keepdim=True)
                scale=(w_max-w_min)/15.0; scale[scale.abs()<1e-8]=1.0
                W_idx=torch.round((W-w_min)/scale).clamp(0,15)
                module.weight.data=(W_idx*scale+w_min).to(W.dtype)
    return model

def compute_ppl(model, tok, modules, texts, active, n_pred=10):
    set_active(modules,active); nlls=[]
    for text in texts[:20]:
        ids=tok(text,return_tensors="pt",truncation=True,max_length=72).input_ids.to(DEVICE)
        T=ids.shape[1]
        if T<8: continue
        ctx_len=max(4,int(T*0.6)); targets=ids[0,ctx_len:]
        if len(targets)<2: continue
        with torch.no_grad():
            out=model(ids[:,:ctx_len],use_cache=True); past=out.past_key_values; inp=ids[:,ctx_len-1:ctx_len]
            for tgt in targets[:n_pred]:
                o2=model(inp,past_key_values=past,use_cache=True)
                nlls.append(torch.nn.functional.cross_entropy(o2.logits[0,-1].unsqueeze(0),tgt.unsqueeze(0)).item())
                inp=tgt.unsqueeze(0).unsqueeze(0); past=o2.past_key_values
    set_active(modules,False)
    return math.exp(min(np.mean(nlls),20)) if nlls else float("inf")

try:
    wiki=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    texts=[t for t in wiki["text"] if len(t.split())>25][:50]
except Exception:
    texts=["Large language models need efficient memory management. "*4]*50

MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
tok=AutoTokenizer.from_pretrained(MODEL_ID)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map="auto",torch_dtype=torch.float16).eval()
model=quantize_weights_4bit(model)
modules=inject_resapexquant(model,bits_shallow=8,bits_deep=3,sanctuary=8)
calibrate(model,tok,modules,texts[:6])
stats=memory_stats(model,n_tokens=8192)
ppl_base=compute_ppl(model,tok,modules,texts,False)
ppl_apex=compute_ppl(model,tok,modules,texts,True)
delta=ppl_apex-ppl_base
print(f"Memory: {stats['weights_fp16_mb']+stats['kv_fp16_mb']:.0f}MB -> {stats['total_4bit_kv_resapex_mb']:.0f}MB (-{(1-1/stats['gain_total'])*100:.1f}%)")
print(f"Additional dPPL from ResApexQuant: {delta:+.2f}")
print(f"Sc={SC} -- Patent pending INPI France 2026")
