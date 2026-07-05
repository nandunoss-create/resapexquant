import sys, os, math, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from resapexquant import inject_resapexquant, calibrate, set_active, SC

SEED = 2025; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_texts(n=50):
    try:
        wiki = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
        return [t for t in wiki["text"] if len(t.split())>25][:n]
    except Exception:
        return ["The transformer architecture relies on self-attention. "*3]*n

def measure_cosine(model, tok, modules, texts):
    cos_list, top1_list = [], []
    for text in texts[:25]:
        ids = tok(text,return_tensors="pt",truncation=True,max_length=56).input_ids.to(DEVICE)
        T = ids.shape[1]
        if T < 6: continue
        for pos in list(range(max(3,T//3),T-1,max(1,(T-4)//3)))[:3]:
            ctx = ids[:,:pos]
            with torch.no_grad():
                set_active(modules,False); lf=model(ctx,use_cache=False).logits[0,-1].float().cpu()
                set_active(modules,True);  lc=model(ctx,use_cache=False).logits[0,-1].float().cpu()
                set_active(modules,False)
            cos_list.append(float(torch.nn.functional.cosine_similarity(lf.unsqueeze(0),lc.unsqueeze(0))))
            top1_list.append((lf.argmax()==lc.argmax()).item())
    return np.mean(cos_list), np.mean(top1_list)*100

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

CONFIGS=[
    {"model_id":"Qwen/Qwen2.5-1.5B-Instruct","sanctuary":8,"bs":8,"bd":3},
    {"model_id":"HuggingFaceTB/SmolLM2-1.7B-Instruct","sanctuary":8,"bs":8,"bd":3},
]
results=[]; texts=get_texts()
for cfg in CONFIGS:
    print(f"\n{cfg['model_id']}")
    tok=AutoTokenizer.from_pretrained(cfg["model_id"])
    model=AutoModelForCausalLM.from_pretrained(cfg["model_id"],device_map="auto",torch_dtype=torch.float16).eval()
    modules=inject_resapexquant(model,bits_shallow=cfg["bs"],bits_deep=cfg["bd"],sanctuary=cfg["sanctuary"])
    calibrate(model,tok,modules,texts[:6])
    NL=model.config.num_hidden_layers
    gain=NL/(cfg["sanctuary"]*(cfg["bs"]/16)+(NL-cfg["sanctuary"])*(cfg["bd"]/16))
    cos,top1=measure_cosine(model,tok,modules,texts)
    ppl_fp=compute_ppl(model,tok,modules,texts,False)
    ppl_q=compute_ppl(model,tok,modules,texts,True)
    delta=ppl_q-ppl_fp
    print(f"  Cosine={cos:.5f} Top1={top1:.0f}% dPPL={delta:+.2f} Gain={gain:.2f}x")
    results.append({"model":cfg["model_id"],"cosine":cos,"top1":top1,"ppl_fp16":ppl_fp,"ppl_q":ppl_q,"delta_ppl":delta,"gain_kv":gain})
    del model; torch.cuda.empty_cache() if DEVICE=="cuda" else None
os.makedirs("../results",exist_ok=True)
with open("../results/validated_results.json","w") as f: json.dump(results,f,indent=2)
print(f"[OK] Sc={SC} -- Patent pending INPI France 2026")
