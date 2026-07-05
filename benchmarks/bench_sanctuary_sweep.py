import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from resapexquant import inject_resapexquant, calibrate, set_active, SC

SEED=2025; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

def get_texts(n=80):
    try:
        wiki=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
        return [t for t in wiki["text"] if len(t.split())>25][:n]
    except Exception:
        return ["Transformer architecture relies on self-attention. "*4]*n

def compute_ppl(model, tok, modules, texts, active, n_pred=20):
    set_active(modules,active); nlls=[]
    for text in texts[:40]:
        ids=tok(text,return_tensors="pt",truncation=True,max_length=96).input_ids.to(DEVICE)
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
    return math.exp(min(np.mean(nlls),20)) if nlls else float("inf"), len(nlls)

texts=get_texts()
for model_id, sizes in [
    ("Qwen/Qwen2.5-1.5B-Instruct",[0,2,3,6,8]),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct",[0,4,6,8]),
]:
    print(f"\n{model_id}")
    tok=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForCausalLM.from_pretrained(model_id,device_map="auto",torch_dtype=torch.float16).eval()
    NL=model.config.num_hidden_layers; NKV=getattr(model.config,"num_key_value_heads",model.config.num_attention_heads)
    NH=model.config.num_attention_heads; print(f"  layers={NL}, n_kv={NKV}, r={NH/NKV:.1f}")
    for s in sizes:
        modules=inject_resapexquant(model,bits_shallow=8,bits_deep=3,sanctuary=s)
        calibrate(model,tok,modules,texts[:6])
        ppl_fp,_=compute_ppl(model,tok,modules,texts,False)
        ppl_q,n=compute_ppl(model,tok,modules,texts,True)
        gain=NL/(s*(8/16)+(NL-s)*(3/16))
        print(f"  sanctuary={s:2d} ({s/NL*100:4.1f}%)  dPPL={ppl_q-ppl_fp:+7.2f} (n={n})  gain={gain:.2f}x")
    del model; torch.cuda.empty_cache() if DEVICE=="cuda" else None
print(f"Sc={SC} -- Patent pending INPI France 2026")
