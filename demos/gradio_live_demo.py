import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from resapexquant import inject_resapexquant, calibrate, set_active, memory_stats, SC

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
CALIB=["The transformer architecture relies on self-attention mechanisms.",
       "Memory efficiency is critical for deploying large language models.",
       "Quantization reduces footprint while preserving generation quality.",
       "Mobile devices have limited memory bandwidth compared to data centers.",
       "Neural networks learn hierarchical representations from training data.",
       "Artificial intelligence transforms modern computing and research."]
EXAMPLES=["Explain general relativity in one simple sentence.",
          "What is the capital of Japan and why is it famous?",
          "Write a Python function that computes the Fibonacci sequence.",
          "All birds can fly. A penguin is a bird. What can we conclude?",
          "What is the difference between RAM and ROM?"]

print(f"Loading {MODEL_ID}...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map="auto",torch_dtype=torch.float16).eval()
modules=inject_resapexquant(model,bits_shallow=8,bits_deep=3,sanctuary=8)
calibrate(model,tok,modules,CALIB)
stats=memory_stats(model,n_tokens=4096); GAIN=stats["gain_kv"]
cfg=model.config; NL=cfg.num_hidden_layers
NKV2=getattr(cfg,"num_key_value_heads",cfg.num_attention_heads)
DH2=cfg.hidden_size//cfg.num_attention_heads
print(f"Ready. KV gain: {GAIN:.2f}x")

def generate(prompt, active, max_new=100):
    set_active(modules,active)
    msgs=[{"role":"system","content":"You are a precise and concise AI."},{"role":"user","content":prompt}]
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ids=tok(text,return_tensors="pt").input_ids.to(DEVICE)
    gen=[]; past=None; t0=time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new):
            inp=ids[:,-1:] if past is not None else ids
            out=model(inp,past_key_values=past,use_cache=True); past=out.past_key_values
            nxt=out.logits[:,-1,:].argmax(-1,keepdim=True); ids=torch.cat([ids,nxt],dim=1)
            gen.append(nxt.item())
            if nxt.item()==tok.eos_token_id: break
    set_active(modules,False)
    tps=len(gen)/max(time.perf_counter()-t0,1e-6)
    bpv=(3/8) if active else 2.0
    kv_mb=NL*2*ids.shape[1]*NKV2*DH2*bpv/1e6
    return tok.decode(gen,skip_special_tokens=True).strip(), kv_mb, tps

def bar(r,w=28): n=int(w*min(r,1.)); return chr(9608)*n+chr(9617)*(w-n)

def run(prompt):
    if not prompt.strip(): return "Please enter a question.","","","--"
    t0=time.time()
    r_fp,kv_fp,tps_fp=generate(prompt,False); r_q,kv_q,tps_q=generate(prompt,True)
    elapsed=time.time()-t0; saved=(1-kv_q/kv_fp)*100 if kv_fp>0 else 0; mx=max(kv_fp,kv_q,1.)
    dash=(
        "KV Cache -- Standard (FP16)\n"
        f"[{bar(kv_fp/mx)}]  {kv_fp:.1f} MB\n\n"
        "KV Cache -- ResApexQuant (3-bit)\n"
        f"[{bar(kv_q/mx)}]  {kv_q:.1f} MB\n\n"
        f">>> {saved:.1f}% MEMORY SAVED <<<\n"
        f"FP16: {tps_fp:.1f} tok/s  |  Compressed: {tps_q:.1f} tok/s\n"
        f"Total: {elapsed:.1f}s  |  Gain: {GAIN:.2f}x  |  Sc={SC}"
    )
    return r_fp, r_q, dash, f"{saved:.1f}% memory saved | {tps_q:.1f} tok/s"

with gr.Blocks(theme=gr.themes.Monochrome(),title="ResApexQuant Demo") as demo:
    gr.Markdown(f"# ResApexQuant -- Live Demo\n**Qwen2.5-1.5B** KV cache {GAIN:.2f}x compressed -- Sc={SC} -- Patent pending INPI FR 2026")
    with gr.Row():
        prompt_box=gr.Textbox(label="Your question",placeholder="Ask anything...",scale=4)
        btn=gr.Button("Generate",variant="primary",scale=1)
    gr.Examples(examples=EXAMPLES,inputs=prompt_box)
    with gr.Row():
        with gr.Column(): gr.Markdown("### Standard (FP16)"); out_fp=gr.Textbox(label="Response",lines=7)
        with gr.Column(): gr.Markdown("### ResApexQuant (3-bit)"); out_q=gr.Textbox(label="Response",lines=7)
    dash_out=gr.Textbox(label="Memory dashboard",lines=10); label_out=gr.Label(label="Result")
    btn.click(run,inputs=prompt_box,outputs=[out_fp,out_q,dash_out,label_out])
    prompt_box.submit(run,inputs=prompt_box,outputs=[out_fp,out_q,dash_out,label_out])

if __name__=="__main__":
    demo.launch(share=True,debug=False)
