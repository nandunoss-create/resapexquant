# ResApexQuant

**KV Cache Compression for Mobile-Targeted Transformer LLMs**

[![Patent Pending](https://img.shields.io/badge/Patent-Pending%20INPI%20FR%202026-blue)](.)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](.)

---

## Key Results (WikiText-2, seed=2025, T4 GPU)

| Method | Bits | ΔPPL | Gain |
|--------|------|------|------|
| TurboQuant (ICLR 2026) | 2-bit | +402 | 8× |
| KVQuant (NeurIPS 2024) | 2-bit | +1376 | 8× |
| **ResApexQuant Qwen2.5-1.5B** | **8b/3b** | **+0.09** | **3.61×** |
| ResApexQuant SmolLM2-1.7B | 8b/3b | +0.62 | 3.43× |

**Speed:** 15.7 tok/s (vs 15.6 FP16) — zero overhead.  
**Memory (8192 tokens):** 3322 MB → 837 MB (-74.8%) with AWQ+ResApexQuant.

---

## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.resapexquant import inject_resapexquant, calibrate, set_active

tok   = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", device_map="auto", torch_dtype="float16").eval()

modules = inject_resapexquant(model, bits_shallow=8, bits_deep=3, sanctuary=8)
calibrate(model, tok, modules, [
    "The transformer architecture relies on self-attention.",
    "Memory efficiency is critical for mobile LLM deployment.",
])
set_active(modules, True)

ids = tok("What is general relativity?", return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=80)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## GQA-Aware Sanctuary Rule (Claim 6)

| Architecture | n_kv | ratio r | Optimal Sanctuary | ΔPPL | Gain |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 2 | 6 | **8 layers (required!)** | +0.09 | 3.61× |
| SmolLM2-1.7B (LLaMA) | 32 | 1 | 0–8 layers | +0.62–0.71 | 3.43–5.33× |

**Rule:** For GQA ratio r = n_h/n_kv ≥ 6, a substantial sanctuary is mandatory
(sanctuary=0 causes ΔPPL=+3868 on Qwen). For r ≈ 1, sanctuary is optional (+55% gain).

---

## File Structure

```
resapexquant/
├── src/resapexquant.py          # Core algorithm (production API)
├── benchmarks/
│   ├── bench_quality.py         # WikiText-2 PPL + cosine logits
│   ├── bench_sanctuary_sweep.py # Optimal sanctuary per architecture
│   ├── bench_vs_awq.py          # AWQ complementarity benchmark
│   └── bench_kernel_speed.py    # GPU compression speedup
├── demos/gradio_live_demo.py    # Interactive Gradio demo (shareable link)
├── docs/results.md              # All validated results
├── results/validated_results.json
└── README.md
```

---

## Citation

```bibtex
@misc{resapexquant2026,
  title  = {ResApexQuant: GQA-Aware KV Cache Compression for Mobile LLM Deployment},
  author = {SR-SE Research Unit},
  year   = {2026},
  note   = {Patent pending INPI France 2026. Sc = 0.769893}
}
```

**Sc = 0.769893 — Patent pending INPI France 2026**
