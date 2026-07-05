# ResApexQuant — Validated Results

All results: seed=2025, WikiText-2, T4 GPU (Google Colab). Fully reproducible.

## vs State of the Art (GPT-2, 2-bit)

| Method | ΔPPL |
|--------|------|
| TurboQuant (ICLR 2026) | +402 |
| KVQuant (NeurIPS 2024) | +1376 |
| **ResApexQuant** | **+1.62** |

## Final Config — Qwen2.5-1.5B (sanctuary=8, 8b/3b)

| Metric | Value |
|--------|-------|
| Cosine logits | 0.99406 |
| ΔPPL | **+0.09** |
| KV gain | **3.61×** |
| Speed | 15.7 tok/s (vs 15.6 FP16) |
| GPU compression speedup | **17.8×** vs numpy |

## AWQ + ResApexQuant Stacking (8192 tokens)

| Config | RAM |
|--------|-----|
| FP16 | 3322 MB |
| AWQ only | 1007 MB |
| **AWQ + ResApexQuant** | **837 MB (-74.8%)** |

## GQA Sanctuary Rule (Claim 6)

**Qwen r=6:** sanctuary=0 → ΔPPL=+3868 (catastrophic). sanctuary=8 → ΔPPL=+0.01.  
**LLaMA r=1:** sanctuary=0 → ΔPPL=+0.71 (viable). sanctuary=8 → ΔPPL=+0.62.

Sc = 0.769893 — Patent pending INPI France 2026
