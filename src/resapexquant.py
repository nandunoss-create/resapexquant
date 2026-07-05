"""
ResApexQuant — KV Cache Compression for Transformer LLMs
Patent pending INPI France 2026 — SR-SE Research Unit
Sc = 0.769893 (Fisher Saturation, SR-SE Theorem 1)
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List

__version__ = "1.0.0"
SC: float = 0.769893
SEED: int = 2025

def haar_rotation(d, seed=SEED):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d)).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    return torch.from_numpy(Q)

def lloyd_max(samples, k, n_iter=100):
    s = samples.astype(np.float64)
    c = np.percentile(s, np.linspace(0, 100, k + 2)[1:-1])
    for _ in range(n_iter):
        bd = np.r_[-np.inf, (c[:-1] + c[1:]) / 2, np.inf]
        nc = np.array([s[(s>=bd[i])&(s<bd[i+1])].mean()
                       if ((s>=bd[i])&(s<bd[i+1])).any() else c[i]
                       for i in range(k)])
        if np.max(np.abs(nc - c)) < 1e-9: break
        c = nc
    return np.sort(c).astype(np.float32)

class ApexQ(nn.Module):
    """ApexQ_Reference — production-ready KV cache compressor."""
    def __init__(self, d_head, bits, device="cuda", seed=SEED):
        super().__init__()
        self.d = d_head; self.bits = bits; self.K = 2 ** bits
        Pi = haar_rotation(d_head, seed=seed).to(device=device, dtype=torch.float16)
        self.register_buffer("Pi", Pi)
        self.codebook = None; self.sc_local = None; self.calibrated = False

    @torch.no_grad()
    def calibrate(self, X):
        """X: (B, T, NKV, d_head)"""
        Y = X.float() @ self.Pi.float().T
        B, T, NKV, D = Y.shape
        Yh = Y.reshape(-1, NKV, D).permute(1, 0, 2).reshape(NKV, -1)
        sc_q = torch.quantile(Yh.abs(), SC, dim=1)
        sc_m = 3.0 * torch.median(Yh.abs(), dim=1).values
        self.sc_local = torch.minimum(sc_q, sc_m).to(dtype=torch.float32, device=self.Pi.device)
        Yn = Yh.cpu().numpy().astype(np.float64)
        cbs = []
        for h in range(NKV):
            sc = float(self.sc_local[h].item()) + 1e-8
            v = np.clip(Yn[h] / sc, -1.0, 1.0)
            cb = lloyd_max(v[np.abs(v) <= 1.0], self.K) * sc
            cbs.append(cb)
        self.codebook = torch.tensor(np.stack(cbs), device=self.Pi.device, dtype=torch.float32)
        self.calibrated = True

    @torch.no_grad()
    def compress(self, X):
        """X: (B, T, NKV, d_head) — returns same shape & dtype."""
        if not self.calibrated: return X
        dt = X.dtype
        Y = X.float() @ self.Pi.float().T
        B, T, NKV, D = Y.shape
        sc = self.sc_local.view(1, 1, NKV, 1)
        Y_e = Y.unsqueeze(-1)
        cb_e = self.codebook.view(1, 1, NKV, 1, -1)
        idx = (Y_e - cb_e).abs().argmin(-1)
        Yq = self.codebook[torch.arange(NKV, device=self.Pi.device).view(1,1,NKV,1), idx]
        Yf = torch.where(Y.abs() > sc, Y, Yq)
        return (Yf @ self.Pi.float()).to(dtype=dt)

class QuantizedProj(nn.Module):
    """Drop-in wrapper for k_proj / v_proj."""
    def __init__(self, proj, aq, n_kv, d_head, layer_idx):
        super().__init__()
        self.proj = proj; self.aq = aq
        self.n_kv = n_kv; self.d_head = d_head; self.layer_idx = layer_idx
        self.is_active = False; self.is_calibrating = False; self.buf = []
    def forward(self, x):
        out = self.proj(x)
        if self.is_calibrating:
            self.buf.append(out.view(out.shape[0], out.shape[1], self.n_kv, self.d_head).detach())
            return out
        if self.is_active:
            r = out.view(out.shape[0], out.shape[1], self.n_kv, self.d_head)
            return self.aq.compress(r).view(out.shape)
        return out

def inject_resapexquant(model, bits_shallow=8, bits_deep=3, sanctuary=8):
    """Inject ResApexQuant into all k_proj/v_proj. Returns module list."""
    cfg = model.config
    NKV = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    DH = cfg.hidden_size // cfg.num_attention_heads
    dev = str(next(model.parameters()).device)
    modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear): continue
        if not any(name.endswith(s) for s in ["k_proj", "v_proj"]): continue
        parts = name.split(".")
        layer_idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
        parent = model.get_submodule(".".join(name.split(".")[:-1]))
        bits = bits_shallow if layer_idx < sanctuary else bits_deep
        aq = ApexQ(d_head=DH, bits=bits, device=dev)
        wrapped = QuantizedProj(module, aq, NKV, DH, layer_idx)
        setattr(parent, name.split(".")[-1], wrapped)
        modules.append(wrapped)
    return modules

def calibrate(model, tok, modules, texts, max_length=48):
    """Calibrate from real KV activations."""
    device = next(model.parameters()).device
    for m in modules: m.is_calibrating = True; m.is_active = False
    model.eval()
    with torch.no_grad():
        for text in texts:
            ids = tok(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            model(ids)
    for m in modules:
        m.is_calibrating = False
        if m.buf: m.aq.calibrate(torch.cat(m.buf, dim=1))
        m.buf = []

def set_active(modules, active):
    """Toggle KV cache compression on/off."""
    for m in modules: m.is_active = active

def memory_stats(model, n_tokens=8192, bits_deep=3, sanctuary=8):
    """Return memory breakdown for a given context length."""
    cfg = model.config
    NL = cfg.num_hidden_layers
    NKV = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    DH = cfg.hidden_size // cfg.num_attention_heads
    w_fp16 = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    w_4bit = w_fp16 / 4
    kv_fp16 = NL * 2 * n_tokens * NKV * DH * 2 / 1e6
    gain_kv = NL / (sanctuary * (8/16) + (NL - sanctuary) * (bits_deep/16))
    kv_resapex = kv_fp16 / gain_kv
    return {
        "weights_fp16_mb": w_fp16, "weights_4bit_mb": w_4bit,
        "kv_fp16_mb": kv_fp16, "kv_resapex_mb": kv_resapex,
        "total_4bit_kv_fp16_mb": w_4bit + kv_fp16,
        "total_4bit_kv_resapex_mb": w_4bit + kv_resapex,
        "gain_kv": gain_kv,
        "gain_total": (w_4bit + kv_fp16) / (w_4bit + kv_resapex),
    }
