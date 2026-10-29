# Misma medición que speed.py, para bigcode/tiny_starcoder_py desde la caché local de Hugging Face.
# Pesos en fp32 y autocast bf16, igual que los modelos de la ronda 5.
import os, time, torch, torch.nn.functional as F
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForCausalLM

dev = "cuda"
MB = 1024 ** 2
torch.cuda.reset_peak_memory_stats()
model = AutoModelForCausalLM.from_pretrained("bigcode/tiny_starcoder_py", dtype=torch.float32).to(dev)
n_params = sum(p.numel() for p in model.parameters())
weights = torch.cuda.memory_allocated() / MB
V = model.config.vocab_size


def timed(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


x = torch.randint(0, V, (8, 512), device=dev)
model.eval(); torch.cuda.reset_peak_memory_stats()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    batch_s = timed(lambda: model(input_ids=x, use_cache=False).logits, 20)
    inf_peak = torch.cuda.max_memory_allocated() / MB
    one_s = timed(lambda: model(input_ids=x[:1], use_cache=False).logits, 30)
print(f"parámetros {n_params:,} | pesos {weights:.0f} MB | inferencia lote {x.numel() / batch_s:,.0f} tok/s"
      f" | pico {inf_peak:.0f} MB | 1 secuencia {one_s * 1000:.1f} ms")

model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
# Con lote 8 pide ~12 GB: en una GPU de 8 GB, Windows lo pasa a RAM compartida y la cifra no sirve.
for batch in (2, 1):
    try:
        xb = x[:batch]; yb = torch.randint(0, V, (batch, 512), device=dev)

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=xb, use_cache=False).logits
                loss = F.cross_entropy(logits.reshape(-1, V).float(), yb.reshape(-1))
            loss.backward(); opt.step()

        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        s = timed(step, 10)
        print(f"entrenamiento lote {batch}x512: {xb.numel() / s:,.0f} tok/s | pico {torch.cuda.max_memory_allocated() / MB:.0f} MB")
        break
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        print(f"entrenamiento lote {batch}x512: no cabe en 8 GB")
