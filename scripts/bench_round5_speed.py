# Mide velocidad y memoria de GPU de cada modelo de la ronda 5, uno a la vez y con la GPU limpia.
# Solo lectura: carga los checkpoints, no escribe nada en el proyecto.
import gc, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.model import build_models, SwitchGPT

dev = "cuda"
CK = str(Path(__file__).resolve().parents[1] / "results/round5/checkpoints/round5c4") + "/"
FILES = {"D1": "D1", "D1+": "D1plus", "D2": "D2", "E1": "E1_all", "M1": "M1"}
ARCH = {"D1": "D1", "D1+": "D1", "D2": "D2", "E1": "E1", "M1": "M1"}
MB = 1024 ** 2


def run(model, x, expert):
    if isinstance(model, SwitchGPT):
        return model(x)
    return model(x, expert_id=expert, use_expert=expert is not None)


def timed(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


print(f"{'modelo':<6} {'pesos MB':>9} {'inf. lote tok/s':>15} {'inf. lote pico MB':>17} "
      f"{'1 secuencia ms':>14} {'entren. tok/s':>13} {'entren. pico MB':>15}")
for name in FILES:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = build_models(vocab_size=16384, experts=12)[ARCH[name]]
    model.load_state_dict(torch.load(CK + FILES[name] + ".pt", map_location="cpu", weights_only=False)["model"])
    model.to(dev)
    expert = 3 if name == "E1" else None
    weights = torch.cuda.memory_allocated() / MB
    x = torch.randint(2, 16384, (8, 512), device=dev)
    y = torch.randint(2, 16384, (8, 512), device=dev)

    model.eval(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        batch_s = timed(lambda: run(model, x, expert), 20)
        inf_peak = torch.cuda.max_memory_allocated() / MB
        one_s = timed(lambda: run(model, x[:1], expert), 30)

    model.train()
    if name == "E1":  # E1 entrena solo sus adaptadores, con la base congelada
        model.adapter_trainable()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = run(model, x, expert)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            if isinstance(model, SwitchGPT):
                loss = loss + 0.1 * model.last_router_aux
        loss.backward(); opt.step()

    torch.cuda.reset_peak_memory_stats()
    train_s = timed(step, 10)
    train_peak = torch.cuda.max_memory_allocated() / MB
    n = x.numel()
    print(f"{name:<6} {weights:>9.0f} {n / batch_s:>15,.0f} {inf_peak:>17.0f} {one_s * 1000:>14.1f} "
          f"{n / train_s:>13,.0f} {train_peak:>15.0f}")
    del model, opt, params
