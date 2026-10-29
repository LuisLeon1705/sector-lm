from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 16384
    width: int = 368
    layers: int = 4
    heads: int = 8
    context: int = 512
    rank: int = 32
    experts: int = 12


class StackedLoRA(nn.Module):
    """Domain LoRA factors with leading expert axis, including vectorized top-1 dispatch."""

    def __init__(self, experts: int, in_width: int, out_width: int, rank: int):
        super().__init__()
        self.a = nn.Parameter(torch.empty(experts, in_width, rank))
        self.b = nn.Parameter(torch.zeros(experts, rank, out_width))
        nn.init.normal_(self.a, std=0.02)
        self.scale = math.sqrt(rank)

    def one(self, x, expert_id: int):
        return ((x @ self.a[expert_id]) @ self.b[expert_id]) / self.scale

    def routed(self, x, gate_probs, dispatch):
        # Pack tokens by top-1 expert, then run two expert-batched GEMMs.
        shape = x.shape
        flat_x = x.reshape(-1, shape[-1])
        order, active_experts, sorted_group_ids, positions, capacity = dispatch
        grouped = flat_x.new_zeros((active_experts.numel(), capacity, flat_x.shape[-1]))
        grouped[sorted_group_ids, positions] = flat_x.index_select(0, order)
        active_a = self.a.index_select(0, active_experts)
        active_b = self.b.index_select(0, active_experts)
        low_rank = torch.bmm(grouped, active_a)
        grouped_delta = torch.bmm(low_rank, active_b)
        grouped_gates = gate_probs.new_zeros((active_experts.numel(), capacity))
        grouped_gates[sorted_group_ids, positions] = gate_probs.reshape(-1).index_select(0, order)
        grouped_delta = grouped_delta * grouped_gates.unsqueeze(-1) / self.scale
        sorted_delta = grouped_delta[sorted_group_ids, positions]
        delta = torch.empty_like(sorted_delta)
        delta.index_copy_(0, order, sorted_delta)
        return delta.reshape(*shape[:-1], self.b.shape[-1])


class LoRABank(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        e, d, r = cfg.experts, cfg.width, cfg.rank
        self.attn_qkv = StackedLoRA(e, d, 3*d, r)
        self.attn_out = StackedLoRA(e, d, d, r)
        self.mlp_up = StackedLoRA(e, d, 4*d, r)
        self.mlp_down = StackedLoRA(e, 4*d, d, r)
        self.mlp_gate = StackedLoRA(e, d, 4*d, r)

    def delta(self, name, x, expert_id=None, routes=None, gate_probs=None, dispatch=None):
        lora = getattr(self, name)
        if routes is None:
            return lora.one(x, int(expert_id))
        return lora.routed(x, gate_probs, dispatch)


class Block(nn.Module):
    def __init__(self, cfg: Config, with_router=False):
        super().__init__()
        d = cfg.width
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.up = nn.Linear(d, 4*d, bias=False)
        self.gate = nn.Linear(d, 4*d, bias=False)
        self.down = nn.Linear(4*d, d, bias=False)
        self.heads = cfg.heads
        self.router = nn.Linear(d, cfg.experts) if with_router else None
        if self.router is not None:
            nn.init.normal_(self.router.weight, std=0.02)
            nn.init.zeros_(self.router.bias)

    def forward(self, x, lora: LoRABank | None = None, expert_id=None, routed=False):
        routes = gate_probs = aux = None
        dispatch = None
        if self.router is not None:
            # Causal prefix summary makes even the first block's decision context-aware.
            positions = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)[None, :, None]
            router_state = x.cumsum(dim=1) / positions
            gate_logits = self.router(router_state)
            probs = gate_logits.softmax(-1)
            routes = gate_logits.argmax(-1)
            gate_probs = probs.gather(-1, routes.unsqueeze(-1)).squeeze(-1)
            flat_routes = routes.reshape(-1)
            active_experts, group_ids = torch.unique(flat_routes, sorted=True, return_inverse=True)
            order = torch.argsort(group_ids, stable=True)
            sorted_group_ids = group_ids.index_select(0, order)
            counts = torch.bincount(group_ids, minlength=active_experts.numel())
            capacity = int(counts.max().item())
            starts = counts.cumsum(0) - counts
            positions = torch.arange(flat_routes.numel(), device=x.device) - starts.index_select(0, sorted_group_ids)
            dispatch = (order, active_experts, sorted_group_ids, positions, capacity)
            hard_fraction = F.one_hot(routes, self.router.out_features).float().mean((0, 1))
            aux = self.router.out_features * torch.sum(hard_fraction.detach() * probs.mean((0, 1)))

        def delta(name, value):
            if lora is None:
                return None
            return lora.delta(name, value, expert_id=expert_id, routes=routes if routed else None,
                              gate_probs=gate_probs if routed else None, dispatch=dispatch)

        b, t, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)
        dv = delta("attn_qkv", h)
        if dv is not None:
            qkv = qkv + dv
        q, k, v = qkv.view(b, t, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(b, t, d)
        proj = self.proj(attn)
        dv = delta("attn_out", attn)
        if dv is not None:
            proj = proj + dv
        x = x + proj
        h = self.ln2(x)
        up, gate = self.up(h), self.gate(h)
        dv = delta("mlp_up", h)
        if dv is not None:
            up = up + dv
        dv = delta("mlp_gate", h)
        if dv is not None:
            gate = gate + dv
        hidden = F.silu(gate) * up
        down = self.down(hidden)
        dv = delta("mlp_down", hidden)
        if dv is not None:
            down = down + dv
        return x + down, aux, routes


class GPT(nn.Module):
    def __init__(self, cfg: Config, with_experts=False, with_router=False):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position_embedding = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.ModuleList([Block(cfg, with_router=with_router) for _ in range(cfg.layers)])
        self.final_norm = nn.LayerNorm(cfg.width)
        self.experts = nn.ModuleList([LoRABank(cfg) for _ in range(cfg.layers)]) if with_experts else nn.ModuleList()
        self.last_routes = []
        self.last_router_aux = None
        self._init_weights()

    def _init_weights(self):
        # GPT-2 style: N(0, 0.02), residual outputs scaled by depth. The output layer is tied to the
        # token embedding, so PyTorch's default N(0, 1) embedding would start with huge logits.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            for residual_out in (block.proj, block.down):
                nn.init.normal_(residual_out.weight, std=0.02 / math.sqrt(2 * self.cfg.layers))

    def forward(self, tokens, expert_id: int | None = None, use_expert=True):
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(pos)[None, :, :]
        self.last_routes = []
        aux_values = []
        for layer, block in enumerate(self.blocks):
            bank = self.experts[layer] if use_expert and self.experts else None
            x, aux, routes = block(x, bank, expert_id=expert_id, routed=block.router is not None)
            if routes is not None:
                self.last_routes.append(routes.detach())
            if aux is not None:
                aux_values.append(aux)
        self.last_router_aux = torch.stack(aux_values).mean() if aux_values else None
        x = self.final_norm(x)
        return F.linear(x, self.token_embedding.weight)

    def adapter_trainable(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.experts.parameters():
            p.requires_grad_(True)


class SwitchGPT(GPT):
    def __init__(self, cfg: Config):
        super().__init__(cfg, with_experts=True, with_router=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_models(vocab_size=16384, experts=12):
    base_cfg = Config(vocab_size=vocab_size, width=368, heads=8, experts=experts)
    d1 = GPT(base_cfg)
    e1 = GPT(base_cfg, with_experts=True)
    m1 = SwitchGPT(base_cfg)

    def dense_count(d):
        return vocab_size*d + 512*d + 64*d*d + 18*d

    base_count = dense_count(base_cfg.width)
    lora_count = 84 * base_cfg.width * base_cfg.rank * experts
    e1_total = base_count + lora_count
    router_count = base_cfg.layers * (base_cfg.width*experts + experts)
    m1_total = e1_total + router_count
    width = min(range(384, 1025, 8), key=lambda d: abs(dense_count(d)-m1_total))
    d2 = GPT(Config(vocab_size=vocab_size, width=width, layers=4, heads=8, experts=experts))
    return {"D1": d1, "D2": d2, "E1": e1, "M1": m1}


def copy_shared_base(source: GPT, target: GPT):
    """Copy D1's dense checkpoint into E1 while leaving its adapter bank initialized."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    shared = {key: value for key, value in source_state.items() if key in target_state and not key.startswith("experts.")}
    target_state.update(shared)
    target.load_state_dict(target_state)
    return len(shared)
