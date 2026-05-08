"""Block-octonion HRR primitives — self-contained, no external deps.

All tensors are float32; the last axis has shape 8 (one octonion) or a
multiple of 8 (a "wide octonion vector" of K blocks, dim = 8 * K).

Main operations:
  * oct_mul(a, b)              — element-wise octonion product on the last axis
  * oct_conj(o)                — octonion conjugate
  * oct_norm_sq / oct_norm     — norm
  * oct_normalize              — projection onto unit S^7 (for unit octonions)
  * block_oct_mul              — same on dim = 8 * K blocks
  * bind / unbind              — HRR bind/unbind on octonions
  * superpose                  — normalized superposition of crystals
  * cleanup                    — iterative nearest-neighbor cleanup
  * random_unit_octonion       — generator of unit octonions on S^7

Moufang identity (alternative algebra): for unit `k`,
    k̄ · (k · v) = (k̄ · k) · v = v
which makes unbind via conjugation correct.
"""
from __future__ import annotations

from typing import Iterable

import torch


# -----------------------------------------------------------------------------
# Cayley–Dickson reference product (используется только для построения структурных
# констант; в forward — один einsum на 8x8x8 константах).
# -----------------------------------------------------------------------------
def _reference_octonion_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """o = (q1, q2), o' = (q3, q4) → o*o' = (q1*q3 - q4_bar*q2, q4*q1 + q2*q3_bar)."""
    a_q1, a_q2 = a[..., :4], a[..., 4:]
    b_q1, b_q2 = b[..., :4], b[..., 4:]

    def quat_mul(p, q):
        a_, b_, c_, d_ = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        e_, f_, g_, h_ = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.stack([
            a_ * e_ - b_ * f_ - c_ * g_ - d_ * h_,
            a_ * f_ + b_ * e_ + c_ * h_ - d_ * g_,
            a_ * g_ - b_ * h_ + c_ * e_ + d_ * f_,
            a_ * h_ + b_ * g_ - c_ * f_ + d_ * e_,
        ], dim=-1)

    def quat_conj(q):
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

    out_q1 = quat_mul(a_q1, b_q1) - quat_mul(quat_conj(b_q2), a_q2)
    out_q2 = quat_mul(b_q2, a_q1) + quat_mul(a_q2, quat_conj(b_q1))
    return torch.cat([out_q1, out_q2], dim=-1)


def _build_struct_const() -> torch.Tensor:
    e = torch.eye(8, dtype=torch.float32)
    C = torch.zeros(8, 8, 8, dtype=torch.float32)
    for i in range(8):
        for j in range(8):
            C[i, j] = _reference_octonion_mul(e[i], e[j])
    return C


_C = _build_struct_const()  # [8, 8, 8] — структурные константы

# Маска сопряжения: ō = (real, -imag1, …, -imag7).
_CONJ = torch.tensor([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])


# -----------------------------------------------------------------------------
# Атомные операции на одном октонионе (последняя ось = 8).
# -----------------------------------------------------------------------------
def oct_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Поэлементный octonion product. a, b: [..., 8] → [..., 8].

    out[..., k] = Σ_{i,j} C[i,j,k] * a[..., i] * b[..., j].
    """
    C = _C.to(device=a.device, dtype=a.dtype)
    return torch.einsum("...i,...j,ijk->...k", a, b, C)


def oct_conj(o: torch.Tensor) -> torch.Tensor:
    sign = _CONJ.to(device=o.device, dtype=o.dtype)
    return o * sign


def oct_norm_sq(o: torch.Tensor) -> torch.Tensor:
    return (o ** 2).sum(dim=-1)


def oct_norm(o: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(oct_norm_sq(o).clamp_min(eps))


def oct_normalize(o: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return o / oct_norm(o, eps).unsqueeze(-1)


def oct_inverse(o: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """o⁻¹ = ō / ||o||². Для unit octonion совпадает с oct_conj."""
    n2 = oct_norm_sq(o).clamp_min(eps).unsqueeze(-1)
    return oct_conj(o) / n2


# -----------------------------------------------------------------------------
# Блочные версии: вектор размерности dim = 8K, K октонион-блоков.
# Это базовая «упаковка» в проекте (см. KellyOctonionLinear).
# -----------------------------------------------------------------------------
def _to_blocks(x: torch.Tensor) -> torch.Tensor:
    *lead, d = x.shape
    assert d % 8 == 0, f"last dim must be divisible by 8, got {d}"
    return x.view(*lead, d // 8, 8)


def _from_blocks(xb: torch.Tensor) -> torch.Tensor:
    *lead, k, eight = xb.shape
    assert eight == 8
    return xb.reshape(*lead, k * 8)


def block_oct_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Поэлементное (поблочное) octonion product на векторе dim=8K.

    bind двух «широких» кристаллов идёт **независимо** по K блокам, что даёт
    K параллельных каналов смыслов.
    """
    return _from_blocks(oct_mul(_to_blocks(a), _to_blocks(b)))


def block_oct_conj(o: torch.Tensor) -> torch.Tensor:
    return _from_blocks(oct_conj(_to_blocks(o)))


def block_oct_normalize(o: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Нормализуем КАЖДЫЙ октонион-блок отдельно к норме 1."""
    ob = _to_blocks(o)
    return _from_blocks(oct_normalize(ob, eps))


def block_oct_inverse(o: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return _from_blocks(oct_inverse(_to_blocks(o), eps))


# -----------------------------------------------------------------------------
# HRR-bind/unbind. Единые имена для скаляр-октониона и для блочного варианта.
# -----------------------------------------------------------------------------
def bind(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """k · v поблочно. Если ||k_block|| = 1, ||bind_block|| = ||v_block||."""
    return block_oct_mul(k, v)


def unbind(c: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """k̄ · c поблочно. Для unit k и одиночной пары даёт точно v.

    Для суперпозиции c = Σ k_i v_i возвращает v_q + crosstalk-noise.
    """
    return block_oct_mul(block_oct_conj(k), c)


def superpose(items: Iterable[torch.Tensor], normalize: bool = True) -> torch.Tensor:
    """Сумма списка кристаллов одинаковой формы.

    При normalize=True вектор поблочно нормализуется к норме 1 (чтобы общий
    "энергетический бюджет" не рос с N).
    """
    items = list(items)
    s = torch.stack(items, dim=0).sum(dim=0)
    if normalize:
        # Нормализуем не поблочно (это испортит линейность superposition),
        # а целиком к norm 1 — интерпретация: общий «единичный» кристалл.
        n = s.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        s = s / n
    return s


# -----------------------------------------------------------------------------
# Резонансные меры
# -----------------------------------------------------------------------------
def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """cos(a, b) по последней оси."""
    a_n = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    b_n = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (a_n * b_n).sum(dim=-1)


def per_block_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Косинус, посчитанный для каждого октонион-блока отдельно и усреднённый.

    Это другая геометрия, чем глобальный cosine: каждый блок весит одинаково,
    маленькие блоки не «тонут» в больших. Для unit-блочных кристаллов:
        per_block_cosine = (1/K) * Σ_b cos(a_b, b_b).
    """
    ab = _to_blocks(a)
    bb = _to_blocks(b)
    ab_n = ab / ab.norm(dim=-1, keepdim=True).clamp_min(eps)
    bb_n = bb / bb.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (ab_n * bb_n).sum(dim=-1).mean(dim=-1)


def resonance_score(a: torch.Tensor, b: torch.Tensor, mode: str = "cosine") -> torch.Tensor:
    """Унифицированная мера схожести.

    mode:
      "cosine"      — стандартный cosine
      "block"       — per_block_cosine (мера, согласованная с октонион-структурой)
      "neg_l2"      — −||a − b||² (классический NN)
    """
    if mode == "cosine":
        return cosine_sim(a, b)
    if mode == "block":
        return per_block_cosine(a, b)
    if mode == "neg_l2":
        return -((a - b) ** 2).sum(dim=-1)
    raise ValueError(f"unknown resonance mode: {mode}")


# -----------------------------------------------------------------------------
# Cleanup-attractor (Hopfield-like)
# -----------------------------------------------------------------------------
def nearest_in_bank(
    query: torch.Tensor,  # [..., dim]
    bank: torch.Tensor,   # [K, dim]
    mode: str = "cosine",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Возвращает (idx[...], value[..., dim]) — ближайший кристалл из банка."""
    # Расширим query для broadcasting: [..., 1, dim] vs [K, dim]
    q = query.unsqueeze(-2)
    b = bank
    if mode == "cosine":
        scores = cosine_sim(q, b)
    elif mode == "block":
        scores = per_block_cosine(q, b)
    elif mode == "neg_l2":
        scores = -((q - b) ** 2).sum(dim=-1)
    else:
        raise ValueError(f"unknown mode {mode}")
    idx = scores.argmax(dim=-1)
    val = bank[idx]
    return idx, val


def cleanup(
    query: torch.Tensor,
    bank: torch.Tensor,
    iters: int = 1,
    mode: str = "cosine",
    blend: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Итеративный cleanup: query ← blend·NN(query) + (1-blend)·query, до сходимости.

    Возвращает (final_query, idx, n_iters_used).
    """
    cur = query
    last_idx = torch.full(query.shape[:-1], -1, dtype=torch.long, device=query.device)
    used = 0
    for it in range(max(iters, 1)):
        idx, val = nearest_in_bank(cur, bank, mode=mode)
        cur = blend * val + (1.0 - blend) * cur
        used = it + 1
        if torch.equal(idx, last_idx):
            break
        last_idx = idx
    return cur, last_idx, used


# -----------------------------------------------------------------------------
# Случайные генераторы
# -----------------------------------------------------------------------------
def random_unit_octonion(*shape: int, device: str | torch.device = "cpu",
                         generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample uniformly from S^7 (unit octonions). shape=(..., 8)."""
    assert shape and shape[-1] == 8, "last dim must be 8"
    if generator is None:
        x = torch.randn(*shape, device=device)
    else:
        x = torch.randn(*shape, generator=generator, device=device)
    return oct_normalize(x)


def random_block_unit(*shape: int, device: str | torch.device = "cpu",
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Wide unit-block crystal: каждый из K октонион-блоков отдельно нормирован.

    shape: (..., dim), dim % 8 == 0.
    """
    assert shape and shape[-1] % 8 == 0
    if generator is None:
        x = torch.randn(*shape, device=device)
    else:
        x = torch.randn(*shape, generator=generator, device=device)
    return block_oct_normalize(x)


# -----------------------------------------------------------------------------
# Альтернативные bind-операторы для сравнения (E1)
# -----------------------------------------------------------------------------
def bind_circular(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Plate-style HRR via FFT circular convolution (на ℝ^dim)."""
    # k * v as circular conv: F⁻¹(F(k) ⊙ F(v))
    K = torch.fft.rfft(k, dim=-1)
    V = torch.fft.rfft(v, dim=-1)
    return torch.fft.irfft(K * V, n=k.shape[-1], dim=-1)


def unbind_circular(c: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Plate inverse: circular correlation = conv с involution(k)."""
    # involution(k)[0] = k[0], involution(k)[i] = k[N-i]
    k_inv = torch.cat([k[..., :1], k[..., 1:].flip(dims=(-1,))], dim=-1)
    return bind_circular(c, k_inv)


def bind_hadamard(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Простейший element-wise bind: k ⊙ v. Для сравнения как baseline."""
    return k * v


def unbind_hadamard(c: torch.Tensor, k: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return c / k.clamp_min(eps).where(k.abs() >= eps, k.sign() * eps + eps)


# -----------------------------------------------------------------------------
# Math sanity check (опционально, выполняется в тестах)
# -----------------------------------------------------------------------------
def _self_check() -> None:
    g = torch.Generator().manual_seed(0)
    k = random_unit_octonion(4, 8, generator=g)
    v = random_unit_octonion(4, 8, generator=g)
    # Norm preservation: ||k·v|| = 1
    c = oct_mul(k, v)
    n = oct_norm(c)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-5), n
    # Moufang/alternative: k̄(kv) = v
    rec = oct_mul(oct_conj(k), c)
    err = (rec - v).abs().max().item()
    assert err < 1e-4, f"Moufang failed, err={err}"


if __name__ == "__main__":
    _self_check()
    print("octonion_ops self-check OK")
