"""Interpolation, effect metrics, and the plateau score.

These functions also run inside nnsight traces (locally or on NDIF), so they
depend only on torch and the standard library.
"""
from __future__ import annotations

import torch


def top_candidates(logits):
    """Top three raw-model candidates, with full-vocabulary probabilities.

    Rank one follows argmax's tie breaking, exactly as greedy decoding does.
    Only the tiny selected tensors need to leave an NDIF trace.
    """
    logits = logits.float()
    first = logits.argmax(-1, keepdim=True)
    rest = logits.scatter(-1, first, float("-inf")).topk(min(2, logits.shape[-1] - 1), dim=-1).indices
    ids = torch.cat((first, rest), dim=-1)
    return ids, logits.softmax(-1).gather(-1, ids)


def interpolate(a, b, t, method="slerp"):
    """SLERP directions, linearly interpolated norms (Matthew Shinkle, footnote 1)."""
    t = torch.as_tensor(t, device=a.device, dtype=a.dtype).reshape(-1, 1)
    if method == "linear":
        return (1 - t) * a + t * b
    na, nb = a.norm(), b.norm()
    if min(na.item(), nb.item()) < 1e-10:
        return (1 - t) * a + t * b
    ua, ub = a / na, b / nb
    cos = torch.dot(ua, ub).clamp(-1, 1)
    if cos.item() < -0.9995:
        raise ValueError("The endpoint directions are nearly opposite, so the spherical path is ambiguous. Choose Linear.")
    if cos.item() > 0.9995:
        direction = (1 - t) * ua + t * ub
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    else:
        theta = torch.acos(cos)
        direction = (torch.sin((1 - t) * theta) * ua + torch.sin(t * theta) * ub) / torch.sin(theta)
    result = direction * ((1 - t) * na + t * nb)
    # Preserve exact endpoints and make endpoint checks meaningful.
    result = torch.where(t == 0, a, result)
    return torch.where(t == 1, b, result)


def relative_distance(x, a, b):
    if torch.linalg.vector_norm(a - b).item() < 1e-8:
        raise ValueError("The endpoint outputs are identical. Their relative distance is undefined, so this curve cannot be computed.")
    da = torch.linalg.vector_norm(x - a, dim=-1)
    db = torch.linalg.vector_norm(x - b, dim=-1)
    return da / (da + db)


def interpolate_tokens(a, b, ts, method="slerp"):
    """Interpolate corresponding token vectors independently, with a shared t."""
    if a.shape != b.shape or a.ndim != 2 or len(a) == 0:
        raise ValueError("Interpolation requires matching, nonempty token spans.")
    if len(a) == 1:
        return interpolate(a[0], b[0], ts, method).unsqueeze(1)
    t = torch.as_tensor(ts, device=a.device, dtype=a.dtype).reshape(-1, 1, 1)
    linear = (1 - t) * a + t * b
    if method == "linear":
        return linear
    na, nb = a.norm(dim=-1, keepdim=True), b.norm(dim=-1, keepdim=True)
    nonzero = (na >= 1e-10) & (nb >= 1e-10)
    ua, ub = a / na.clamp_min(1e-10), b / nb.clamp_min(1e-10)
    cos = (ua * ub).sum(dim=-1, keepdim=True).clamp(-1, 1)
    if ((cos < -0.9995) & nonzero).any().item():
        raise ValueError("A pair of token-state directions is nearly opposite. Choose Linear interpolation.")
    direction_linear = (1 - t) * ua + t * ub
    direction_linear = direction_linear / direction_linear.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    theta = torch.acos(cos)
    direction_spherical = (torch.sin((1 - t) * theta) * ua + torch.sin(t * theta) * ub) / torch.sin(theta).clamp_min(1e-12)
    direction = torch.where(cos > 0.9995, direction_linear, direction_spherical)
    result = torch.where(nonzero, direction * ((1 - t) * na + t * nb), linear)
    return torch.where(t == 0, a, torch.where(t == 1, b, result))


# Effect metrics compare path samples x [n, dim] with the endpoints a, b [dim] (the outputs at
# t = 0 and t = 1). Each is 0 at a and 1 at b. The caller checks ||a-b|| before using the values.
def relative_l2_shinkle(x, a, b):
    """d = ||x-a|| / (||x-a|| + ||x-b||)  (Shinkle & Heimersheim 2025)."""
    da = torch.linalg.vector_norm(x - a, dim=-1)
    db = torch.linalg.vector_norm(x - b, dim=-1)
    return da / (da + db)


def relative_l2_janiak(x, a, b):
    """d = ||x-a|| / ||a-b||  (Janiak et al. 2024). Can exceed 1."""
    return torch.linalg.vector_norm(x - a, dim=-1) / torch.linalg.vector_norm(a - b)


def relative_l2_projected(x, a, b):
    """d = (x-a)·(b-a) / ||b-a||²: position along the straight line from a to b.

    Only the component parallel to b-a counts; the perpendicular part is ignored.
    Can leave [0, 1] when x overshoots either endpoint.
    """
    direction = b - a
    return ((x - a) * direction).sum(dim=-1) / (direction * direction).sum()


# Registry, by id; the first entry is the default. All metrics are computed for every recorded layer
# inside the path trace. "range" is the y-range the plot always shows (extended to fit the data);
# "plateau_score" says whether the 0.1 -> 0.9 transition width applies.
METRICS = {
    "relative_l2_shinkle": {"label": "Relative L2 (Shinkle & Heimersheim 2025)", "fn": relative_l2_shinkle,
                            "range": [0, 1], "plateau_score": True},
    "relative_l2_janiak": {"label": "Relative L2 (Janiak et al. 2024)", "fn": relative_l2_janiak,
                           "range": [0, 1], "plateau_score": True},
    "relative_l2_projected": {"label": "Relative L2, projected onto A→B", "fn": relative_l2_projected,
                              "range": [0, 1], "plateau_score": True},
}
DEFAULT_METRIC = next(iter(METRICS))


def effect_metrics(x, a, b):
    """Every registered metric for samples x, plus the endpoint gap ||a-b||; runs inside traces."""
    return {name: metric["fn"](x, a, b) for name, metric in METRICS.items()}, torch.linalg.vector_norm(a - b)


def arc_segments(x, previous=None):
    """Real samples only; keep one float64 vector across trace/batch boundaries.

    Transfer before subtraction: MPS has no float64, and casting a float32
    difference afterwards loses information. No endpoint-reference rows belong
    in this sequence. Return the last vector separately, never the whole path.
    """
    current = x.detach().cpu().to(dtype=torch.float64)
    first = torch.zeros(1, dtype=torch.float64) if previous is None else torch.linalg.vector_norm(
        current[0] - previous.detach().cpu().to(dtype=torch.float64)).reshape(1)
    lengths = torch.cat([first, torch.linalg.vector_norm(current[1:] - current[:-1], dim=-1)])
    # A singleton nonfinite first sample must invalidate the arc as well.
    lengths = torch.where(torch.isfinite(current).all(dim=-1), lengths, float("nan"))
    return lengths, current[-1].clone()


def transition_width(ts, values, low=0.1, high=0.9):
    """Plateau score: t needed to go from the first crossing of `low` to the first crossing of `high`.

    Crossings are linearly interpolated between samples. None if either level is never reached.
    """
    def crossing(level):
        for i, value in enumerate(values):
            if value >= level:
                if i == 0:
                    return ts[0]
                t0, t1, v0 = ts[i - 1], ts[i], values[i - 1]
                return t0 + (level - v0) / (value - v0) * (t1 - t0)
        return None
    start, end = crossing(low), crossing(high)
    return None if start is None or end is None else end - start
