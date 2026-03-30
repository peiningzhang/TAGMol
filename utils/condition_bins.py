import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


PROPERTY_ALIASES = {
    "vina": ("vina_score", "vina_dock", "vina"),
    "qed": ("qed",),
    "sa": ("sa",),
}

PROPERTY_DIRECTION = {
    "vina": "lower_is_better",
    "qed": "higher_is_better",
    "sa": "higher_is_better",
}


def _has_key(data, key: str) -> bool:
    try:
        return key in data
    except Exception:
        return hasattr(data, key)


def resolve_property_field(data, property_name: str) -> Optional[str]:
    """Return the first available raw field name for a logical property."""
    for field in PROPERTY_ALIASES[property_name]:
        if _has_key(data, field):
            return field
    return None


def to_float_scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).reshape(-1)[0])
    return float(value)


def orient_value(property_name: str, value: float) -> float:
    """Map all properties to a common 'higher is better' orientation."""
    if PROPERTY_DIRECTION[property_name] == "lower_is_better":
        return -value
    return value


def extract_property_values(dataset, indices: Sequence[int], property_name: str) -> Tuple[List[float], str]:
    """Read one logical property from the raw dataset and orient it."""
    values = []
    resolved_field = None
    for idx in indices:
        data = dataset.get_ori_data(int(idx))
        field = resolve_property_field(data, property_name)
        if field is None:
            raise KeyError(
                f"Property '{property_name}' is missing for sample {idx}. "
                f"Expected one of: {PROPERTY_ALIASES[property_name]}"
            )
        if resolved_field is None:
            resolved_field = field
        raw_value = to_float_scalar(data[field])
        values.append(orient_value(property_name, raw_value))
    return values, resolved_field


def compute_equal_frequency_edges(values: Sequence[float], num_bins: int = 5) -> np.ndarray:
    """Compute bin edges using empirical quantiles.

    The returned array contains `num_bins - 1` monotonically non-decreasing
    thresholds. If duplicate quantiles occur, we nudge them slightly to keep the
    thresholds ordered so `np.searchsorted` remains stable.
    """
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute bin edges from an empty value list")

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    edges = np.quantile(arr, quantiles)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim == 0:
        edges = edges.reshape(1)
    if edges.size > 1:
        eps = np.finfo(np.float64).eps
        for i in range(1, edges.size):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], math.inf) + eps * i
    return edges


def bucketize_oriented_value(value: float, edges: Sequence[float]) -> int:
    edges_arr = np.asarray(edges, dtype=np.float64)
    return int(np.searchsorted(edges_arr, value, side="right"))


def build_condition_bin_spec(dataset, train_indices: Sequence[int], test_indices: Optional[Sequence[int]] = None,
                             num_bins: int = 5) -> Dict[str, object]:
    """Scan the dataset and build a condition bin spec from the train split."""
    spec = {
        "num_bins": int(num_bins),
        "properties": {},
        "scan": {},
    }

    for prop in ("vina", "qed", "sa"):
        train_values, train_field = extract_property_values(dataset, train_indices, prop)
        edges = compute_equal_frequency_edges(train_values, num_bins=num_bins)
        entry = {
            "field": train_field,
            "direction": PROPERTY_DIRECTION[prop],
            "edges": [float(x) for x in edges.tolist()],
            "train_count": len(train_values),
            "train_min": float(np.min(train_values)),
            "train_max": float(np.max(train_values)),
            "train_mean": float(np.mean(train_values)),
        }

        if test_indices is not None:
            test_values, test_field = extract_property_values(dataset, test_indices, prop)
            if test_field != train_field:
                entry["test_field"] = test_field
            entry.update({
                "test_count": len(test_values),
                "test_min": float(np.min(test_values)),
                "test_max": float(np.max(test_values)),
                "test_mean": float(np.mean(test_values)),
            })

        spec["properties"][prop] = entry
        spec["scan"][prop] = {
            "train_field": train_field,
            "test_field": entry.get("test_field", train_field),
        }

    return spec

