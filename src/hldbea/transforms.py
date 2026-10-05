"""Reproducible objective-space rotations and PCA coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAResult:
    transformed: np.ndarray
    center: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray


def orthonormal_rotation(n_obj: int, seed: int) -> np.ndarray:
    """Generate a seeded proper orthonormal rotation matrix."""

    if (
        not isinstance(n_obj, (int, np.integer))
        or isinstance(n_obj, bool)
        or n_obj < 2
    ):
        raise ValueError("n_obj must be an integer of at least two")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    generator = np.random.default_rng(int(seed))
    q, r = np.linalg.qr(generator.normal(size=(n_obj, n_obj)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1.0
    return q


def rotate_objectives(
    F: np.ndarray,
    rotation: np.ndarray,
    *,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate objective rows around a declared center using row-vector convention."""

    values = np.asarray(F, dtype=float)
    matrix = np.asarray(rotation, dtype=float)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] < 2:
        raise ValueError("F must be a non-empty two-dimensional objective matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("F must contain only finite values")
    n_obj = values.shape[1]
    if matrix.shape != (n_obj, n_obj) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite square objective matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(n_obj), atol=1e-12, rtol=1e-12):
        raise ValueError("rotation must be orthonormal")
    if center is None:
        origin = np.mean(values, axis=0)
    else:
        origin = np.asarray(center, dtype=float)
        if origin.shape != (n_obj,) or not np.all(np.isfinite(origin)):
            raise ValueError("center must contain one finite value per objective")
    return (values - origin) @ matrix + origin


def pca_coordinates(F: np.ndarray) -> PCAResult:
    """Return full deterministic PCA coordinates for objective rows."""

    values = np.asarray(F, dtype=float)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] < 2:
        raise ValueError("F must be a non-empty two-dimensional objective matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("F must contain only finite values")

    center = np.mean(values, axis=0)
    centered = values - center
    n_obj = values.shape[1]
    if np.count_nonzero(centered) == 0:
        components = np.eye(n_obj)
        singular_values = np.zeros(min(values.shape), dtype=float)
        transformed = np.zeros_like(values, dtype=float)
        return PCAResult(transformed, center, components, singular_values)

    _, singular_values, components = np.linalg.svd(centered, full_matrices=True)
    for row in range(len(components)):
        pivot = int(np.argmax(np.abs(components[row])))
        if components[row, pivot] < 0:
            components[row] *= -1.0
    transformed = centered @ components.T
    return PCAResult(transformed, center, components, singular_values)
