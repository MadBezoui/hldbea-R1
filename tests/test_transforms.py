import numpy as np
import numpy.testing as npt
import pytest


def _transform_api():
    from hldbea.transforms import orthonormal_rotation, rotate_objectives

    return orthonormal_rotation, rotate_objectives


def pca_coordinates(*args, **kwargs):
    from hldbea.transforms import pca_coordinates as implementation

    return implementation(*args, **kwargs)


def test_rotation_is_orthonormal_and_proper():
    """Catches scaled/reflected matrices masquerading as objective rotations."""
    orthonormal_rotation, _ = _transform_api()
    rotation = orthonormal_rotation(5, seed=17)

    npt.assert_allclose(rotation.T @ rotation, np.eye(5), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_rotation_seed_is_reproducible_and_discriminating():
    """Catches global RNG use or an ignored rotation seed."""
    orthonormal_rotation, _ = _transform_api()

    first = orthonormal_rotation(4, seed=3)
    repeated = orthonormal_rotation(4, seed=3)
    different = orthonormal_rotation(4, seed=4)

    npt.assert_array_equal(first, repeated)
    assert not np.allclose(first, different)


def test_rotation_round_trip_reconstructs_objectives_about_declared_center():
    """Catches row/column convention mismatches in stored benchmark rotations."""
    orthonormal_rotation, rotate_objectives = _transform_api()
    objectives = np.array([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]])
    center = np.array([0.5, 0.5, 0.5])
    rotation = orthonormal_rotation(3, seed=9)

    rotated = rotate_objectives(objectives, rotation, center=center)
    reconstructed = rotate_objectives(rotated, rotation.T, center=center)

    npt.assert_allclose(reconstructed, objectives, atol=1e-12)


def test_rotation_defaults_to_objective_centroid():
    """Catches rotation about the origin when the declared center is omitted."""
    _, rotate_objectives = _transform_api()
    objectives = np.array([[10.0, 0.0], [12.0, 2.0]])
    quarter_turn = np.array([[0.0, -1.0], [1.0, 0.0]])

    rotated = rotate_objectives(objectives, quarter_turn)

    npt.assert_allclose(np.mean(rotated, axis=0), [11.0, 1.0], atol=1e-12)


@pytest.mark.parametrize(
    ("call", "args", "kwargs"),
    [
        ("matrix", (1,), {"seed": 0}),
        ("matrix", (3,), {"seed": True}),
        ("rotate", (np.array([1.0, 2.0]), np.eye(2)), {}),
        ("rotate", (np.ones((2, 2)), np.eye(3)), {}),
        ("rotate", (np.ones((2, 2)), np.ones((2, 2))), {}),
        ("rotate", (np.ones((2, 2)), np.eye(2)), {"center": np.ones(3)}),
    ],
)
def test_rotation_rejects_invalid_dimensions_and_matrices(call, args, kwargs):
    """Catches rotations that cannot be replayed as valid orthogonal transforms."""
    orthonormal_rotation, rotate_objectives = _transform_api()
    with pytest.raises(ValueError):
        if call == "matrix":
            orthonormal_rotation(*args, **kwargs)
        else:
            rotate_objectives(*args, **kwargs)


def test_pca_coordinates_are_centered_orthonormal_and_reconstruct_input():
    """Catches a PCA transform whose stored basis cannot reproduce the coordinates."""
    objectives = np.array(
        [[0.0, 1.0, 3.0], [1.0, 3.0, 2.0], [3.0, 2.0, 0.0], [4.0, 5.0, 1.0]]
    )

    result = pca_coordinates(objectives)
    reconstructed = result.transformed @ result.components + result.center

    npt.assert_allclose(np.mean(result.transformed, axis=0), np.zeros(3), atol=1e-12)
    npt.assert_allclose(result.components @ result.components.T, np.eye(3), atol=1e-12)
    npt.assert_allclose(reconstructed, objectives, atol=1e-12)


def test_pca_component_signs_are_canonical_and_reproducible():
    """Catches run-to-run sign flips in stored PCA benchmark coordinates."""
    objectives = np.array(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0], [4.0, 8.0, 12.0]]
    )

    first = pca_coordinates(objectives)
    second = pca_coordinates(objectives)

    npt.assert_array_equal(first.components, second.components)
    npt.assert_array_equal(first.transformed, second.transformed)
    for component in first.components:
        pivot = int(np.argmax(np.abs(component)))
        assert component[pivot] >= 0.0


@pytest.mark.parametrize(
    "objectives",
    [
        np.array([[2.0, 3.0, 4.0]]),
        np.ones((4, 3)),
        np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]),
    ],
)
def test_pca_is_finite_for_singleton_constant_and_singular_data(objectives):
    """Catches NaN coordinates on populations with deficient covariance rank."""
    result = pca_coordinates(objectives)

    assert np.all(np.isfinite(result.transformed))
    assert np.all(np.isfinite(result.components))
    assert np.all(np.isfinite(result.singular_values))
    if len(objectives) == 1 or np.all(objectives == objectives[0]):
        npt.assert_array_equal(result.transformed, np.zeros_like(objectives))


@pytest.mark.parametrize(
    "objectives",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 2)),
        np.ones((2, 1)),
        np.array([[1.0, np.inf], [2.0, 3.0]]),
    ],
)
def test_pca_rejects_invalid_objective_matrices(objectives):
    """Catches malformed objective data entering rotation-sensitive experiments."""
    with pytest.raises(ValueError):
        pca_coordinates(objectives)
