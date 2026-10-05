import numpy as np
import numpy.testing as npt


def test_frequency_decode_matches_declared_cosine_series():
    from hldbea.recent_algorithms import frequency_decode

    parameters = np.array(
        [
            [0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.50, np.pi / 2.0, np.pi, 0.4],
        ]
    )
    decoded = frequency_decode(parameters, n_var=4)

    expected = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [
                0.2
                + 0.5 * np.cos(np.pi * j + np.pi / 2.0)
                + np.pi * np.cos(2.0 * np.pi * j + 0.4)
                for j in range(1, 5)
            ],
        ]
    )
    npt.assert_allclose(decoded, expected, atol=1e-12)


def test_maoea_angle_indicator_is_finite_and_uses_nearest_objective_angles():
    from hldbea.recent_algorithms import associate_angle

    objectives = np.eye(3)
    indicator = associate_angle(objectives)

    assert indicator.shape == (3,)
    assert np.all(np.isfinite(indicator))
    npt.assert_allclose(
        indicator,
        np.full(3, (np.pi / 2) * (1.0 + 1.0 / 2.0 + 1.0 / 3.0)),
    )


def test_shape_estimate_selects_one_of_the_reference_exponents():
    from hldbea.recent_algorithms import estimate_shape_exponent

    objectives = np.array(
        [
            [0.0, 1.0, 1.0],
            [0.3, 0.8, 0.9],
            [0.7, 0.6, 0.7],
            [1.0, 0.0, 1.0],
        ]
    )
    exponent = estimate_shape_exponent(objectives)

    assert exponent in np.round(np.arange(0.5, 2.01, 0.1), 10)


def test_recent_algorithm_provenance_is_pinned_to_audited_platemo_commit():
    from hldbea.registry import available_algorithms

    descriptors = available_algorithms()
    for algorithm_id in ("maoea_hap", "fdsea"):
        descriptor = descriptors[algorithm_id]
        assert descriptor.source == "PlatEMO"
        assert descriptor.version == "d25e65d1ffba58dbf4d7e1b5259786187d12968a"
        assert descriptor.citation_key
