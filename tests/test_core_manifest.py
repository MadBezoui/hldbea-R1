from pathlib import Path

import yaml


MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "manifests"
    / "core-smoke.yaml"
)


def _load_manifest():
    assert MANIFEST.is_file(), f"missing core smoke manifest: {MANIFEST}"
    with MANIFEST.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_core_manifest_covers_declared_correctness_matrix():
    document = _load_manifest()
    cases = document["cases"]

    assert document["schema_version"] == 1
    assert document["claim_scope"] == "correctness_smoke_only"
    assert {case["n_obj"] for case in cases} == {2, 3, 5, 8, 10, 15}
    assert {case["scoring"] for case in cases} == {"strict", "cone"}
    assert {case["selection_mode"] for case in cases} == {"local", "global"}
    assert {case["restart"] for case in cases} == {False, True}
    assert {case["objective_policy"] for case in cases} == {
        "round_robin",
        "adaptive",
    }


def test_core_manifest_case_ids_and_seeds_are_fixed_and_unique():
    cases = _load_manifest()["cases"]
    identifiers = [case["id"] for case in cases]
    seeds = [case["seed"] for case in cases]

    assert len(identifiers) == len(set(identifiers))
    assert all(isinstance(identifier, str) and identifier for identifier in identifiers)
    assert len(seeds) == len(set(seeds))
    assert all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
    assert all(case["evaluation_budget"] >= case["population_size"] for case in cases)


def test_core_manifest_records_reproducible_environment_and_commands():
    document = _load_manifest()
    environment = document["environment"]

    assert environment == {
        "python": "3.10.1",
        "numpy": "1.26.4",
        "scipy": "1.14.1",
        "pandas": "2.3.3",
        "pymoo": "0.6.1.3",
        "matplotlib": "3.10.7",
        "pyyaml": "6.0.2",
        "pytest": "8.4.1",
    }
    assert document["verification"]["test_command"] == (
        "MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 "
        ".venv/bin/python -m pytest -q"
    )
    assert document["verification"]["performance_claims"] is False
