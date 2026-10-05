from copy import deepcopy
from pathlib import Path

import pytest


MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "manifests"
    / "integration-smoke.yaml"
)


def _api():
    from hldbea.run_spec import RunSpec, expand_manifest, load_manifest

    return RunSpec, expand_manifest, load_manifest


def _minimal_document():
    return {
        "schema_version": 1,
        "manifest_id": "unit-matrix",
        "stage": "smoke",
        "seeds": [7, 11],
        "problems": [
            {
                "id": "dtlz2-m3",
                "name": "dtlz2",
                "n_obj": 3,
                "n_var": 12,
                "parameters": {},
                "transform": {"kind": "identity"},
            }
        ],
        "algorithms": [
            {"id": "hldbea", "variant": "full", "parameters": {"k": 1.0}}
        ],
        "execution": {
            "population_size": 10,
            "evaluation_budget": 40,
            "checkpoints": [10, 20, 40],
            "save_history": False,
        },
    }


def test_integration_manifest_expands_complete_shared_seed_matrix():
    _, expand_manifest, load_manifest = _api()
    document = load_manifest(MANIFEST)
    specs = expand_manifest(document)

    assert len(specs) == 16
    assert {spec.problem_name for spec in specs} == {"dtlz2", "wfg2"}
    assert {spec.n_obj for spec in specs} == {3, 5}
    assert {spec.algorithm_id for spec in specs} == {"hldbea", "nsga2"}
    assert {spec.seed for spec in specs} == {31001, 31002}
    assert len({spec.run_id for spec in specs}) == len(specs)

    combinations = {
        (spec.problem_id, spec.algorithm_id): set() for spec in specs
    }
    for spec in specs:
        combinations[(spec.problem_id, spec.algorithm_id)].add(spec.seed)
    assert all(seeds == {31001, 31002} for seeds in combinations.values())


def test_run_hash_is_independent_of_mapping_key_order():
    RunSpec, expand_manifest, _ = _api()
    document = _minimal_document()
    first = expand_manifest(document)[0]

    reordered = {
        "execution": {
            "save_history": False,
            "checkpoints": [10, 20, 40],
            "evaluation_budget": 40,
            "population_size": 10,
        },
        "algorithms": [
            {"parameters": {"k": 1.0}, "variant": "full", "id": "hldbea"}
        ],
        "problems": [
            {
                "transform": {"kind": "identity"},
                "parameters": {},
                "n_var": 12,
                "n_obj": 3,
                "name": "dtlz2",
                "id": "dtlz2-m3",
            }
        ],
        "seeds": [7, 11],
        "stage": "smoke",
        "manifest_id": "unit-matrix",
        "schema_version": 1,
    }
    second = expand_manifest(reordered)[0]

    assert isinstance(first, RunSpec)
    assert first.canonical_dict() == second.canonical_dict()
    assert first.config_hash == second.config_hash
    assert first.run_id == second.run_id


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("mystery",), 1),
        (("execution", "generations"), 2),
        (("algorithms", 0, "citation"), "hidden"),
        (("problems", 0, "reference_point"), [1.1, 1.1, 1.1]),
    ],
)
def test_manifest_rejects_unknown_fields(path, value):
    _, expand_manifest, _ = _api()
    document = _minimal_document()
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="unknown"):
        expand_manifest(document)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc.update(seeds=[7, 7]),
        lambda doc: doc["execution"].update(population_size=41),
        lambda doc: doc["execution"].update(checkpoints=[10, 10, 40]),
        lambda doc: doc["execution"].update(checkpoints=[10, 20, 39]),
        lambda doc: doc["problems"][0].update(n_obj=1),
        lambda doc: doc["problems"][0].update(n_var=True),
    ],
)
def test_manifest_rejects_invalid_scientific_contract(mutation):
    _, expand_manifest, _ = _api()
    document = deepcopy(_minimal_document())
    mutation(document)

    with pytest.raises(ValueError):
        expand_manifest(document)


def test_run_spec_is_deeply_isolated_from_manifest_mutation():
    _, expand_manifest, _ = _api()
    document = _minimal_document()
    spec = expand_manifest(document)[0]
    before = spec.canonical_dict()

    document["algorithms"][0]["parameters"]["k"] = 99
    document["problems"][0]["transform"]["kind"] = "rotation"

    assert spec.canonical_dict() == before
