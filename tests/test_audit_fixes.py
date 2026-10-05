import numpy as np
import pytest
from pymoo.core.population import Population
from pymoo.core.problem import Problem


class _Box(Problem):
    def __init__(self, n_var):
        super().__init__(n_var=n_var, n_obj=2, xl=np.zeros(n_var), xu=np.ones(n_var))


def _mutated_share(scope, n_var=12, n=10_000, seed=17):
    from hldbea.algorithm_adapters import _operators

    parameters = {
        "crossover_probability": 0.9,
        "crossover_eta": 20,
        "mutation_probability": "inverse_n_var",
        "mutation_eta": 20,
    }
    if scope is not None:
        parameters["mutation_scope"] = scope
    _, mutation = _operators(parameters, _Box(n_var))
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.05, 0.95, size=(n, n_var))
    np.random.seed(seed)
    pop = mutation.do(_Box(n_var), Population.new("X", X.copy()))
    return float(np.mean(pop.get("X") != X))


def test_per_variable_mutation_changes_one_coordinate_in_n_on_average():
    """Catches the pymoo individual gate turning 1/n into 1/n**2."""
    assert _mutated_share("per_variable") == pytest.approx(1 / 12, rel=0.1)


def test_legacy_mutation_scope_is_preserved_for_v2_reproducibility():
    assert _mutated_share(None) == pytest.approx(1 / 144, rel=0.25)


def test_unknown_mutation_scope_is_rejected():
    with pytest.raises(ValueError, match="mutation_scope"):
        _mutated_share("per_individual")


@pytest.mark.parametrize("angle", [0.0, 15.0, 30.0, 45.0, 60.0])
def test_rotated_linear_front_keeps_every_reference_point_nondominated(angle):
    from hldbea.metrics import build_reference_geometry
    from hldbea.problems import ProblemSpec
    from hldbea.scoring import pareto_nondominated_mask

    spec = ProblemSpec(
        id=f"rlinear-{angle}",
        name="rlinear",
        n_obj=3,
        n_var=12,
        parameters={"angle_degrees": angle},
        transform={"kind": "identity"},
    )
    front = np.asarray(build_reference_geometry(spec).reference_set)
    unrotated = np.asarray(
        build_reference_geometry(
            ProblemSpec(
                id="rlinear-0",
                name="rlinear",
                n_obj=3,
                n_var=12,
                parameters={"angle_degrees": 0.0},
                transform={"kind": "identity"},
            )
        ).reference_set
    )
    assert len(front) == len(unrotated)
    assert pareto_nondominated_mask(front).all()
    np.testing.assert_allclose(front.sum(axis=1), 1.0)
    # Rigid rotation: all pairwise distances are unchanged.
    def distances(points):
        return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    np.testing.assert_allclose(distances(front), distances(unrotated), atol=1e-12)


def test_rotated_linear_front_pareto_set_is_unchanged_and_off_front_is_dominated():
    from hldbea.rotated_dtlz import RotatedLinearFront

    problem = RotatedLinearFront(n_var=12, n_obj=3, angle_degrees=45.0)
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(200, 12))
    on_front = X.copy()
    on_front[:, 2:] = 0.5
    F = problem.evaluate(X, return_values_of=["F"])
    F0 = problem.evaluate(on_front, return_values_of=["F"])
    g = np.sum((X[:, 2:] - 0.5) ** 2, axis=1)
    np.testing.assert_allclose(F - F0, np.repeat(g[:, None], 3, axis=1))
    np.testing.assert_allclose(F0.sum(axis=1), 1.0)


def test_rotation_changes_orientation_with_respect_to_the_axes():
    from hldbea.rotated_dtlz import RotatedLinearFront

    vertex = np.array([[1.0, 0.0, 0.0]])
    still = RotatedLinearFront(n_var=12, n_obj=3, angle_degrees=0.0)._place(vertex)
    turned = RotatedLinearFront(n_var=12, n_obj=3, angle_degrees=60.0)._place(vertex)
    np.testing.assert_allclose(still, vertex)
    assert np.min(turned) < 0.0


def test_candidate_mode_global_rank_draws_only_nondominated_individuals():
    from ibeas.nibea.custominfill import _alternative_candidates

    class Holder:
        pass

    holder = Holder()
    F = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 3.0]])
    holder.pop = Population.new("F", F, "X", np.zeros((4, 2)))
    holder.random_state = np.random.default_rng(0)
    chosen = {int(_alternative_candidates(holder, "global_rank", 1)[0]) for _ in range(50)}
    assert chosen <= {0, 1}
    with pytest.raises(ValueError, match="candidate_mode"):
        _alternative_candidates(holder, "best", 1)


def _duplicate_ablation_spec(variant, budget):
    from dataclasses import replace
    from pathlib import Path

    from hldbea.run_spec import expand_manifest, load_manifest

    manifest = (
        Path(__file__).resolve().parents[1]
        / "experiments/manifests/reviewer-dtlz3-duplicates-v3.yaml"
    )
    spec = next(
        s for s in expand_manifest(load_manifest(manifest))
        if s.variant == variant and s.seed == 51006
    )
    return replace(spec, evaluation_budget=budget, checkpoints=(budget,))


def test_dropped_rejection_at_the_end_of_the_budget_terminates_cleanly(tmp_path):
    """Catches the crash when the last solver call exhausts the budget and is dropped."""

    import json

    from hldbea.runner import execute_run

    outcome = execute_run(_duplicate_ablation_spec("core-with-ls-drop", 437), tmp_path)
    assert outcome.status == "budget_exhausted", outcome.error_message
    events = json.loads((outcome.artifact_path / "events.json").read_text())
    last = events["local_search"][-1]
    assert last["evaluation_after"] == 437 and not last["accepted"]


def test_duplicate_elimination_leaves_no_exact_copy_among_survivors(tmp_path):
    """Catches duplicates surviving when exact copies are removed from the merged set."""

    import json

    from hldbea.runner import execute_run

    outcome = execute_run(_duplicate_ablation_spec("core-with-ls-eliminate", 2000), tmp_path)
    metadata = json.loads((outcome.artifact_path / "metadata.json").read_text())
    summary = metadata["run_metadata"]["duplicate_summary"]
    assert summary["merged_duplicates"] > 0
    assert summary["eliminated"] == summary["merged_duplicates"]
    assert summary["survivor_duplicates"] == 0
    assert summary["final_population_duplicates"] == 0
