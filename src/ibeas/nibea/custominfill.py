import numpy as np

from pymoo.core.population import Population

from hldbea.evaluation import EvaluationLedger
from hldbea.local_search import select_objective, solve_epsilon_constraint
from hldbea.metrics import compute_hypervolume
from ibeas.nibea.util import Advance4Util


def _marginal_hv_gain(algorithm, objectives, candidate):
    reference = np.asarray(getattr(algorithm, "metric_ref_point", None), dtype=float)
    values = np.asarray(objectives, dtype=float)
    point = np.asarray(candidate, dtype=float).reshape(1, -1)
    if (
        values.ndim != 2
        or point.shape[1] != values.shape[1]
        or reference.shape != (values.shape[1],)
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError("local-search HV diagnostic has invalid geometry")
    ideal = getattr(algorithm, "metric_ideal_point", None)
    gain = float(
        compute_hypervolume(
            np.vstack((values, point)), reference, ideal_point=ideal
        )
        - compute_hypervolume(values, reference, ideal_point=ideal)
    )
    if gain < -1e-12:
        raise ValueError("accepted local-search candidate has negative HV gain")
    return max(0.0, gain)


def _alternative_candidates(algorithm, mode, tau):
    """Candidate-selection ablation for the deterministic track.

    ``global_rank`` draws the refinement candidates uniformly from the
    globally nondominated individuals, ``random`` from the whole population.
    """

    from hldbea.scoring import pareto_nondominated_mask

    rng = algorithm.random_state
    if mode == "global_rank":
        pool = np.flatnonzero(pareto_nondominated_mask(algorithm.pop.get("F")))
    elif mode == "random":
        pool = np.arange(len(algorithm.pop))
    else:
        raise ValueError(f"unknown candidate_mode: {mode}")
    return np.sort(rng.choice(pool, size=min(tau, len(pool)), replace=False))


def _refine_elite(algorithm, elite_pop):
    """Refine elite points under an explicit policy and FE ledger."""

    ledger = getattr(algorithm, "evaluation_ledger", None)
    if not isinstance(ledger, EvaluationLedger):
        raise ValueError("hybrid infill requires an EvaluationLedger")
    if not hasattr(algorithm, "local_search_events"):
        algorithm.local_search_events = []

    decisions = np.asarray(elite_pop.get("X"), dtype=float)
    objectives = np.asarray(elite_pop.get("F"), dtype=float)
    drop_unchanged = getattr(algorithm, "refinement_duplicates", "keep") == "drop_rejected"
    if int(getattr(algorithm, "local_search_max_iter", 2)) == 0:
        # Without local search the elite is copied unchanged into Q_nlp,
        # unless unchanged copies are dropped.
        if drop_unchanged:
            decisions = decisions[:0]
            objectives = objectives[:0]
        refined = Population.new("X", decisions.copy(), "F", objectives.copy())
        evaluated_keys = getattr(
            getattr(algorithm, "evaluator", None), "evaluate_values_of", ["F"]
        )
        refined.apply(lambda individual: individual.evaluated.update(evaluated_keys))
        return refined
    per_axis = elite_pop.get("ScoreAxis")
    generation = max(0, int(getattr(algorithm, "n_gen", 0)))
    refined_x = []
    refined_f = []
    population = getattr(algorithm, "pop", None)
    population_objectives = (
        objectives
        if population is None
        else np.asarray(population.get("F"), dtype=float)
    )

    for elite_index, (x_before, f_before) in enumerate(zip(decisions, objectives)):
        counts = None
        if per_axis is not None:
            candidate_counts = np.asarray(per_axis[elite_index])
            if candidate_counts.ndim == 1 and len(candidate_counts) == algorithm.problem.n_obj:
                counts = candidate_counts
        objective_index = select_objective(
            getattr(algorithm, "objective_policy", "round_robin"),
            n_obj=int(algorithm.problem.n_obj),
            generation=generation + elite_index,
            per_axis_counts=counts,
            rng=algorithm.random_state,
        )
        evaluation_before = ledger.used
        result = solve_epsilon_constraint(
            algorithm.problem,
            x_before,
            main_obj_index=objective_index,
            max_iter=int(getattr(algorithm, "local_search_max_iter", 2)),
            ledger=ledger,
            initial_f=f_before,
            acceptance=getattr(algorithm, "local_search_acceptance", "converged"),
        )
        evaluation_after = ledger.used
        if evaluation_after - evaluation_before != result.evaluations:
            raise ValueError("local-search event disagrees with the evaluation ledger")
        hv_gain = (
            _marginal_hv_gain(algorithm, population_objectives, result.f)
            if result.accepted
            else 0.0
        )
        gain_per_evaluation = (
            hv_gain / result.evaluations if result.evaluations else 0.0
        )
        refined_x.append(result.x)
        refined_f.append(result.f)
        algorithm.local_search_events.append(
            {
                "generation": generation,
                "elite_index": elite_index,
                "objective_index": result.objective_index,
                "success": result.success,
                "feasible": result.feasible,
                "accepted": result.accepted,
                "evaluations": result.evaluations,
                "evaluation_before": evaluation_before,
                "evaluation_after": evaluation_after,
                "hv_gain": hv_gain,
                "gain_per_evaluation": gain_per_evaluation,
                "message": result.message,
                "x_before": x_before.tolist(),
                "x_after": result.x.tolist(),
                "f_before": f_before.tolist(),
                "f_after": result.f.tolist(),
            }
        )

    if hasattr(algorithm, "evaluator") and hasattr(
        algorithm.evaluator, "sync_from_ledger"
    ):
        algorithm.evaluator.sync_from_ledger()

    refined_x = np.asarray(refined_x, dtype=float).reshape(len(refined_x), -1)
    refined_f = np.asarray(refined_f, dtype=float).reshape(len(refined_f), -1)
    if drop_unchanged:
        # A rejected call returns the parent unchanged; it produces no offspring.
        moved = np.any(refined_x != decisions[: len(refined_x)], axis=1)
        refined_x = refined_x[moved]
        refined_f = refined_f[moved]
    refined = Population.new("X", refined_x, "F", refined_f)
    evaluated_keys = getattr(
        getattr(algorithm, "evaluator", None),
        "evaluate_values_of",
        ["F"],
    )
    refined.apply(lambda individual: individual.evaluated.update(evaluated_keys))
    return refined


class CustomInfill:
    """
    Collection of custom infill strategies for generating offspring in MOEAs.

    Paper-compliant method (Algorithm 2 + Algorithm 3 of ACT_Ghani_V2-6):
        _custom_infill_paper :
            • Elite set E  = top-τ individuals by fitness (τ = algorithm.tau, default 1)
            • E undergoes ε-constraint local improvement  (Algorithm 3, k_max=2 SLSQP iters)
            • Remaining (P \\ E) generates offspring via SBX + polynomial mutation
            • Offspring = merge(E_improved, genetic_offspring)

    Other methods are retained as experimental variants.
    Les infills sont liés à l'algorithme et à la population globale...
    """

    # ------------------------------------------------------------------
    # PAPER-COMPLIANT METHOD (Algorithm 2 + Algorithm 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _custom_infill_paper(self):
        """
        Paper-compliant offspring generation (Algorithm 2, Section 3.3 of ACT_Ghani_V2-6).

        Two-track generation:
          1. Elite track (Algorithm 3):
               E = top-τ individuals (highest fitness, i.e. fitness closest to 0).
               Each individual in E is improved by the ε-constraint NLP solver
               (SLSQP, k_max=2 iterations, random objective index).
          2. Genetic track:
               The remaining N-τ individuals produce offspring via standard
               SBX crossover + polynomial mutation.

        τ is read from algorithm.tau (default 1, set in _initialize_advance).
        k_max (NLP iterations) is fixed to 2 as specified in the paper.
        """
        assert len(self.pop) == self.init_pop_size, "len(self.pop) != self.init_pop_size"

        Ff = self.pop.get("Fit")
        FfSorted, fn1 = Advance4Util.get_sorted(Ff)

        # τ: size of the elite set (paper Algorithm 3, Section 3.3)
        tau = getattr(self, 'tau', 1)
        # --- Elite set E = top-τ individuals (highest fitness = least negative) ---
        elite_idx = FfSorted[len(FfSorted) - tau:]   # last τ in sorted order = best
        rest_idx  = FfSorted[:len(FfSorted) - tau]
        candidate_mode = getattr(self, "candidate_mode", "fitness")
        if candidate_mode != "fitness":
            elite_idx = _alternative_candidates(self, candidate_mode, tau)
            rest_idx = FfSorted[~np.isin(FfSorted, elite_idx)]

        elite_pop = self.pop[elite_idx]
        rest_pop  = self.pop[rest_idx]

        # --- Track 1: ε-constraint improvement of each elite individual ---
        pb = self.problem
        off_elite = _refine_elite(self, elite_pop)

        # Track elite points for diagnostics
        if hasattr(self, 'exact_selected_tab'):
            self.exact_selected_tab.append(elite_pop.get("F"))
        if hasattr(self, 'exact_generated_tab'):
            self.exact_generated_tab.append(off_elite.get("F"))

        # --- Track 2: genetic operators on the rest ---
        n_genetic = self.init_pop_size - len(off_elite)
        n_genetic = min(n_genetic, self.evaluation_ledger.remaining)
        if len(rest_pop) > 0 and n_genetic > 0:
            off_genetic = self.mating.do(self.problem, rest_pop,
                                         n_genetic, algorithm=self, random_state=self.random_state)
        else:
            off_genetic = Population.new("X", np.zeros((0, pb.n_var)))

        # --- Merge both tracks ---
        off = Population.merge(off_elite, off_genetic)

        if len(off) == 0:
            self.termination.force_termination = True
            return

        if len(off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off
