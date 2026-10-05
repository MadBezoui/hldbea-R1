import types

import numpy as np

from pymoo.core.population import Population
from pymoo.operators.selection.tournament import TournamentSelection

from hldbea.evaluation import EvaluationLedger
from hldbea.metrics import compute_hypervolume
from hldbea.restart import RestartPolicy, RestartState, update_restart_state
from hldbea.selection import select_survivor_indices
from ibeas.nibea.custominfill import CustomInfill
from ibeas.nibea.temppopmethod import TempPopMethod
from ibeas.nibea.selectpopmethod import SelectPopMethod
from ibeas.nibea.envselectionmethod import EnvSelectionMethod
from ibeas.nibea.util import Advance4Util


# ---------------------------------------------------------------------------
# Stagnation detection (paper Section 3.4)
# ---------------------------------------------------------------------------
# Parameters (from the paper):
#   theta_restart = 0.95   : fraction of zero-score individuals that triggers the check
#   G_stag        = 20     : number of consecutive generations to monitor HV
#   delta_HV      = 1e-4   : minimum HV improvement to NOT consider stagnation
#   rho           = 0.20   : fraction of the population to restart randomly
# ---------------------------------------------------------------------------
_THETA_RESTART = 0.95
_G_STAG        = 20
_DELTA_HV      = 1e-4
_RHO           = 0.20


def _restart_event(algorithm, **values):
    if not hasattr(algorithm, "restart_events"):
        algorithm.restart_events = []
    ledger = getattr(algorithm, "evaluation_ledger", None)
    evaluation = (
        int(ledger.used)
        if isinstance(ledger, EvaluationLedger)
        else int(getattr(getattr(algorithm, "evaluator", None), "n_eval", 0))
    )
    event = {
        "generation": int(getattr(algorithm, "n_gen", 0)),
        "evaluation": evaluation,
        "evaluation_after": evaluation,
        "hv": None,
        "zero_fraction": None,
        "replacement_count": 0,
        **values,
    }
    algorithm.restart_events.append(event)
    return event


def _check_and_apply_restart(algorithm):
    """Apply the explicit raw-score/HV restart policy and record every check."""

    ref_point = getattr(algorithm, "metric_ref_point", None)
    objectives = np.asarray(algorithm.pop.get("F"), dtype=float)
    if ref_point is None:
        _restart_event(algorithm, status="skipped_missing_reference")
        return
    ref_point = np.asarray(ref_point, dtype=float)
    if (
        ref_point.shape != (objectives.shape[1],)
        or not np.all(np.isfinite(ref_point))
    ):
        _restart_event(algorithm, status="skipped_invalid_reference")
        return

    try:
        hv_current = compute_hypervolume(
            objectives,
            ref_point,
            ideal_point=getattr(algorithm, "metric_ideal_point", None),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        _restart_event(
            algorithm,
            status="skipped_metric_error",
            message=f"{type(exc).__name__}: {exc}",
        )
        return

    if not hasattr(algorithm, "_restart_state"):
        algorithm._restart_state = RestartState()
    policy = RestartPolicy(
        theta_zero=float(getattr(algorithm, "restart_theta_zero", _THETA_RESTART)),
        window=int(getattr(algorithm, "restart_window", _G_STAG)),
        delta_hv=float(getattr(algorithm, "restart_delta_hv", _DELTA_HV)),
        fraction=float(getattr(algorithm, "restart_fraction", _RHO)),
    )
    decision = update_restart_state(
        algorithm._restart_state,
        policy,
        raw_scores=algorithm.pop.get("ScoreRaw"),
        fitness=algorithm.pop.get("Fit"),
        hv=hv_current,
        generation=int(getattr(algorithm, "n_gen", 0)),
    )
    event = _restart_event(
        algorithm,
        status="triggered" if decision.triggered else "observed",
        hv=hv_current,
        zero_fraction=decision.zero_fraction,
        hv_delta=decision.hv_delta,
        replace_indices=decision.replace_indices.tolist(),
        requested_replacement_count=len(decision.replace_indices),
    )
    if not decision.triggered:
        return

    ledger = getattr(algorithm, "evaluation_ledger", None)
    required = len(decision.replace_indices)
    if ledger is not None and ledger.remaining < required:
        event["status"] = "skipped_budget"
        event["required_evaluations"] = required
        event["remaining_evaluations"] = ledger.remaining
        return

    from pymoo.operators.sampling.rnd import FloatRandomSampling

    new_pop = FloatRandomSampling()(
        algorithm.problem,
        len(decision.replace_indices),
        random_state=algorithm.random_state,
    )
    algorithm.evaluator.eval(algorithm.problem, new_pop, algorithm=algorithm)
    keep = np.ones(len(algorithm.pop), dtype=bool)
    keep[decision.replace_indices] = False
    algorithm.pop = Population.merge(algorithm.pop[keep], new_pop)

    # Recompute all state on the post-restart population; random points must not
    # inherit privileged zero fitness from the legacy implementation.
    fitness = Advance4Util.calc_scores(algorithm, pop=algorithm.pop)
    algorithm.pop.set("Fit", fitness)
    algorithm.pop.set("FitMin", fitness.copy())
    algorithm.pop.set("FitLife", np.zeros(len(algorithm.pop)))
    event["population_size_after"] = len(algorithm.pop)
    event["replacement_count"] = required
    event["evaluation_after"] = int(ledger.used)


def _apply_environmental_selection(algorithm):
    """Select exactly N survivors from raw scores or the global ablation."""

    objectives = np.asarray(algorithm.pop.get("F"), dtype=float)
    indices = select_survivor_indices(
        objectives,
        algorithm.pop.get("ScoreRaw"),
        int(algorithm.init_pop_size),
        mode=getattr(algorithm, "selection_mode", "local"),
    )
    keep = np.zeros(len(algorithm.pop), dtype=bool)
    keep[indices] = True
    algorithm.eliminated_tab = objectives[~keep].copy()
    algorithm.pop = algorithm.pop[indices]


class Advance4Base:
    """
    Base class with core logic for population fitness updates and selection initialization.
    Provides common functionality for fitness assignment, population merging, 
    and selection method dispatch in MOEAs.
    """

    @staticmethod
    def selection(**kwargs):
        from ibea_selection import hldbea_binary_tournament
        print('...Using custom selection')
        return TournamentSelection(func_comp=hldbea_binary_tournament)

    @classmethod
    def _initialize_advance(cls, algorithm, infills=None, **kwargs):
        """
        Sets paper-compliant defaults for all HLDBEA hyper-parameters and performs
        the initial fitness assignment.

        Paper defaults (ACT_Ghani_V2-6, Section 3):
            sc_method      = "0"      → fit(A) = -score(A) - λ·dist(A)
            env_sel_method = "crowd"  → Rank & Crowding as tie-breaker (Algorithm 4)
            sel_pop        = "2"      → iterative fitness recalculation
            temp_pop       = "0"      → standard merge P ∪ P'
            exact_method   = "epsilon"→ ε-constraint for exact infill (Algorithm 3)
            tau            = 1        → size of elite set for exact infill (Algorithm 3)
            lmbda          = 0.1      → distance weight in fitness (already set in IBEA.__init__)
        """
        # sc_method: "0" is paper-compliant (augmented fitness)
        if not hasattr(algorithm, "fitm"):
            print('Attention: Default FitMin to True')
            algorithm.fitm = True
        if not hasattr(algorithm, "sc_method"):
            print('Attention: Default Score method → "0" (paper-compliant)')
            algorithm.sc_method = "0"
        if not hasattr(algorithm, "env_sel_method"):
            print('Attention: Default ENV Selection → "crowd" (paper: Rank & Crowding)')
            algorithm.env_sel_method = "crowd"
        if not hasattr(algorithm, "sel_pop"):
            print('Attention: Default Sel pop → "2"')
            algorithm.sel_pop = "2"
        if not hasattr(algorithm, "temp_pop"):
            print('Attention: Default Temp pop → "0"')
            algorithm.temp_pop = "0"
        if not hasattr(algorithm, "exact_method"):
            print('Attention: Default Exact method → "epsilon"')
            algorithm.exact_method = "epsilon"
        # tau: number of elite individuals to apply exact infill (Algorithm 3 of the paper)
        if not hasattr(algorithm, "tau"):
            algorithm.tau = 1
        # lmbda: may have been set in IBEA.__init__; ensure it exists here too
        if not hasattr(algorithm, "lmbda"):
            algorithm.lmbda = 0.1

        print(f"...Using adaptive_k: {hasattr(algorithm, 'adaptive_k')}")
        print(f"...Using FitMin: {algorithm.fitm}")
        print(f"...Using TempPopMethod: {algorithm.temp_pop}")
        print(f"...Using SelectPop: {algorithm.sel_pop}")
        print(f"...Using EnvSelectionMethod: {algorithm.env_sel_method}")
        print(f"...Using sc_method: {algorithm.sc_method}")
        print(f"...Using exact_method: {algorithm.exact_method}")
        print(f"...Using tau (elite set size): {algorithm.tau}")
        print(f"...Using lambda (distance weight): {algorithm.lmbda}")
        print(f"...Using restart mechanism: {algorithm.use_restart}")

        # Compute and set Fitness
        _F = algorithm.pop.get("F")
        Ff = Advance4Util.calc_scores(algorithm, _F)
        algorithm.pop.set("Fit", Ff)

        #algorithm.pop.set("FitCumul", Ff)
        if algorithm.fitm:
            algorithm.pop.set("FitMin", Ff)
        #TODO: not used yet 11/06/2025
        algorithm.pop.set("FitLife", np.zeros(len(algorithm.pop)))

        # Compter le nombre de fois d'appel de sel_pop (et déduire par la suite env_sel)
        algorithm.count_sel_pop = 0

    @staticmethod
    def _temp_pop(algorithm, infills, **kwargs):
        if algorithm.temp_pop == '2':
            TempPopMethod._temp_pop_2(algorithm, infills, **kwargs)
        elif algorithm.temp_pop == '1':
            TempPopMethod._temp_pop_1(algorithm, infills, **kwargs)
        else : # '0', ...
            TempPopMethod._temp_pop_0(algorithm, infills, **kwargs)

    @classmethod #Not used yet. Ne doit pas avoir pasx et pasy comme params...
    def calc_fitness(cls, algorithm, _F):
        return Advance4Util.calc_scores(algorithm, _F)

    @staticmethod
    def do_sel_pop(algorithm, _F, FfSorted, fn1):
        sel_pop = algorithm.sel_pop
        if  (sel_pop == "0") or (fn1 == algorithm.init_pop_size):
            SelectPopMethod.sel_pop0(algorithm, _F, FfSorted)
        elif sel_pop == "1":
            SelectPopMethod.sel_pop1(algorithm, FfSorted, fn1)
        elif sel_pop == "3": #Remove worst
            SelectPopMethod.sel_pop3(algorithm, FfSorted, fn1)
        elif sel_pop == "4": #select best and continue with ranking
            SelectPopMethod.sel_pop4(algorithm, FfSorted, fn1)
        else: #sel_pop == "2": #select best
            SelectPopMethod.sel_pop2(algorithm, FfSorted, fn1)

    @staticmethod
    def do_env_selection(algorithm, _F, FfSorted, fn1):
        """
        Dispatches to the appropriate environmental selection method.

        Paper (Algorithm 4): when more than N individuals have fitness=0,
        use NSGA-II Rank & Crowding ("crowd") as the tie-breaker.
        """
        env_sel_method = algorithm.env_sel_method
        if env_sel_method == "0":
            EnvSelectionMethod.method0(algorithm, _F, FfSorted, fn1)
        elif env_sel_method == "1":
            EnvSelectionMethod.method1(algorithm, _F, FfSorted, fn1)
        elif env_sel_method == "2":
            EnvSelectionMethod.method2(algorithm, _F, FfSorted, fn1)
        elif env_sel_method == "rand":
            EnvSelectionMethod.method_rand(algorithm, _F, FfSorted, fn1) # -
        elif env_sel_method == "crowd":
            # Paper-compliant: Rank & Crowding tie-breaker (Algorithm 4)
            EnvSelectionMethod.method_crowd(algorithm, _F, FfSorted, fn1)
        elif env_sel_method == "exact":
            EnvSelectionExactMethod.method_exact(algorithm, _F, FfSorted, fn1)
        elif env_sel_method == "rank":
            EnvSelectionMethod.method_rank(algorithm, _F, FfSorted, fn1)
        elif env_sel_method == "inv":
            EnvSelectionMethod.method_inv(algorithm, _F, FfSorted, fn1)
        else: # default: paper-compliant crowd
            EnvSelectionMethod.method_crowd(algorithm, _F, FfSorted, fn1)


class Advance4Common(Advance4Base):

    @classmethod
    def update_population_fitness(cls, algorithm):
        Ff = Advance4Util.calc_scores(algorithm)
        #algorithm.pop.set("Fit", Ff)
        #
        #TODO: idea "id-1"
        #algorithm.pop.set("FitCumul", ...)
        #algorithm.pop.set("FitLife", ...)
        if algorithm.fitm:
            fmin = algorithm.pop.get("FitMin") # TODO: exist pr t lé pts ?
            algorithm.pop.set("FitMin", np.minimum(fmin, Ff))
            algorithm.pop.set("Fit", np.minimum(fmin, Ff))
        else:
            algorithm.pop.set("Fit", Ff)

        fl = algorithm.pop.get("FitLife") + 1
        algorithm.pop.set("FitLife", fl)
        #print(f"fl: ({algorithm.n_iter})", np.sort(fl))

        #LOG
        #fit_arr = algorithm.pop.get("Fit")
        #print("Fitness: ", fit_arr)
        #unique_elements, counts = np.unique(fit_arr, return_counts=True)
        #print("Unique elements:", unique_elements)
        #print("Counts:", counts)


    @classmethod
    def _advance(cls, algorithm, infills=None, **kwargs):
        '''
        Main iteration step of HLDBEA (Algorithm 2 of the paper).

        Steps:
          1. Merge offspring (infills) with current population P → P̃ = P ∪ P'
          2. Compute fitness for P̃: fit(A) = -score(A) - λ·dist(A)
          3. Environmental selection: keep N best individuals
             a. If |{A ∈ P̃ : fit(A)=0}| < N  → call sel_pop (fills up from next-best)
             b. If |{A ∈ P̃ : fit(A)=0}| > N  → call env_selection (trim with Rank & Crowding)
          4. [Optional] Stagnation detection + restart (paper Section 3.4)

        algorithm.pop : current population P (before the merge)
        infills       : offspring P'
        '''

        # the current population; backup copy.
        pop = algorithm.pop
        # Bug-fix (line 181 original): was 'fills' (NameError), must be 'infills'
        if algorithm.fitm:
            infills.set("FitMin", np.zeros(len(infills)))
        infills.set("FitLife", np.zeros(len(infills)))

        # Add infills to history
        algorithm.Finfills_archive = infills.get("F")

        # Intermediate pop: merge (P + infills) — Step 1
        cls._temp_pop(algorithm, infills, tech='2')
        #Now algorithm.pop = merge (P + infills)

        _F = algorithm.pop.get("F") # After the merge (see pop for the original) 
        # Step 2: Calculate fit(A) = -score(A) - λ·dist(A) for the merged population
        Ff = Advance4Util.calc_scores(algorithm, _F)

        # idea "id-1": keep track of historical minimum fitness
        if algorithm.fitm:
            Ff = np.minimum(Ff, algorithm.pop.get("FitMin"))

        if algorithm.fitm:
            algorithm.pop.set("FitMin", Ff)
        algorithm.pop.set("Fit", Ff)



        ###### SURVIVAL / ENVIRONMENTAL SELECTION (Algorithm 4 of the paper)
        # Goal: select N individuals from P̃ = P ∪ P' (size 2N).
        #   • All individuals with fit=0 (locally non-dominated in P̃) are preferred.
        #   • If their count < N: complete with the next-best (sel_pop).
        #   • If their count > N: trim using Rank & Crowding (env_sel, paper Section 3.3).

        _apply_environmental_selection(algorithm)

        # Step 4: Stagnation detection + restart (paper Section 3.4)
        if getattr(algorithm, "use_restart", True):
            _check_and_apply_restart(algorithm)

        #TODO Inutile peut être (?). Les individus ont déjà 'fit' calculée à partir de la pop augmentée:
        # doit-on recalculer avec la new pop ou laisser les anciens?
        #Pour l'instant, ne pas recalculer
        if False:
            #fit est utilisée dans la génération d'un offspring, la recalculer ou pas l'influence donc!
            cls.update_population_fitness(algorithm)

        ##idea: update K with n_iter
        if hasattr(algorithm, "adaptive_k"):
        #if algorithm._adaptive_k:
            #algorithm.K = algorithm.K + algorithm.termination.perc
            algorithm.K = algorithm.K + (algorithm.count_sel_pop - algorithm.n_iter/2)/(algorithm.count_sel_pop+1)

        # LOG: some stats for debugging
        if False:
            #LOG: HV
            from pymoo.indicators.hv import HV
            indh = HV(ref_point=algorithm.problem.nadir_point())
            hv = indh(algorithm.pop.get('F'))
            #print(f"NIBEA: gen = {algorithm.n_gen} Hv = {hv:.{5}f}")

            from pymoo.indicators.igd import IGD
            pf = algorithm.problem.pareto_front()
            indigd = IGD(pf)
            igd = indigd(algorithm.pop.get('F'))
            #print(f"NIBEA: gen = {algorithm.n_gen} IGD = {igd:.{5}f}")

            print(f"NIBEA: gen = {algorithm.n_gen} Hv = {hv:.{5}f} IGD = {igd:.{5}f}")


class AdvanceCustomInfill(Advance4Common):
    """
    Extends HLDBEA with a custom offspring generation strategy.

    The paper (Algorithm 2 + Algorithm 3) describes two tracks:
      • Elite track  : top-τ individuals undergo ε-constraint local improvement (Algorithm 3)
      • Genetic track: remaining individuals undergo SBX + polynomial mutation

    This is activated by setting inf='1a' (τ individuals from custom_infill_1a).
    """

    @staticmethod
    def _infill(algorithm, **kwargs):

        #return # No custom infill
        if not hasattr(algorithm, "inf"):
            algorithm.infill_name = ""
            return

        inf = getattr(algorithm, "inf")
        if inf is None:
            algorithm.infill_name = ""
            return

        print(f"...Using custom infills: {inf}")

        # Contient les points générés par la méthode exacte si utilisée.
        algorithm.exact_generated_tab = []
        # Contient les points sélectionnés pour la méthode exacte (ensemble E)
        algorithm.exact_selected_tab = []

        if inf == 'hybrid':
            # PAPER-COMPLIANT: elite τ individuals → ε-constraint (Algorithm 3)
            algorithm._infill = types.MethodType(CustomInfill._custom_infill_paper, algorithm)
            algorithm.infill_name = "hybrid"
        else:
            print("WARNING: inf value not valid. No custom infill will be used.")



#class Advance(Advance4Common): #Méthode de base
class Advance(AdvanceCustomInfill): #Définit un autre _infill
    @classmethod
    def adv_name(cls):
        chain = ''
        for base in cls.__mro__:
            chain += f"{base.__name__}"
        #return chain
        return cls.__mro__[0].__name__ + cls.__mro__[1].__name__
