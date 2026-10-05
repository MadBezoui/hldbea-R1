import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput

from pymoo.operators.selection.tournament import TournamentSelection # Voir: https://pymoo.org/operators/selection.html

from ibea_selection import ibea_binary_tournament
from hldbea.evaluation import EvaluationLedger, LedgerEvaluator


####################################
#### Algorithm
# Générer une population initiale
# Iterate
#   Fitness assignment
#   Environmental selection:
#       Iterate...
#   Mating selection: temporary mating pool P'.
#   Variation: recombination and mutation to P'. Merge the offspring with P.
#
#### Fin algorithm.

class IBEA(GeneticAlgorithm):

    def __init__(self,
                 ibea_advance,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=ibea_binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 output=MultiObjectiveOutput(),
                 K=1,
                 lmbda=0.1,
                 cone_epsilon=0.0,
                 neighborhood_mode="axis",
                 score_aggregation="sum",
                 selection_mode="local",
                 objective_policy="round_robin",
                 evaluation_budget=None,
                 evaluation_ledger=None,
                 metric_ideal_point=None,
                 metric_ref_point=None,
                 use_restart=False,
                 restart_theta_zero=0.95,
                 restart_window=20,
                 restart_delta_hv=1e-4,
                 restart_fraction=0.2,
                 local_search_max_iter=2,
                 local_search_acceptance="converged",
                 candidate_mode="fitness",
                 **kwargs):
        """
        Parameters
        ----------
        K     : float
            Neighbourhood radius multiplier.  Paper formula: pₐ = K·(max fₐ - min fₐ)/N.
            The paper fixes K=1 (Section 3.1); kept as a parameter for ablation.
        lmbda : float
            Weight of the distance term in the fitness function (λ in the paper, Section 3.2).
            fit(A) = -score(A) - λ · dist(A).  Default 0.1 as in the paper.
        """
        if evaluation_budget is not None and evaluation_ledger is not None:
            raise ValueError("provide evaluation_budget or evaluation_ledger, not both")
        if evaluation_ledger is not None and not isinstance(
            evaluation_ledger, EvaluationLedger
        ):
            raise TypeError("evaluation_ledger must be an EvaluationLedger")
        ledger = (
            evaluation_ledger
            if evaluation_ledger is not None
            else None if evaluation_budget is None else EvaluationLedger(evaluation_budget)
        )
        if ledger is not None:
            if kwargs.get("evaluator") is not None:
                raise ValueError("evaluation_budget cannot be combined with a custom evaluator")
            kwargs["evaluator"] = LedgerEvaluator(ledger)

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            output=output,
            **kwargs)

        self.ibea_advance = ibea_advance
        self.init_pop_size = pop_size
        self.K = K
        # λ: weight of the normalised distance-to-ideal term in the fitness (paper Section 3.2)
        self.lmbda = lmbda
        if selection_mode not in {"local", "global"}:
            raise ValueError("selection_mode must be 'local' or 'global'")
        if neighborhood_mode not in {"axis", "box", "knn"}:
            raise ValueError("unknown neighborhood_mode")
        if score_aggregation not in {"sum", "union"}:
            raise ValueError("unknown score_aggregation")
        if neighborhood_mode != "axis" and score_aggregation != "union":
            raise ValueError("box and knn neighbourhoods require union aggregation")
        if objective_policy not in {"round_robin", "adaptive", "random"}:
            raise ValueError("unknown objective_policy")
        self.cone_epsilon = float(cone_epsilon)
        self.neighborhood_mode = neighborhood_mode
        self.score_aggregation = score_aggregation
        self.selection_mode = selection_mode
        self.objective_policy = objective_policy
        self.evaluation_ledger = ledger
        self.metric_ref_point = (
            None if metric_ref_point is None else np.asarray(metric_ref_point, dtype=float)
        )
        self.metric_ideal_point = (
            None
            if metric_ideal_point is None
            else np.asarray(metric_ideal_point, dtype=float)
        )
        self.use_restart = bool(use_restart)
        self.restart_theta_zero = float(restart_theta_zero)
        self.restart_window = int(restart_window)
        self.restart_delta_hv = float(restart_delta_hv)
        self.restart_fraction = float(restart_fraction)
        self.local_search_max_iter = int(local_search_max_iter)
        self.local_search_acceptance = str(local_search_acceptance)
        if candidate_mode not in {"fitness", "global_rank", "random"}:
            raise ValueError("candidate_mode must be 'fitness', 'global_rank' or 'random'")
        self.candidate_mode = candidate_mode
        self.restart_events = []
        self.local_search_events = []
        #Save eliminated points in each iteration
        self.eliminated_tab = None
        #Save the infills to be merged with the current pop
        self.Finfills_archive = None
        #Save the merged pop + infills
        self.Fmerged_pop = None

    def setup(self, problem, **kwargs):
        result = super().setup(problem, **kwargs)
        self.random_state = np.random.default_rng(self.seed)
        return result

    def _initialize_advance(self, infills=None, **kwargs):
        if False:
            print("********* _initialize_advance (Initial fitness assignment)\nlen(infills)=", len(infills))
        self.ibea_advance._initialize_advance(self, infills=infills, **kwargs)
        if False:
            print("<<<<<<<<< End _initialize_advance\n")


    def _advance(self, infills=None, **kwargs):
        """
          In _advance (advance) we are changing algorithm.pop
        """
        self.ibea_advance._advance(self, infills=infills, **kwargs)
        #LOG: print("<<<<<<<<< End _advance")
