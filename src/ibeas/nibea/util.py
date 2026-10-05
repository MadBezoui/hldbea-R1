import numpy as np

from pymoo.core.population import Population

from hldbea.scoring import compute_local_scores

from util.misc import get_caller_function

class Advance4Util:

    # Attention: maybe used also in advance4a
    @staticmethod #TODO: replace _calc_scores
    def calc_scores(algorithm, _F=None, pop=None, K=None, steps=None):
        '''
        Computes the neighbourhood radius (step) for each objective axis and
        returns the fitness scores of all individuals in the population.

        Paper formula (Section 3.1, Definition 2):
            pₐ = K · (max fₐ - min fₐ) / N

        where N = |pop| and K is the neighbourhood radius multiplier.
        The paper fixes K = 1 (see ibea.py, IBEA.__init__, default K=1).
        K is kept as a tunable parameter for ablation studies.

        _F is given priority over pop if both are provided.
        '''
        target_pop = pop
        if target_pop is None and hasattr(algorithm, "pop"):
            candidate = algorithm.pop
            if _F is None or len(candidate) == len(_F):
                target_pop = candidate
        if _F is None:
            if pop is None:
                pop = algorithm.pop
            _F = pop.get('F')
        if K is None:
            K = getattr(algorithm, 'K', 1.0)
        result = compute_local_scores(
            _F,
            k=K,
            cone_epsilon=getattr(algorithm, 'cone_epsilon', 0.0),
            lmbda=getattr(algorithm, 'lmbda', 0.1),
            neighborhood_mode=getattr(algorithm, 'neighborhood_mode', 'axis'),
            score_aggregation=getattr(algorithm, 'score_aggregation', 'sum'),
        )
        if target_pop is not None and len(target_pop) == len(_F):
            target_pop.set('ScoreRaw', result.raw_scores)
            target_pop.set('ScoreAxis', result.per_axis_counts)
            target_pop.set('FitDistance', result.normalized_distances)
            target_pop.set('GlobalND', result.nondominated)
            target_pop.set('FalseZero', result.false_zero)
        algorithm._last_score_result = result
        return result.fitness

    @staticmethod
    def get_sorted(Ff):
        '''
        fn1: position dans FfSorted du 1er élément avec le plus grand score.
        Ff n'est pas obligatoirement les scores de la pop globale. Aucun lien à pop ou à algorithm.
        Trier par scores croissants. Récupérer les indices. (les scores sont tous <= 0, donc, max=0)
        FfSorted: Contient les indices du vecteur Ff trié
        Ff0: vecteur Ff trié
        Ffnul: Indices des Ff nuls
        '''
        FfSorted = np.argsort(Ff)
        Ff0 = Ff[FfSorted]
        i = Ff0[len(Ff0)-1] # Score max dans Ff0 (le max c 0 si exist car les scores sont <= 0)
        #Ffnul = np.where(Ff0 == i)[0] # Peut ne pas exister !?
        Ffnul = np.flatnonzero(Ff0==i)
        fn1 = Ffnul[0] # 1er indice dans FfSorted ou fitness=i. FfSorted[fn1:] ou FfSorted[Ffnul]donne tous les nuls si exst
                            # Att: indice du max ps auto 0
        #print(FfSorted, Ff[FfSorted[Ffnul]])
        #LOG:        print("Fitness nulle à partir de l'indice (dans FfSorted): ", fn1) # Dans FfSorted pas dans Ff!
        return FfSorted, fn1

    @staticmethod
    def do_exact_infill(algorithm, pop, Ni=None):
        '''
        Returns an offspring from pop using an exact method.
        '''
        assert get_caller_function() == '_custom_infill_1a', f"do_exact_infill() called elsewhere than _custom_infill_1a ({get_caller_function()})."

        from ibeas.nibea.nibea_util import solve_scalarized_problem #do_lbfgsb_infill
        #list_x = do_lbfgsb_infill(algorithm, pop, Ni)
        list_x = solve_scalarized_problem(algorithm, pop, Ni, method=algorithm.exact_method)
        off1 = Population.new("X", np.array(list_x))
        #algorithm.evaluator.eval(algorithm.problem, off1, algorithm=algorithm)
        #print(self.pop[:l].get("Fit"))
        #print(len(off1))
        return off1

    @staticmethod
    def get_dominant_x(x, px, y, py):
        #Idea: utiliser la dominance

        dm = Dominator.get_relation(px, py)
        if dm == 1:
            return x
        elif dm == -1:
            return px
        else:
            return x
