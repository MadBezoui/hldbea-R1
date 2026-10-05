import numpy as np

from pymoo.core.population import Population

from ibeas.nibea.util import Advance4Util

#class EnvironmentalSelection: TODO new name
class EnvSelectionMethod:
    """
    Provides various strategies for population reduction and survival selection.

    Paper-compliant method (Algorithm 4, Section 3.3 of ACT_Ghani_V2-6):
        method_crowd :
            1. Keep all individuals with fit=0 (locally non-dominated in P̃)
            2. If their count > N, use NSGA-II Rank & Crowding as the tie-breaker
               to select exactly N survivors.
            → Selected as the default env_sel_method in _initialize_advance.

    Other methods are retained as experimental variants.
    These methods determine which individuals survive to the next generation
    when the number of best individuals in the merged population is higher than N.
    """

    @staticmethod
    def first_sel(algorithm, _F, FfSorted, fn1):
        """
        Initial filtering step to remove individuals with non-zero fitness.
        
        Args:
            algorithm: The optimization algorithm instance containing the population.
            _F (np.array): Matrix of objective function values.
            FfSorted (np.array): Indices of the population sorted by fitness.
            fn1 (int): The count of individuals with a fitness score of zero.
            
        Returns:
            np.array: The objective values of the remaining population.
        """
        assert (len(algorithm.pop) - fn1) >= algorithm.init_pop_size, f"first_sel: fn1 = {fn1}, pop: {len(algorithm.pop)}"
        algorithm.pop = algorithm.pop[FfSorted[fn1:]]
        algorithm.eliminated_tab = _F[FfSorted[:fn1]]
        return _F[FfSorted[fn1:]] # Car la pop a changé

    @staticmethod #TODO: Idée: éliminer les points dominés par un pt sélectioné (milieu de chq objectif)
    def second_sel(algorithm, pop):
        """
        Filters the population based on a 'middle-point' dominance strategy.
        
        Calculates a reference point at the center of each objective's range and 
        identifies individuals that are not dominated by this center point.
        
        Args:
            algorithm: The optimization algorithm instance.
            pop: The population to evaluate.
            
        Returns:
            np.array: Indices of individuals that survive the dominance check.
        """
        #pass
        _F = pop.get("F")
        middle_pt = np.zeros(_F.shape[1])

        dims = list(range(_F.shape[1]))
        for dim in dims:
            middle_val = (np.max(_F[:,dim]) - np.min(_F[:,dim]))/2 #Milieu de chaque objectif
            middle_pt[dim] = middle_val

        middle_pt = middle_pt.reshape(1, -1)
        #print(_F.shape, middle_pt.shape)
        from pymoo.util.dominator import Dominator
        D = Dominator.calc_domination_matrix(middle_pt, _F)
        #print("D: ", D.shape, D)
        Dm = np.where(D != 1)
        return Dm[1]
        #
        print("Dm: ", Dm[1])
        if len(Dm) < algorithm.init_pop_size:
            print(len(Dm[1]))
            # TODO: Compléter

    @staticmethod
    def get_nb_worst(algorithm, _F, FfSorted, fn1):
        """
        Identifies the number of individuals sharing the worst fitness score. (the first ones in FfSorted)
        
        Args:
            algorithm: The optimization algorithm instance.
            _F (np.array): Matrix of objective function values.
            FfSorted (np.array): Sorted indices of the population.
            fn1 (int): Fitness count threshold.
            
        Returns:
            int: The index/count of the worst individuals to be removed.
        """
        '''
        A utiliser dans method2.
        Used in `sel_pop3()`
        '''
        #assert (len(algorithm.pop) - fn1) >= algorithm.init_pop_size, f"first_sel: fn1 = {fn1}, pop: {len(algorithm.pop)}"

        arr = algorithm.pop[FfSorted].get("Fit")
        #print(np.unique(arr, return_index=True)[1])
        idx = np.unique(arr, return_index=True)[1]
        if len(idx) > 1:
            return idx[1]
        else:
            return idx[0]

    @staticmethod # On enlève juste le surplus.
    def method0(algorithm, _F, FfSorted, fn1):
        """
        Simple surplus removal strategy.
        
        Filters for zero-fitness individuals first, then truncates the remaining 
        population to the initial population size by removing the excess from the top.
        """
        assert len(algorithm.pop) == len(_F)
        # Prendre à partir du 1er fitness = 0, donc, len(pop) > pop_size.
        _F = EnvSelectionMethod.first_sel(algorithm, _F, FfSorted, fn1)
        # Taille de la pop a changé.
        # Tous les éléments de la pop ont à présent une fitness = 0
        w = len(algorithm.pop) - algorithm.init_pop_size
        algorithm.pop = algorithm.pop[w:] # Enlever juste le surplus.
        algorithm.eliminated_tab = np.vstack((algorithm.eliminated_tab, _F[w:]))

    @staticmethod
    def method1(algorithm, _F, FfSorted, fn1):
        """
        Iterative fitness recalculation strategy.
        
        Repeatedly recalculates fitness scores for the surviving individuals and 
        removes the worst until the population size reaches the required limit.
        """
        '''
        This method selects the individuals with fitness score equal 0 (?) and then recalculates
        their scores and then doing the selection again. Here the neighborhood length will
        change because of the number of individuals in the selection...
        see sel_pop_2.
        '''
        EnvSelectionMethod.first_sel(algorithm, _F, FfSorted, fn1)
        off1  = algorithm.pop
        #print("*****len off1:", len(off1))

        while len(off1) > algorithm.init_pop_size:
            #Compute fitness again for off1
            _Fr = off1.get("F")
            Ffr = Advance4Util.calc_scores(algorithm, _Fr, off1)
            if algorithm.fitm:
                Ffr = np.minimum(off1.get("FitMin"), Ffr)
                off1.set("FitMin", Ffr)
            off1.set("Fit", Ffr)

            #nl = len(off1) - fn1

            FfSorted, fn1 = Advance4Util.get_sorted(Ffr)
            #print("fn1:", fn1, "len off1:", len(off1))
            #if fn1 >= algorithm.init_pop_size:
            if fn1 == 0 or (len(off1) - fn1) <= algorithm.init_pop_size:
                # Prendre algorithm.init_pop_size individus de off1 trié
                w = len(off1) - algorithm.init_pop_size
                #print("w:", w)
                algorithm.pop = off1[FfSorted[w:]]
                algorithm.eliminated_tab = np.vstack((algorithm.eliminated_tab, (off1[FfSorted[:w]]).get("F")))
                assert len(algorithm.pop) == algorithm.init_pop_size
                break
            elif len(off1) - fn1 > algorithm.init_pop_size:
                algorithm.eliminated_tab = np.vstack((algorithm.eliminated_tab, off1[FfSorted[:fn1]].get("F")))
                off1 = off1[FfSorted[fn1:]]

    @staticmethod
    def method2(algorithm, _F, FfSorted, fn1):
        """
        Aggressive truncation strategy based on worst fitness scores.
        
        Iteratively identifies and removes blocks of 'worst' individuals, 
        recalculating scores at each step until the population is sufficiently reduced.
        """
        '''
        This method selects the individuals with bad fitness remove them and then recalculates
        the fitness of the rest and then doing the selection again. Here the neighborhood length will
        change because of the number of individuals in the selection...
        see sel_pop_2.
        '''
        pop = algorithm.pop # Copy
        #
        while (len(algorithm.pop) - fn1) >= algorithm.init_pop_size: #or <= ?
            idx = EnvSelectionMethod.get_nb_worst(algorithm, _F, FfSorted, fn1)
            if idx == 0:
                break;
            #print(algorithm.n_iter)
            #print(len(algorithm.pop), fn1, idx)
            algorithm.pop = algorithm.pop[FfSorted[idx:]]
            #algorithm.eliminated_tab = _F[FfSorted[:idx]] #TODO
            _F = algorithm.pop.get("F")
            #
            #Compute fitness again for the truncated pop
            Ff = Advance4Util.calc_scores(algorithm) #, _F)
            if algorithm.fitm:
                Ff = np.minimum(algorithm.pop.get("FitMin"), Ff)
                algorithm.pop.set("FitMin", Ff)
            algorithm.pop.set("Fit", Ff)

            FfSorted, fn1 = Advance4Util.get_sorted(Ff) # Les bons points se trouvent à la fin.
        #For now, we'll just take the last N good individuals...
        algorithm.pop = algorithm.pop[FfSorted[(len(algorithm.pop) - algorithm.init_pop_size):]]

    @staticmethod # Sélectionner au hasard init_pop_size individus. old name: method0a
    def method_rand(algorithm, _F, FfSorted, fn1):
        """
        Randomized selection strategy.
        
        Keeps zero-fitness individuals and then randomly picks candidates 
        from the pool to fill the required population size.
        """
        EnvSelectionMethod.first_sel(algorithm, _F, FfSorted, fn1)
        w = len(algorithm.pop) - algorithm.init_pop_size
        pr = np.random.permutation(len(algorithm.pop))
        algorithm.pop = algorithm.pop[pr[w:]]
        algorithm.eliminated_tab = np.vstack((algorithm.eliminated_tab, _F[pr[:w]]))

    @staticmethod
    def method_rank(algorithm, _F, FfSorted, fn1):
        """
        Standard Rank and Crowding selection (NSGA-II style).
        
        Uses Pymoo's RankAndCrowding operator to select the most diverse and 
        non-dominated individuals.
        """
        EnvSelectionMethod.first_sel(algorithm, _F, FfSorted, fn1)
        #Voir NSGA2
        from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
        survival = RankAndCrowding()
        pop = algorithm.pop
        algorithm.pop = survival.do(algorithm.problem, pop,
                          n_survive=algorithm.init_pop_size,
                          algorithm=algorithm, random_state=algorithm.random_state)
        # Find eliminated points
        # Combine A and B, get unique rows and inverse mapping
        combined = np.vstack([pop.get('F'), algorithm.pop.get('F')])
        unq, inv = np.unique(combined, axis=0, return_inverse=True)

        # Count occurrences from A (positive) and from B (negative)
        cnt_A = np.bincount(inv[:len(pop)], minlength=len(unq))
        cnt_B = np.bincount(inv[len(pop):], minlength=len(unq))

        # The difference per unique row
        diff = cnt_A - cnt_B  # can't be negative if B ⊆ A

        # Keep only rows with remaining count > 0
        mask = diff > 0
        remaining_rows = unq[mask]
        counts = diff[mask]

        # Reconstruct the full multiset result
        result = np.repeat(remaining_rows, counts, axis=0)
        algorithm.eliminated_tab = np.vstack((algorithm.eliminated_tab, result))

    @staticmethod #Use Crowding distance pour départager les points avec fit=0
    def method_crowd(algorithm, _F, FfSorted, fn1):
        """
        Crowding Distance selection.
        
        Sorts the population by crowding distance and keeps the individuals 
        located in the least dense regions of the objective space.
        """
        from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
        from pymoo.util.randomized_argsort import randomized_argsort
        #print(fn1, "**********")
        EnvSelectionMethod.first_sel(algorithm, _F, FfSorted, fn1)
        _F = algorithm.pop.get("F")
        Fcd = calc_crowding_distance(_F, filter_out_duplicates=True)
        I = randomized_argsort(Fcd, order='descending', method='numpy', random_state=algorithm.random_state)

        # Prendre les individuals avec les CD les plus grands
        algorithm.pop = algorithm.pop[I[:algorithm.init_pop_size]]
        #for i in range(len(algorithm.pop)):
        #    pop[i].set("crowding", Fcd[I[i]])
        #print(pop.get("crowding"))
        algorithm.eliminated_tab = np.vstack((algorithm.eliminated_tab, _F[I[algorithm.init_pop_size:]]))

    @staticmethod #Method_inv: Use inv scores pour départager les points avec fit=0. ndim version
    def method_inv(algorithm, _F, FfSorted, fn1):
        EnvSelectionMethod.first_sel(algorithm, _F, FfSorted, fn1)
        K = algorithm.K
        _F = algorithm.pop.get("F")
        steps = [] #TODO: test if given
        dims = list(range(_F.shape[1]))
        for dim in dims:
            steps.append(K*(np.max(_F[:,dim]) - np.min(_F[:,dim]))/len(algorithm.pop))
        Ffinv = CalcInvScore.calc_all_inv_scores_ndim(_F, steps)
        I = np.argsort(Ffinv)
        #LOG: print("Inv scores: ", Ffinv)
        algorithm.pop = algorithm.pop[I[:algorithm.init_pop_size]]
        algorithm.eliminated_tab = np.vstack((algorithm.eliminated_tab, _F[I[algorithm.init_pop_size:]]))


class EnvSelectionExactMethod:

    @staticmethod
    def method_exact(algorithm, _F, FfSorted, fn1):
        print(f"fn1:{fn1}. Env sel exact method, iter: {algorithm.n_iter}")
        from ibeas.nibea.nibea_util import do_lbfgsb_infill
        EnvSelectionMethod.first_sel(algorithm, _F, FfSorted, fn1)
        #EnvSelectionMethod.second_sel(algorithm)
        #list_x = []
        #for px in pop.get("X"):
        #lg = len(algorithm.pop)//20 # Prendre le 1er 10ème de la pop
        lg = algorithm.init_pop_size//20 # Prendre le 1er 10ème de la pop
        list_x = do_lbfgsb_infill(algorithm, algorithm.pop[:lg], Ni=3, exm='2', tech='2')
        off = Population.new("X", np.array(list_x))
        algorithm.evaluator.eval(algorithm.problem, off, algorithm=algorithm)
        #TODO: if idea "id-1":
        off.set("FitMin", np.zeros(len(off)))
        algorithm.pop = Population.merge(algorithm.pop[lg:(algorithm.init_pop_size)], off)

    @staticmethod
    def method_exact_todo(algorithm, _F, FfSorted, fn1): #TODO: incub
        from ibeas.nibea.nibea_util import do_lbfgsb_infill
        EnvSelectionMethod.first_sel(algorithm, _F, FfSorted, fn1)
        #TO BE CONTINUED...



