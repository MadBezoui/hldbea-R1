import numpy as np

from pymoo.core.population import Population

from ibeas.nibea.envselectionmethod import EnvSelectionMethod
from ibeas.nibea.util import Advance4Util

class SelectPopMethod:
    '''
    This class contains static methods to be used when the number of individuals in the
    merged population with the best fitness score is less than the
    initial population size.
    '''

    @staticmethod
    def sel_pop0(algorithm, _F, FfSorted):
        '''
        This method selects the last init_pop_size individuals. which will include the points
        with best(max) fitness score and completes the rest with other points with fitness smaller
        than max. This is without recalculating fitness...
        '''
        #algorithm.pop = algorithm.pop[FfSorted[algorithm.init_pop_size:]]
        #algorithm.eliminated_tab = _F[FfSorted[:algorithm.init_pop_size]]
        algorithm.pop = algorithm.pop[FfSorted[len(FfSorted)-algorithm.init_pop_size:]]
        algorithm.eliminated_tab = _F[FfSorted[:len(FfSorted)-algorithm.init_pop_size]]
        assert len(algorithm.pop) == algorithm.init_pop_size, f"Error: pop size ({len(algorithm.pop)})."

    @staticmethod
    def sel_pop1(algorithm, FfSorted, fn1): #TODO: Choisir les 0 mais pour le reste ?
        '''
        This method selects the individuals with the max fitness and then use
        the method second_sel (middle points technique) to continue...
        '''
        hFfSorted = FfSorted[fn1:]
        off1 = algorithm.pop[hFfSorted]
        reste = algorithm.pop[FfSorted[:fn1]]
        #algorithm.pop = ...
        #algorithm.eliminated_tab = ...
        #Dm: vecteur des indices des pts non dominés par le middle pt...
        Dm = EnvSelectionMethod.second_sel(algorithm, reste)
        print("len Dm: ", len(Dm))
        if len(Dm) < fn1-algorithm.init_pop_size:
            # Create a boolean mask
            mask = np.ones(reste.shape, dtype=bool)
            mask[Dm] = False
            # Filter the array using the mask
            reste1 = reste[mask]

            off1 = Population.merge(off1, reste[Dm], reste1[:algorithm.init_pop_size-len(Dm)-len(off1)])
            #off1 maintn contient scor=0 + non-tabou elts
        else:
            off1 = Population.merge(off1, reste[Dm[:fn1-algorithm.init_pop_size]])
        algorithm.pop = off1
        #algorithm.eliminated_tab = ...

    @staticmethod
    def sel_pop2(algorithm, FfSorted, fn1):
        '''
        This method selects the individuals with max fitness (equal 0?) and then recalculates
        the fitness of all the remaining individuals and then doing the selection again.
        '''
        #print("FfSorted", FfSorted)
        #print("pop F", algorithm.pop.get("F"))
        #Fg = algorithm.pop.get("F")
        #print("FfSorted reste", FfSorted[:fn1])
        off1  = algorithm.pop[FfSorted[fn1:]]
        reste = algorithm.pop[FfSorted[:fn1]]
        #print("reste F", reste.get("F"))

        #from matplotlib import pyplot as plt
        #from pymoo.visualization.scatter import Scatter
        #plot = Scatter(title="reste F")
        #Fp = reste.get("F")
        #plot.add(Fg, facecolor="none", edgecolor="red")
        #plot.add(Fp, facecolor="none", edgecolor="yellow")
        #plot.show()

        while len(off1) < algorithm.init_pop_size:
            # La boucle est finie car si aucun point avec fit==0, get_sorted() renvoie fn1==1 par ex. ...
            #print(len(off1), len(reste))
            #Compute fitness again for just reste
            _Fr = reste.get("F")
            #print("Old FitMin reste: ", reste.get("FitMin"))
            Ffr = Advance4Util.calc_scores(algorithm, _Fr, reste)
            if algorithm.fitm:
                Ffr = np.minimum(reste.get("FitMin"), Ffr)
                reste.set("FitMin", Ffr)
            reste.set("Fit", Ffr)
            # Le score changent car fmax et fmin de reste sont diff de ceux de algo.pop
            # Si on laisse les steps utilisées dans algo.pop, le résultat sera m^ que sel_pop0.
            # Mais ici steps de reste augmente le score (-) des pts car on divise par
            # len(reste) qui é < len(pop) ce qui fait que steps est plus large.
            #print("New Ff reste: ", Ffr)

            FfSorted, fn1 = Advance4Util.get_sorted(Ffr)
            #print("FfSorted reste", FfSorted)
            #print(fn1)
            m = np.maximum(len(reste) - (algorithm.init_pop_size - len(off1)), fn1)
            #print(m)
            if True: # Autre technique: Trop de points avec score inf: choisir encore
                if fn1 < m:
                    if fn1 != 0:
                        reste = reste[FfSorted[fn1:]]
                        continue
            #off1 = Population.merge(off1, reste[FfSorted[m:]])
            off1 = Population.merge(reste[FfSorted[m:]], off1)
            reste = reste[FfSorted[:m]]
            if len(off1) == algorithm.init_pop_size:
                break
        algorithm.pop = off1
        algorithm.eliminated_tab = reste.get("F")

    @staticmethod
    def sel_pop3(algorithm, FfSorted, fn1):
        '''
        This method selects the individuals with bad fitness remove them and then recalculates
        the fitness of the rest and then doing the selection again. Here the neighborhood length will
        change because of the number of individuals in the selection...
        Same as method2 of env sel.
        '''
        #pop = algorithm.pop # Copy
        #
        _F = algorithm.pop.get("F")
        while True:
            idx = EnvSelectionMethod.get_nb_worst(algorithm, _F, FfSorted, fn1)
            if idx == 0:
                break
            if (len(algorithm.pop) - idx) <= algorithm.init_pop_size:
                break
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

            FfSorted, fn1 = Advance4Util.get_sorted(Ff)
        #For now, we'll just take the last N good individuals...
        algorithm.pop = algorithm.pop[FfSorted[(len(algorithm.pop) - algorithm.init_pop_size):]]
        #TODO Not finished (tech: remove this comment after finishing)


    @staticmethod
    def sel_pop4(algorithm, FfSorted, fn1): #Used
        '''
        Take all best individuals, then complete the remaining with ranking and crowding.
        See pop_1()
        '''
        hFfSorted = FfSorted[fn1:]
        off1 = algorithm.pop[hFfSorted]
        reste = algorithm.pop[FfSorted[:fn1]]
        #Voir NSGA2
        from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
        survival = RankAndCrowding()
        reste = survival.do(algorithm.problem, reste,
                          n_survive=algorithm.init_pop_size-len(off1),
                          algorithm=algorithm, random_state=algorithm.random_state)
        algorithm.pop = Population.merge(reste, off1)
        #algorithm.eliminated_tab = TODO


