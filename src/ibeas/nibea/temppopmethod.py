import numpy as np

from pymoo.core.population import Population

class TempPopMethod:

    @staticmethod # Méthode redéfinie dans advance4A
    def _temp_pop_0(algorithm, infills, **kwargs):

        # merge the offsprings with the current population
        if infills is not None:
            # Merge
            algorithm.pop = Population.merge(algorithm.pop, infills)
        else: #DEBUG
            print("No merge, infills is None")

    @staticmethod
    def _temp_pop_1(algorithm, infills, **kwargs):

        from ibeas.nibea.nibea_util import do_lbfgsb_infill
        # merge the offsprings with the current population
        if infills is not None:
            Ff = algorithm.pop.get("FitMin")
            FfSorted, fn1 = Advance4Util.get_sorted(Ff)
            off0 = algorithm.pop[FfSorted[:fn1]]

            list_x = do_lbfgsb_infill(algorithm, algorithm.pop[FfSorted[fn1:]], Ni=3, exm='2', tech='2')
            off1 = Population.new("X", np.array(list_x))
            algorithm.evaluator.eval(algorithm.problem, off1, algorithm=algorithm)
            off1.set("FitMin", np.zeros(len(off1)))
            off1.set("FitLife", np.zeros(len(off1)))
            algorithm.pop = Population.merge(off0, off1, infills)
        else: #DEBUG
            print("No merge, infills is None")

    @staticmethod
    def _temp_pop_2(algorithm, infills, **kwargs):

        from ibeas.nibea.nibea_util import do_lbfgsb_infill
        # merge the offsprings with the current population
        if infills is not None:
            if algorithm.n_iter % 60 == 0:
                Ff = algorithm.pop.get("FitMin")
                FfSorted, fn1 = Advance4Util.get_sorted(Ff)
                off0 = algorithm.pop[FfSorted[:fn1]]

                list_x = do_lbfgsb_infill(algorithm, algorithm.pop[FfSorted[fn1:]], 3, 2, 2)
                off1 = Population.new("X", np.array(list_x))
                algorithm.evaluator.eval(algorithm.problem, off1, algorithm=algorithm)
                off1.set("FitMin", np.zeros(len(off1)))
                off1.set("FitLife", np.zeros(len(off1)))
                algorithm.pop = Population.merge(off0, off1, infills)
            else:
                algorithm.pop = Population.merge(algorithm.pop, infills)
                algorithm.pop.set("FitLife", np.zeros(len(algorithm.pop)))
        else: #DEBUG
            print("No merge, infills is None")


