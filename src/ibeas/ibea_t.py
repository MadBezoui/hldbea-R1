
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
#from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput

from ibea_callbacks import MyCallback
from ibea import IBEA

class IBEATest:
    '''
    This class centralize the calling of my different algorithms.
    '''

    @staticmethod
    def __ibea(Advance, problem, K, pop_size, n_gen, **kwargs):
        #TODO: add selection to all Advance
        if hasattr(Advance, 'selection'):
            selection = Advance.selection() 
        else:
            selection = TournamentSelection(func_comp=ibea_binary_tournament)

        algorithm = IBEA(Advance,
                         pop_size=pop_size,
                         K = K,
                         sampling=FloatRandomSampling(),
                         selection=selection,
                         crossover=SBX(eta=20, prob=0.9),
                         mutation=PM(prob=1/problem.n_var, eta=20),
                         output=MultiObjectiveOutput(),
                         **kwargs)
        criterion = 'n_gen'
        if 'stop_criterion' in kwargs:
            criterion = kwargs.get('stop_criterion')

        termination = (criterion, n_gen)
        algorithm.termination = termination
        #algorithm.setup(problem=problem, callback=MyCallback(), **kwargs)
        algorithm.setup(problem=problem, **kwargs)
        #Une technique pour définir un _infill personnalisé
        if hasattr(Advance, "_infill"):
            Advance._infill(algorithm, **kwargs)
        res = algorithm.run()
        return res, algorithm, n_gen


    @staticmethod
    def ibea_4(problem=None, K=0.1, pop_size=100, n_gen=8, **kwargs):
        from ibeas.nibea.ibea_advance_4 import Advance               #  (nibea)
        return IBEATest.__ibea(Advance, problem, K, pop_size, n_gen, **kwargs)

