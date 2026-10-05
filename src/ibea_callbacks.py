from pymoo.core.callback import Callback
import numpy as np

from pymoo.indicators.hv import HV

from idea1_util import _plotX, _plotF

class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["ihv"] = [] #Hypervolume indicator
        print('init callback (Nothing done here)')
        self.fig = None

    def update(self, algorithm):
        ref_point = np.array([1, 1])
        ind = HV(ref_point=ref_point)
        H = ind(algorithm.pop.get("F"))
        self.data["ihv"].append(H)
        #print("IHV:\n", np.array2string(H, precision=2, separator=', ' , suppress_small=True))

        print('update callback (inside _post_advance): end of n_iter n° ', algorithm.n_iter)
        #print('pop_size = ', algorithm.pop_size)
        #print(algorithm.pop)
        if (algorithm.n_iter == 1):
            F = algorithm.pop.get("F")
            X = algorithm.pop.get("X")
            print("X initiale:\n", np.array2string(X, precision=2, separator=', ' , suppress_small=True))
            print("F initiale:\n", np.array2string(F, precision=2, separator=', ' , suppress_small=True))
            #print("X initiale:\n", X)
            #print("F initiale:\n", F)
            self.fig = _plotF(F, algorithm.problem, fig=self.fig, i=0, title='Initial F')
            if len(X) == 2:
                _plotX(X, fig=self.fig, i=1, title='X')
        if (algorithm.n_iter == 50):
            F = algorithm.pop.get("F")
            X = algorithm.pop.get("X")
            print("X initiale:\n", np.array2string(X, precision=2, separator=', ' , suppress_small=True))
            print("F initiale:\n", np.array2string(F, precision=2, separator=', ' , suppress_small=True))
            #print("X initiale:\n", X)
            #print("F initiale:\n", F)
            self.fig = _plotF(F, algorithm.problem, fig=self.fig, i=2, title='F iter 50')
            if len(X) == 2:
                _plotX(X, fig=self.fig, i=3, title='X')


