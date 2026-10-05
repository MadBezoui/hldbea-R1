import numpy as np

from pymoo.util.dominator import Dominator
from pymoo.operators.selection.tournament import compare


def hldbea_binary_tournament(pop, P, algorithm, **kwargs):
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape

    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")

    S = np.full(n_tournaments, -1) #, dtype=np.int)

    # now do all the tournaments
    for i in range(n_tournaments):
        a, b = P[i]

        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        a_fit, b_fit = pop[a].get("Fit"), pop[b].get("Fit")

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True, random_state=algorithm.random_state)

        # both solutions are feasible
        else:

            if  a_fit > b_fit:
                S[i] = a

            elif a_fit < b_fit:
                S[i] = b

            else:
                dm = Dominator.get_relation(pop[a].get("F"), pop[b].get("F"))
                if dm == 1:
                    S[i] = a
                elif dm == -1:
                    S[i] = b
                else:
                    S[i] = b

    return S

def ibea_binary_tournament(pop, P, **kwargs):
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape

    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")

    S = np.full(n_tournaments, -1) #, dtype=np.int)

    # now do all the tournaments
    for i in range(n_tournaments):
        a, b = P[i]

        # if the first individual is better, choose it
        if pop[a].get("Fit") > pop[b].get("Fit"):
            S[i] = a

        # otherwise take the other individual
        else:
            S[i] = b

    return S