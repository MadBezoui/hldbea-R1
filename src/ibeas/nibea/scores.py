import numpy as np

from util.misc import get_caller_function

class CalcScore:

    @staticmethod #ndim
    def get_dom_points(i, F):
        """
        Get The dominating points of i.
        A point j dominates i if f_j(k) <= f_i(k) for all k, with strict inequality on at least one.
        Here we use the <= convention on the first axis; the mask then filters the rest.
        """
        #assert get_caller_function() == 'calc_all_scores', f"get_dom_points() called elsewhere ({get_caller_function()})."

        v = (F[:, 0] <= F[i,0]) # Points at the same level on the first axis are also counted.
        v[i] = False
        Fv = F[v]
        #
        # Initialize the filter mask with all True values
        mask = np.ones(Fv.shape[0], dtype=bool)
        # Deduce the dimensions from the shape of F
        dims = list(range(Fv.shape[1]))
        dims.remove(0)
        for dim in dims:
            # Update the mask to include only points within the bounds
            mask &= (Fv[:, dim] <= F[i, dim])
        Fv = Fv[mask]

        return Fv

    @staticmethod
    def calc_score_axis(i, F, Fv, pas, ax, K):
        '''
        Returns the list of points that dominate i in the neighborhood along the ax axis.
        Fv contains all the dominating points of i; here we select just those along the ax axis.
        Fv as a param to not call get_dom_points() for each axis.

        Paper reference: Definition 3 — Vₐ(A) = { B ∈ Dom(A) : fₐ(A) − pₐ ≤ fₐ(B) ≤ fₐ(A) }
        where pₐ = (max fₐ − min fₐ) / N  (computed upstream in calc_scores()).
        '''
        if len(Fv)==0:
            return []

        a = F[i, ax] - pas
        v = (Fv[:, ax] >= a)
        Fv = Fv[v]

        return Fv


    @classmethod
    def calc_score(cls, i, F, Fv, steps, K):
        '''
        Calculates the score of i.
        Paper: score(A) = Σₐ |Vₐ(A)|
        Returns a list of the number of dominating points of i along each axis.
        Be aware that some points will be used more times in the global score.
        '''
        ls = []
        axis = list(range(F.shape[1]))
        for ax in axis:
            # Get the F of all the points dominating i along the ax axis in its neighborhood
            ll = cls.calc_score_axis(i, F, Fv, steps[ax], ax, K)
            # Calculates the score along the ax axis as the nb of dominating points of i
            ls.append(len(ll))

        return ls


    @classmethod
    def calc_all_scores(cls, F, steps, algorithm):
        """
        Dispatches to the appropriate scoring aggregation method.

        The default (sc_method == "0") implements the formula from the paper:
            fit(A) = -score(A) - λ · dist(A)
        where score(A) = Σₐ |Vₐ(A)|  and
              dist(A)  = ‖f(A) - z*‖₂ / d_max  (normalised distance to the ideal point).
        λ is read from algorithm.lmbda (default 0.1, as per paper Section 3.2).
        """
        #Getting The dominating points of all points once. To be used for all axis.
        Fvs = [cls.get_dom_points(i1, F) for i1 in np.arange(len(F))]
        scores = [cls.calc_score(i1, F, Fvs[i1], steps, algorithm.K) for i1 in np.arange(len(F))]

        sc_method = algorithm.sc_method
        if (sc_method == "0"):
            return cls.calc_all_scores_0(F, scores, steps, algorithm)
        # Add other methods for calculating the scores here
        else:
            return cls.calc_all_scores_0(F, scores, steps, algorithm)


    @classmethod
    def calc_all_scores_0(cls, F, scores, steps, algorithm=None):
        '''
        PAPER-CONFORMANT fitness formula (Algorithm 1, Section 3.2):
            fit(A) = -score(A) - λ · dist(A)

        where:
            score(A) = Σₐ |Vₐ(A)|        (sum of dominating neighbours per axis)
            dist(A)  = ‖f(A) - z*‖₂ / d_max   (normalised L2 distance to the ideal point z*)
            λ        = algorithm.lmbda    (default 0.1, paper Section 3.2)

        If algorithm is None (e.g., called from legacy code), falls back to -Σ(score).
        '''
        base_scores = np.array([(-1) * np.sum(ll) for ll in scores])

        if algorithm is None:
            return base_scores

        lmbda = getattr(algorithm, 'lmbda', 0.1)

        if lmbda == 0.0:
            return base_scores

        # z*: ideal point — component-wise minimum over the population
        z_star = np.min(F, axis=0)      # shape (n_obj,)

        # Raw L2 distances to ideal point
        dists = np.linalg.norm(F - z_star, axis=1)   # shape (N,)

        # Normalise by the maximum distance (avoid div-by-zero)
        d_max = np.max(dists)
        if d_max > 0:
            dists = dists / d_max

        return base_scores - lmbda * dists
