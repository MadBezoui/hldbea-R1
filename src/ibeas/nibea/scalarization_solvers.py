import numpy as np
from math import comb

from scipy.optimize import minimize

from hldbea.local_search import solve_epsilon_constraint

class ScalarizationProvider:
    """
    Handles the transformation of multi-objective problems into 
    single-objective sub-problems (Scalarization).
    """

    @staticmethod
    def kth_uniform_weight(m, divisions, k):
        """
        Return the k-th weight vector in lexicographic order for
        the uniform grid over the simplex, with wrap-around indexing.

        m         = number of objectives
        divisions = number of grid divisions
        k         = index (can be any integer, positive or negative)
        """
        """
        k is taken equal to the current iteration (generation).
        This means that we change the weights by iteration...
        """
        total = comb(divisions + m - 1, m - 1)
        k = k % total  # wrap-around

        n = divisions + m - 1  # range end
        r = m - 1              # number of bars

        combo = []
        x = 0
        for i in range(r):
            while True:
                count = comb(n - x - 1 + r - i - 1, r - i - 1)
                if k < count:
                    combo.append(x)
                    break
                k -= count
                x += 1

        # Convert combination to weights
        points = [combo[0]] \
               + [combo[i] - combo[i - 1] - 1 for i in range(1, m - 1)] \
               + [divisions + m - 2 - combo[-1]]

        return np.array(points) / divisions

    @staticmethod
    def get_weight_vector(pb, weight_strategy, n_iter):
        '''
        Generates weight vectors based on the chosen strategy.
        n_iter needed only for uniform_weights()
        '''
        L = pb.n_obj
        if weight_strategy == 'rnd':
            #Generate L random numbers
            #lambdas = np.random.rand(L)
            lambdas = algorithm.random_state.random((L,))
            lambdas /= np.sum(lambdas)
        elif weight_strategy == '1': # == one objective
            lambdas = np.zeros(L) # Same values
            lambdas[0] = 1
        elif weight_strategy == '2':
            lambdas = ScalarizationProvider.kth_uniform_weight(L, 3, n_iter)
            #print(lambdas)
        elif weight_strategy == 'ex': # == eq
            lambdas = np.ones(L) # Same values
            lambdas /= np.sum(lambdas)
        return lambdas

    @staticmethod
    def get_constraints(pb):
        # === Check for constraints and build SciPy constraint list ===
        constraints = []

        # 1. Inequality constraints: G <= 0  --> c(x) = -G >= 0
        if pb.n_ieq_constr > 0:
            def ineq_constraint(x):
                x = x.reshape(1, -1)
                G_val = pb.evaluate(x, return_values_of=["G"])
                return -G_val[0]  # c(x) >= 0
            constraints.append({'type': 'ineq', 'fun': ineq_constraint})
            #print("Inequality constraints detected and added.")

        # 2. Equality constraints: H == 0  --> h(x) == 0
        if pb.n_eq_constr > 0:
            def eq_constraint(x):
                x = x.reshape(1, -1)
                H_val = pb.evaluate(x, return_values_of=["H"])
                return H_val[0]  # h(x) == 0
            constraints.append({'type': 'eq', 'fun': eq_constraint})
            #print("Equality constraints detected and added.")

        return constraints

class EpsilonConstraintNLP(ScalarizationProvider):
    """
    Exact Method using the Epsilon-Constraint formulation.
    Minimizes one objective while treating others as inequality constraints.
    """

    @staticmethod
    def get_epsilon_bounds(pb, initial_x, objective_index):
        """
        Calculates epsilon values based on the current individual's performance.
        Typically, epsilon is set to the current objective values to ensure 
        the solver finds a point that is at least as good as the current one.
        """
        f_values = pb.evaluate(initial_x, return_values_of=["F"])
        # Epsilon is usually the current vector of objectives
        return f_values

    @staticmethod
    def solve(
        pb,
        px,
        max_iter,
        main_obj_index=0,
        method='SLSQP',
        *,
        ledger=None,
        initial_f=None,
    ):
        if method != 'SLSQP':
            raise ValueError("epsilon-constraint refinement supports only SLSQP")
        if ledger is None:
            raise ValueError("epsilon-constraint refinement requires an EvaluationLedger")
        return solve_epsilon_constraint(
            pb,
            px,
            main_obj_index=main_obj_index,
            max_iter=max_iter,
            ledger=ledger,
            initial_f=initial_f,
        )

    @staticmethod
    def solve_unaccounted_legacy(pb, px, max_iter, main_obj_index=0, method='SLSQP'):
        """Deprecated implementation retained only for historical comparisons."""
        # 1. Get current performance to set epsilon bounds
        epsilons = EpsilonConstraintNLP.get_epsilon_bounds(pb, px, main_obj_index)
        
        # 2. Define the single-objective function (Minimize f_i)
        def objective_function(x):
            return pb.evaluate(x, return_values_of=["F"])[main_obj_index]

        # 3. Define the epsilon constraints: f_j(x) <= epsilon_j for all j != i
        def epsilon_constraint(x):
            f_vals = pb.evaluate(x, return_values_of=["F"])
            # Scipy expects constraints in the form g(x) >= 0, so: epsilon - f(x) >= 0
            constraints = []
            for j in range(len(f_vals)):
                if j != main_obj_index:
                    constraints.append(epsilons[j] - f_vals[j])
            return np.array(constraints)

        # 4. Combine with existing problem constraints (G and H)
        nlp_constraints = EpsilonConstraintNLP.get_constraints(pb) 
        nlp_constraints.append({'type': 'ineq', 'fun': epsilon_constraint})

        bounds = np.column_stack((pb.xl, pb.xu))
        
        # 5. Run the NLP Solver (SLSQP is preferred for multiple constraints)
        result = minimize(
            objective_function, 
            px, 
            method=method, 
            bounds=bounds, 
            constraints=nlp_constraints, 
            options={'maxiter': max_iter}
        )
        return result.x


class GoalProgrammingOptimizer(): #TODO compare do_3() to this if the same, then delete do_3()
    """NLP using a Goal Programming approach to minimize deviations from a target point."""

    @staticmethod
    def solve(problem, x0, max_iter):
        f0 = problem.evaluate(x0, return_values_of=["F"])
        def goal_obj(x):
            f = problem.evaluate(x)
            return np.sum(np.maximum(f0 - f, 0)) + np.sum(np.maximum(f - f0, 0))
        constraints = ScalarizationProvider.get_constraints(problem)
        bounds = np.column_stack((problem.xl, problem.xu))
        res = minimize(goal_obj, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': max_iter})
        return res.x


class WeightedSumOptimizer(): #TODO to replace do_2() when ready
    """NLP using the Weighted Sum scalarization method (Standard NLP)."""

    @staticmethod
    def solve(problem, x0, max_iter, method='COBYLA', strategy='rnd'):
        weights = ScalarizationProvider.get_weight_vector(problem, strategy, 0)
        def obj(x): return np.sum(weights * problem.evaluate(x, return_values_of=["F"]))
        bounds = np.column_stack((problem.xl, problem.xu))
        res = minimize(obj, x0, method=method, bounds=bounds, options={'maxiter': max_iter})
        return res.x

