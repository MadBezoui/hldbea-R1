import numpy as np

from util.misc import get_caller_function

from pymoo.util.dominator import Dominator
from scipy.optimize import minimize

from ibeas.nibea.scalarization_solvers import ScalarizationProvider, EpsilonConstraintNLP, GoalProgrammingOptimizer, WeightedSumOptimizer
from ibeas.nibea.util import Advance4Util

def solve_scalarized_problem(algorithm, pop=None, Ni=None, method='weighted'):
    """
    Dispatcher for exact mathematical programming methods (NLP).
    
    Parameters:
    - method: 'weighted', 'epsilon', or 'goal'
    - tech: '1' for direct replacement, '2' for dominance-based selection
    """
    assert get_caller_function() == 'do_exact_infill', f"solve_scalarized_problem() called elsewhere ({get_caller_function()})."
    if pop is None:
        pop = algorithm.pop
    
    pb = algorithm.problem
    
    # Ni represents the max iterations for the NLP solver
    if Ni is None:
        Ni = len(pop) // 20 

    # 1. Select the specific Exact Method solver class
    solvers = {
        'weighted': WeightedSumOptimizer,
        'epsilon': EpsilonConstraintNLP,
        'goal': GoalProgrammingOptimizer
    }
    
    solver_class = solvers.get(method, WeightedSumOptimizer)
    
    list_x = []
    for px in pop.get("X"):
        # 2. Execute the solver
        if method == 'epsilon':
            # Epsilon method randomly picks an objective to minimize
            main_idx = algorithm.random_state.integers(0, pb.n_obj)
            r = solver_class.solve(pb, px, Ni, main_obj_index=main_idx)#np.random.randint(0, pb.n_obj))
        elif method == 'weighted':
            # Weighted sum can use 'rnd', 'eq', or 'uniform' strategies
            r = solver_class.solve(pb, px, Ni, method='SLSQP', strategy='rnd')
        else:
            # Goal Programming
            r = solver_class.solve(pb, px, Ni)

        list_x.append(r)
            
    return list_x


class ExactMethodBase:
    """
    Provides local search refinement methods for multi-objective problems 
    using scalarization techniques and scipy.optimize.
    """

    @staticmethod
    def my_callback(intermediate_result):
        print(intermediate_result.fun)

    @staticmethod
    def do_2(pb, px, Ni, method='COBYLA', lmbda='rnd', n_iter=1): #TODO
        '''
        Prend en compte les contraintes.
        Method: See (https://docs.scipy.org/doc/scipy-1.16.0/reference/optimize.minimize.html)
        lmbda == rnd : Take random lambdas
        lmbda == eq  : Take equal lambdas
        '''

        lambdas = ScalarizationProvider.get_weight_vector(pb, lmbda, n_iter)
        #n_iter needed only for uniform_weights()

        def objective_function(x):
            return np.sum(lambdas * pb.evaluate(x, return_values_of=["F"]))

        constraints = []
        if pb.n_eq_constr > 0 or pb.n_ieq_constr > 0:
            constraints = ScalarizationProvider.get_constraints(pb)

        bounds = np.column_stack((pb.xl, pb.xu)) #[(-1, 1), (0, 2), (-3, 3)]  # Bounds for each variable
        initial_guess = px  # Initial guess
        result = minimize(objective_function, initial_guess, method=method, #'Nelder-Mead', #'SLSQP',
                     bounds=bounds, constraints=constraints, options={'maxiter':1})
                     #callback=ExactMethodBase.my_callback, tol=1e-1)

        # === Results ===
        if result.success:
            x_opt = result.x
            F_opt = pb.evaluate(x_opt.reshape(1, -1), return_values_of=["F"])[0]
            #print("\n Optimization successful!")

            # Check constraint violations
            #info = problem.evaluate(x_opt.reshape(1, -1), return_values_of=["F", "G", "H"])
            #if "G" in info:
            #    print("Inequality constraints (G):", info["G"][0])
            #if "H" in info:
            #    print("Equality constraints (H):", info["H"][0])
            return result.x
        else:
            #print("\n Optimization failed:", result.message)
            #print(px, result.x)
            #return px
            return result.x


        #return result.x

    @staticmethod #TODO Goal programming approach
    def do_3(pb, px, Ni, lmbda='rnd'):
        '''
        Goal programming approach.
        '''

        #Goal programming approach.
        #Note: Since scipy doesn’t handle auxiliary variables (n, p) directly, we
        # compute deviations analytically using max (softened via ReLU-like form). This works well in practice.
        f0 = pb.evaluate(px)
        weights_neg = None
        weights_pos = None
        if False:
            L = pb.n_obj #L = len(pop.get("F"))
            lambdas = None
            #Generate L random numbers
            #lambdas = np.random.rand(L)
            #lambdas /= np.sum(lambdas)
            weights_neg = np.random.rand(L)
            weights_pos = np.random.rand(L)
        else:
            weights_neg = np.ones_like(f0)  # weight on n_i
            weights_pos = np.ones_like(f0)  # weight on p_i
        def goal_objective(x):
            f = pb.evaluate(x)
            neg_dev = np.maximum(f0 - f, 0)    # n_i >= max(f0_i - f_i, 0)
            pos_dev = np.maximum(f - f0, 0)    # p_i >= max(f_i - f0_i, 0)
            return np.dot(weights_neg, neg_dev) + np.dot(weights_pos, pos_dev)

        constraints = ScalarizationProvider.get_constraints(pb)

        bounds = np.column_stack((pb.xl, pb.xu)) #[(-1, 1), (0, 2), (-3, 3)]  # Bounds for each variable
        initial_guess = px  # Initial guess
        result = minimize(goal_objective, initial_guess, method='SLSQP',
                     bounds=bounds, constraints=constraints, options={'maxiter':2})

        # === Results ===
        if result.success:
            x_opt = result.x
            F_opt = pb.evaluate(x_opt.reshape(1, -1), return_values_of=["F"])[0]
            #print("\n🎉 Optimization successful!")
            #print("Optimal objectives:", F_opt)

            # Check constraint violations
            #info = problem.evaluate(x_opt.reshape(1, -1), return_values_of=["F", "G", "H"])
            #if "G" in info:
            #    print("Inequality constraints (G):", info["G"][0])
            #if "H" in info:
            #    print("Equality constraints (H):", info["H"][0])
            return result.x
        else:
            #print("\nOptimization failed:", result.message)
            #print(px, result.x)
            #return px
            return result.x


        #return result.x



# For each individual xp in the pop, taken as initial sol, generate new x...
def do_lbfgsb_infill(algorithm, pop=None, Ni=None, exm='1', tech='1'):
    if pop is None:
        pop = algorithm.pop
    # Attention, pop (dans les params) peut être différent de algorithm.pop
    pb  = algorithm.problem
    if Ni is None:
        Ni = len(pop)//20 #//5 #=20

    list_x = []
    for px in pop.get("X"):
        r = None
        if exm == '1': #NIBEA+
            #r = ExactMethodBase.do_2(pb, px, Ni, method='L-BFGS-B', lmbda='rnd')

            #r = EpsilonConstraintNLP.solve(pb, px, Ni, main_obj_index=np.random.randint(0, pb.n_obj))
            #r = ExactMethodBase.do_2(pb, px, Ni, lmbda='rnd')
            r = WeightedSumOptimizer.solve(pb, px, Ni, 'SLSQP', 'rnd')

            #r = ExactMethodBase.do_2(pb, px, Ni, lmbda='1')
            #r = ExactMethodBase.do_2(pb, px, Ni, lmbda='2', n_iter=algorithm.n_iter)
            #r = ExactMethodBase.do_2(pb, px, Ni, lmbda='ex')

            #r = GoalProgrammingOptimizer.solve(pb, px, Ni) #Remplace do_3() une fois prête
            #r = ExactMethodBase.do_3(pb, px, Ni, lmbda='rnd')
            if False: #DEBUG
                print("exact: ")
                Fx = pb.evaluate(px)
                Fy = pb.evaluate(r)
                #print(Fx)
                #print(Fy)
                dm = Dominator.get_relation(Fx, Fy)
                if dm == 1:
                    print("px domine r")
                elif dm == -1:
                    print("r domine px")
                else:
                    print("Pas de dominance")
        else:
            r = ExactMethodBase.do_2(pb, px, Ni, method='L-BFGS-B', lmbda='eq')

        if tech == '1': #NIBEA+
            #Method1: Choisir directement la solution trouvée par la méthode exacte
            list_x.append(r)
        else:
            #Method2: Prendre le point dominant entre px et le point donnée par la méthode exacte
            y = pb.evaluate(r, return_values_of=["F"])
            py = pb.evaluate(px, return_values_of=["F"])
            list_x.append(Advance4Util.get_dominant_x(r,px, y, py))
        #print(np.array2string(pb.evaluate(px, return_values_of=["F"]), precision=2, separator=', ' , suppress_small=True),
        #     np.array2string(pb.evaluate(result.x, return_values_of=["F"]), precision=2, separator=', ' , suppress_small=True))
    return list_x


