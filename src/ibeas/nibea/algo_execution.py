from hldbea.runtime import configure_headless_matplotlib

configure_headless_matplotlib()

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize

from ibeas.nibea.ibea_test_4 import IBEATest4
from ibea_stats import show_stats
from hldbea.metrics import metric_reference_directions, metric_reference_point

GENERATED_PATH="./generated"

def run_algorithm(algorithm, problem, NIter, NPop, algo_name=None, **kwargs):
    """
    Runs a specified algorithm on a given problem.

    Parameters:
    ----------
    algorithm : pymoo algorithm object
        The algorithm to be used for optimization.
    problem : pymoo problem object
        The problem to be optimized.
    NIter : int
        Number of iterations for the algorithm.
    NPop : int
        Size of the population for the algorithm.
    algo_name : str, optional
        Custom name for the algorithm (default: algorithm's class name).
    **kwargs : 
        Additional keyword arguments, currently supported:
        - cb: callback function
        - do_g: flag to generate graphs
        - ref_point: reference point for hypervolume calculation (default: problem's nadir point)

    Returns:
    -------
    stats : dict
        Dictionary containing statistics (e.g., HV, GD, IGD) for the run.
    """
    criterion = 'n_gen'
    if 'stop_criterion' in kwargs:
        criterion = kwargs.get('stop_criterion')

    termination = (criterion, NIter)

    res = minimize(problem, algorithm,
                   termination,
                   #seed=1,
                   verbose=False, **kwargs)
    if algo_name is None:
        algo_name = algorithm.__class__.__name__

    ref_point = kwargs.get('ref_point')
    if ref_point is None:
        ref_point = metric_reference_point(problem)

    return show_stats(problem, ref_point, res, algo_name)

class TestAlgorithms:
    """
    A class containing methods to test various multi-objective optimization algorithms.
    
    All test methods share a common signature:
    
    Parameters:
    ----------
    NIter : int
        Number of iterations for the algorithm.
    NPop : int
        Size of the population for the algorithm.
    problem : pymoo problem object
        The problem to be optimized.
    **kwargs : 
        Additional keyword arguments (varies by algorithm, see below for specifics).
    
    Returns:
    -------
    stats : dict
        Dictionary containing statistics (e.g., HV, GD, IGD) for the run.
    """

    @staticmethod
    def get_ref_dirs(problem, n_partitions=None):
        if n_partitions is None:
            return metric_reference_directions(problem.n_obj)
        from pymoo.util.ref_dirs import get_reference_directions

        return get_reference_directions(
            "das-dennis", problem.n_obj, n_partitions=n_partitions
        )

    @staticmethod
    def test_hldbea(NIter, NPop, problem, **kwargs): #Not yet used!
        """
        Tests the HLDBEA algorithm.
        
        Additional kwargs:
        - ref_point: reference point for hypervolume calculation (default: problem's nadir point)
        - Other kwargs are passed to IBEATest4.test
        """
        print("Attention: test_hldbea() used")
        return IBEATest4.test(NIter, NPop, problem, env_sel_method="exact", sel_pop="2")#, cb=IBEATest4.callback)

    @staticmethod
    def test_nsga2(NIter, NPop, problem, **kwargs):
        """
        Tests the NSGA2 algorithm.
        
        Additional kwargs:
        - cb: callback function
        - do_g: flag to generate graphs
        - ref_point: reference point for hypervolume calculation (default: problem's nadir point)
        """
        algorithm = NSGA2(pop_size=NPop, crossover=SBX(eta=20, prob=0.9), mutation=PM(prob=1/problem.n_var, eta=20))
        return run_algorithm(algorithm, problem, NIter, NPop, "NSGA2", **kwargs)

    @staticmethod
    def test_spea2(NIter, NPop, problem, **kwargs):
        """
        Tests the SPEA2 algorithm.
        
        Additional kwargs:
        - cb: callback function
        - do_g: flag to generate graphs
        - ref_point: reference point for hypervolume calculation (default: problem's nadir point)
        """
        algorithm = SPEA2(pop_size=NPop, crossover=SBX(eta=20, prob=0.9), mutation=PM(prob=1/problem.n_var, eta=20))
        return run_algorithm(algorithm, problem, NIter, NPop, "SPEA2", **kwargs)

    @staticmethod
    def test_smsemoa(NIter, NPop, problem, **kwargs):
        """
        Tests the SMSEMOA algorithm.
        
        Additional kwargs:
        - cb: callback function
        - do_g: flag to generate graphs
        - ref_point: reference point for hypervolume calculation (default: problem's nadir point)
        """
        algorithm = SMSEMOA(pop_size=NPop, crossover=SBX(eta=20, prob=0.9), mutation=PM(prob=1/problem.n_var, eta=20))
        return run_algorithm(algorithm, problem, NIter, NPop, "SMSEMOA", **kwargs)

    @staticmethod
    def test_nsga3(NIter, NPop, problem, **kwargs):
        """
        Tests the NSGA-III algorithm.
        
        Additional kwargs:
        - cb: callback function
        - do_g: flag to generate graphs
        - ref_point: reference point for hypervolume calculation (default: problem's nadir point)
        """
        if 'ref_dirs' in kwargs:
            ref_dirs = kwargs.get('ref_dirs')
        else:
            ref_dirs = TestAlgorithms.get_ref_dirs(problem)
        algorithm = NSGA3(pop_size=NPop, ref_dirs=ref_dirs, crossover=SBX(eta=20, prob=0.9), mutation=PM(prob=1/problem.n_var, eta=20))
        return run_algorithm(algorithm, problem, NIter, NPop, "NSGA3", **kwargs)

    @staticmethod
    def test_rvea(NIter, NPop, problem, **kwargs):
        """
        Tests the RVEA algorithm.
        
        Additional kwargs:
        - cb: callback function
        - do_g: flag to generate graphs
        - ref_point: reference point for hypervolume calculation (default: problem's nadir point)
        """
        if 'ref_dirs' in kwargs:
            ref_dirs = kwargs.get('ref_dirs')
        else:
            ref_dirs = TestAlgorithms.get_ref_dirs(problem)
        algorithm = RVEA(pop_size=NPop, ref_dirs=ref_dirs, crossover=SBX(eta=20, prob=0.9), mutation=PM(prob=1/problem.n_var, eta=20))
        return run_algorithm(algorithm, problem, NIter, NPop, "RVEA", **kwargs)

    @staticmethod
    def test_moead(NIter, NPop, problem, **kwargs):
        """
        Tests the MOEA/D algorithm.
        
        Additional kwargs:
        - cb: callback function
        - do_g: flag to generate graphs
        - ref_point: reference point for hypervolume calculation (default: problem's nadir point)
        """
        if 'ref_dirs' in kwargs:
            ref_dirs = kwargs.get('ref_dirs')
        else:
            ref_dirs = TestAlgorithms.get_ref_dirs(problem)
        algorithm = MOEAD(ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7, crossover=SBX(eta=20, prob=0.9), mutation=PM(prob=1/problem.n_var, eta=20))
        return run_algorithm(algorithm, problem, NIter, NPop, "MOEAD", **kwargs)

    @staticmethod
    def test_agemoea(NIter, NPop, problem, **kwargs):
        """
        Tests the AGEMOEA algorithm.
        
        Additional kwargs:
        - cb: callback function
        - do_g: flag to generate graphs
        - ref_point: reference point for hypervolume calculation (default: problem's nadir point)
        """
        from pymoo.algorithms.moo.age import AGEMOEA

        algorithm = AGEMOEA(pop_size=NPop, crossover=SBX(eta=20, prob=0.9), mutation=PM(prob=1/problem.n_var, eta=20))
        return run_algorithm(algorithm, problem, NIter, NPop, "AGEMOEA", **kwargs)
