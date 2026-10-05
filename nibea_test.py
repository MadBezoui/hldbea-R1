import argparse
import sys

sys.path.append('./src')

from hldbea.runtime import configure_headless_matplotlib

configure_headless_matplotlib()

import yaml

from pymoo.config import Config
Config.warnings['not_compiled'] = False

from pymoo.problems import get_problem

from ibeas.nibea.ibea_test_4 import IBEATest4
from ibeas.nibea.data_analysis import prepare_data, calculate_summary_stats, perform_wilcoxon_tests
from ibeas.nibea.algo_execution import TestAlgorithms
from ibea_stats import plot_median_run, save_boxplots, save_convergence_curves, do_iter_graph, show_perf_graph, \
    IBEAStats
from hldbea.metrics import metric_reference_point

global_verbose = False

RESULTS_OUTPUT_DIR="generated"

def test_pb_by_algos(NIter=100, NPop=100, problem=None, algos=None, ref_point=None, **kwargs):
    """
    Tests multiple algorithms on a given problem and returns their results.

    Parameters:
    ----------
    NIter : int, optional
        Number of iterations for the algorithms (default: 100).
    NPop : int, optional
        Size of the population for the algorithms (default: 100).
    problem : pymoo problem object, optional
        The problem to be optimized.
    algos : dict, optional
        Dictionary with algorithm names as keys and boolean flags as values (default: None).
        Supported algorithms: 'HLDBEA', 'NSGA2', 'SPEA2', 'SMSEMOA'
    ref_point : list, optional
        Reference point for hypervolume calculation (default: problem's nadir point).

    Returns:
    -------
    results : list
        List of dictionaries containing statistics (e.g., HV, GD, IGD) for each algorithm.
    """
    if global_verbose:
        print("*** PROBLEM INFOS: ***\n", problem)
    else:
        print(f"*** PROBLEM : {problem.name()} ***")

    if ref_point is None:
        ref_point = metric_reference_point(problem)

    res = []
    if algos["HLDBEA"]:
        a = IBEATest4.test(NIter, NPop, problem, ref_point, K=1, **kwargs)
        res.append(a)
    baseline_kwargs = {**kwargs, "ref_point": ref_point}
    if algos["NSGA2"]:
        a = TestAlgorithms.test_nsga2(NIter, NPop, problem, **baseline_kwargs)
        res.append(a)
    if algos["SPEA2"]:
        a = TestAlgorithms.test_spea2(NIter, NPop, problem, **baseline_kwargs)
        res.append(a)
    if algos["NSGA3"]:
        a = TestAlgorithms.test_nsga3(NIter, NPop, problem, **baseline_kwargs)
        res.append(a)
    if algos["RVEA"]:
        a = TestAlgorithms.test_rvea(NIter, NPop, problem, **baseline_kwargs)
        res.append(a)
    if algos["MOEAD"]:
        a = TestAlgorithms.test_moead(NIter, NPop, problem, **baseline_kwargs)
        res.append(a)
    if algos["AGEMOEA"]:
        a = TestAlgorithms.test_agemoea(NIter, NPop, problem, **baseline_kwargs)
        res.append(a)
    #a = TestAlgorithms.test_smsemoa(NIter, NPop, problem, **kwargs)#, cb=IBEARecord.record_video)
    return res

def run_and_collect(NIter=100, NPop=100, problem=None, sp=2, **kwargs):
    """
    Runs multiple algorithms on a given problem for a specified number of samples and collects their results.

    Parameters:
    ----------
    NIter : int, optional
        Number of iterations for the algorithms (default: 100).
    NPop : int, optional
        Size of the population for the algorithms (default: 100).
    problem : pymoo problem object, optional
        The problem to be optimized.
    sp : int, optional
        Number of samples (i.e., independent runs) to collect results for (default: 5).

    Returns:
    -------
    results : dict
        Dictionary with algorithm names as keys and dictionaries of metric values as values.
        Metric dictionaries have metric names (e.g., 'HV', 'GD', 'IGD') as keys and lists of sampled values as values.
    """
    algorithms = kwargs.get('algorithms')
    if not algorithms:
        algorithms = {"HLDBEA":True, "NSGA2":True, "SPEA2":True, "NSGA3":True, "RVEA":True, "MOEAD":True, "AGEMOEA":True}#, "SMSEMOA":False}
    metrics = ["HV", "GD", "IGD","IGDPlus"]
    results = {alg: {metric: [] for metric in metrics} for alg, val in algorithms.items() if val}
    res_by_algos = {alg: [] for alg, val in algorithms.items() if val}

    for s in range(sp):
        print(f"Run n° {s+1:{2}} / {sp}")
        algo_res = test_pb_by_algos(NIter, NPop, problem, algorithms, seed=s, **kwargs)
        #TODO: if there is something to do after each iteration, do it here.
        i = 0
        for alg, _ in results.items():
            results[alg]["HV"].append(algo_res[i][0])
            results[alg]["GD"].append(algo_res[i][1])
            results[alg]["IGD"].append(algo_res[i][2])
            results[alg]["IGDPlus"].append(algo_res[i][3])
            res_by_algos[alg].append(algo_res[i][4])
            i += 1

    if 'do_g' in kwargs and kwargs.get('do_g'):
        print("GENERATING AND SAVING FIGURES...")
        save_boxplots(problem, results, kwargs.get('exp_name', ''), f"{RESULTS_OUTPUT_DIR}")
        #TODO Attention, Trop gourmand en ressources!
        #save_convergence_curves(problem, res_by_algos, kwargs.get('exp_name', ''), f"{RESULTS_OUTPUT_DIR}")
        for alg, _ in results.items():
            plot_median_run(res_by_algos[alg], results[alg]["HV"], problem, alg, kwargs.get('exp_name', ''), f"{RESULTS_OUTPUT_DIR}")

    return results

def do_algos_stats(probs_config, n_samp=2, Probs=None, **kwargs):
    """
    Orchestrates the comparison of multiple algorithms on a set of problems, collecting and analyzing their results.

    Parameters:
    ----------
    NIter : int, optional
        Number of iterations for the algorithms (default: 100).
    NPop : int, optional
        Size of the population for the algorithms (default: 100).
    n_samp : int, optional
        Number of samples (i.e., independent runs) to collect results for (default: 2).
    Probs : list, optional
        List of problem names to include in the comparison (default: a predefined list of problems).

    Returns:
    -------
    None
    """
    if Probs is None:
        #Error: you must give the problems...
        return
    else:
        ALL_TEST_PROBS = Probs


    print("Test problems:", str(ALL_TEST_PROBS))
    #[print(pb_name) for pb_name in ALL_TEST_PROBS]

    if len(ALL_TEST_PROBS) > 1 and isinstance(probs_config, tuple):
        probs_config = [probs_config] * len(ALL_TEST_PROBS)

    data = {str(pb_name[0]): run_and_collect(*probs_config[i], get_problem(*pb_name if isinstance(pb_name, tuple) else (pb_name,)), n_samp, **kwargs) for i, pb_name in enumerate(ALL_TEST_PROBS)}

    # Convert the list of dictionaries to a DataFrame
    df = prepare_data(data)

    # Display the summary statistics
    #print(df)
    #df.to_csv(f"output/results.csv", index=False, header=False)

    if n_samp >= 2:
        wr = perform_wilcoxon_tests(data, ALL_TEST_PROBS, **kwargs)

        calculate_summary_stats(df, wr)
    #print(f"POP:{NPop}\nITER:{NIter}\nSAMPLE:{n_samp}")
    print(f"{probs_config}\nSAMPLE:{n_samp}")
    print(f"**kwargs: {kwargs}")









def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_interactive_menu(args, problem, res):
    """
    Interactive menu to explore the results archive (res.history).
    """
    history = res.history
    if history is None:
        print("Error: History was not saved. Ensure 'save_history' is True.")
        return

    n_gen_max = len(history) - 1

    # Initialize indicators if possible
    # Note: For IGD, you typically need the Pareto Front of the problem
    pf = problem.pareto_front()

    while True:
        print(f"\n--- POST-EXECUTION EXPLORATION (Gens: 1 to {n_gen_max+1}) ---")
        print("1. Generate Scatter Plot for a specific generation")
        print("2. Calculate Indicator (HV/IGD) for a specific generation")
        print("3. Generate Evolution Video")
        print("4. Show Convergence Curve (HV and IGD+)")
        print("5. Generate Scatter Plot using subplot")
        print("q. Quit")

        choice = input("\nSelect an option: ").strip().lower()

        if choice == 'q':
            break

        if choice in ['1', '2']:
            try:
                idx = int(input(f"Enter generation index (1-{n_gen_max+1}): ")) - 1
                if 0 <= idx <= n_gen_max:
                    algo_at_gen = history[idx]
                    # Extract the objective values of the population at this generation
                    F = algo_at_gen.pop.get("F")

                    if choice == '1':
                        print(f"Generating scatter plot for generation {idx+1}...")
                        do_iter_graph(problem, res, idx, 'HLDBEA')

                    elif choice == '2':
                        print(f"\n--- Indicators for Generation {idx+1} ---")
                        # 1. Hypervolume
                        def get_hv(F, problem):
                            from pymoo.indicators.hv import HV
                            metric_hv = HV(ref_point=problem.nadir_point())
                            return metric_hv(F)
                        print(f"Hypervolume: {get_hv(F, problem):.4f}")

                        # 2. IGD (if Pareto Front is available)
                        def get_igd(F):
                            from pymoo.indicators.igd import IGD
                            metric_igd = IGD(pf)
                            return metric_igd(F)
                        if pf is not None:
                            print(f"IGD:         {get_igd(F):.4f}")
                        else:
                            print("IGD:         Not available (Pareto Front unknown)")
                else:
                    print("Index out of bounds.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        elif choice == '3':
            print("Compiling video from history...")
            filename = IBEATest4.construct_filename(args.n_gen, args.n_pop, res.history[0], problem, 1)
            IBEATest4.generate_video(res, filename, 1) #Without the extension

        elif choice == '4':
            print("Plotting convergence...")
            show_perf_graph(problem, res, 'HLDBEA')

        elif choice == '5':
            print("Plotting convergence...")
            IBEAStats.stats_3(res, args.n_gen, 8)
        else:
            continue


def run_single_test(args):
    """Runs the algorithm once and opens the interactive explorer."""
    print(f"--- Single Test Mode ---")
    try:
        # Load only the necessary params for the test
        params_raw = load_yaml(args.params)
        algo_params = {**params_raw['fitness'], **params_raw['infill'], **params_raw['env_selection'],
                       **params_raw['general']}

        print(f"Using parameters from: {args.params}")

        if args.pb.upper().startswith("ZDT"):
            # ZDT only takes name and n_var
            problem = get_problem(args.pb, n_var=args.n_var)
        else:
            # DTLZ and others take name, n_var, and n_obj
            problem = get_problem(args.pb, args.n_var, args.n_obj)

        # In interactive mode, we disable auto-outputs to handle them manually later
        do_g = args.do_g if not args.interactive else False
        do_v = args.do_v if not args.interactive else False

        print(f"Executing {args.pb} with pop size {args.n_pop} for  {args.n_gen} generations...")
        reference = metric_reference_point(problem)
        *perf_indicators, res = IBEATest4.test(
            args.n_gen, args.n_pop,
            problem,
            reference,
            K=1,
            seed=1,
            do_g=do_g,
            do_v=do_v,
            save_history=args.save_history,
            **algo_params
        )

        if args.interactive:
            res.indicators = perf_indicators
            run_interactive_menu(args, problem, res)

    except FileNotFoundError:
        print(f"Error: Configuration file '{args.params}' not found.")


def main():
    parser = argparse.ArgumentParser(description="Run HLDBEA Optimization Experiments")

    # Arguments for the three config types
    parser.add_argument("--profile", type=str, default="configs/profile_debug.yaml", help="Path to profile yaml")
    parser.add_argument("--params", type=str, default="configs/params_standard.yaml", help="Path to params yaml")
    parser.add_argument("--algos", type=str, default="configs/algos.yaml", help="Path to algos yaml")
    parser.add_argument("--out", type=str, default="results", help="Suffix for output files")
    # Test-Single & Interactive Logic
    parser.add_argument("--test-single", action="store_true", help="Run a single HLDBEA benchmark run")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive exploration after test")
    # Problem & Algo Setup
    parser.add_argument("--pb", type=str, default="ZDT1", help="Problem name for single test (e.g., ZDT1, DTLZ2)")
    parser.add_argument("--n_var", type=int, default=12, help="Number of variables for the problem")
    parser.add_argument("--n_obj", type=int, default=2, help="Number of objectives for the problem")
    parser.add_argument("--n_gen", type=int, default=250, help="Number of generations/iterations")
    parser.add_argument("--n_pop", type=int, default=100, help="Population size")
    # Visualization and Video flags
    parser.add_argument("--do_g", action="store_true", help="Plot or save figures/plots")
    parser.add_argument("--do_v", action="store_true", help="Generate video of the population evolution")
    parser.add_argument("--save-history", action="store_true", help="Retain every generation in memory")

    args = parser.parse_args()
    # 1. Handle "test-single" immediately
    if args.test_single:
        run_single_test(args)
        return  # Exit to prevent running the full suite

    # 1. Load Configs
    profile = load_yaml(args.profile)
    params_raw = load_yaml(args.params)
    algos = load_yaml(args.algos)

    # 2. Flatten Params for the function
    # Merges fitness, infill, and env_selection into one dict
    algo_params = {**params_raw['fitness'], **params_raw['infill'], **params_raw['env_selection'], **params_raw['general']}
    
    # 3. Prepare Problem Config
    num_probs = len(profile['probs'])
    p_config = profile['probs_config']

    test_params = {"stop_criterion": profile['stop_criterion'], "save_history": False}

    # 4. Execute
    print(f"Launching Experiment: {profile['name']} for {profile['n_samp']} runs...")
    do_algos_stats(
        probs_config=[tuple(p) for p in p_config],
        n_samp=profile['n_samp'],
        Probs=[tuple(p) for p in profile['probs']],
        algorithms=algos,
        exp_name=profile["name"], #args.out, # Ensure do_algos_stats uses this for naming files
        **test_params,
        **algo_params
    )

if __name__ == "__main__":
    main()
