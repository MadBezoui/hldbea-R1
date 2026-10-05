import os
import numpy as np
from hldbea.runtime import configure_headless_matplotlib

configure_headless_matplotlib()
import matplotlib
from matplotlib import pyplot as plt

from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from hldbea.metrics import metric_reference_point, pareto_front_for_metrics
from util.misc import get_caller_function

def save_convergence_curves(problem, all_algo_results, prefix='', generated_path="generated"):
    """
    Generates and saves convergence curves (HV and IGD+) over generations.
    Plots the mean value with a shaded region representing the standard deviation across 30 runs.
    
    :param problem: The pymoo problem instance.
    :param all_algo_results: Dictionary mapping algorithm names to lists of pymoo result objects 
                             e.g., {"NIBEA": [res1...res30], "NSGA2": [res1...res30]}
    """
    output_dir = os.path.join(generated_path, f"{prefix}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except:
        print(f"Found dir {output_dir}...")
    problem_name = problem.name() if hasattr(problem, 'name') else problem.__class__.__name__

    #TODO: Attention, ça doit être le(s) même(s) ref_point(s) utilisé(s) dans les(l') algorithme(s)
    # 1. Initialize Indicators
    ref_point = metric_reference_point(problem)
    pf = pareto_front_for_metrics(problem)
    
    ind_hv = HV(ref_point=ref_point)
    ind_igd_plus = IGDPlus(pf)

    # 2. Create the Figure (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define distinct colors so algorithms match across all your plots
    colors = ["red", "blue", "green", "orange", "purple", "brown"]
    
    for idx, (algo_name, results_list) in enumerate(all_algo_results.items()):
        color = colors[idx % len(colors)]
        
        all_runs_hv = []
        all_runs_igd = []
        
        # 3. Extract history for each run
        for res in results_list:
            if res is None or res.history is None:
                continue
                
            run_hv = []
            run_igd = []
            
            for entry in res.history:
                F = entry.pop.get("F")
                if F is not None:
                    run_hv.append(ind_hv(F))
                    run_igd.append(ind_igd_plus(F))
                else:
                    run_hv.append(0.0)
                    run_igd.append(float('nan'))
                    
            if run_hv:
                all_runs_hv.append(run_hv)
                all_runs_igd.append(run_igd)
        
        if not all_runs_hv:
            print(f"[{algo_name}] Warning: No history found. Ensure save_history=True in minimize().")
            continue
            
        # 4. Align lengths (in case some runs stopped a generation early)
        min_len = min([len(r) for r in all_runs_hv])
        all_runs_hv = np.array([r[:min_len] for r in all_runs_hv])
        all_runs_igd = np.array([r[:min_len] for r in all_runs_igd])
        
        # 5. Calculate Mean and Standard Deviation across the 30 runs
        mean_hv = np.nanmean(all_runs_hv, axis=0)
        std_hv = np.nanstd(all_runs_hv, axis=0)
        
        mean_igd = np.nanmean(all_runs_igd, axis=0)
        std_igd = np.nanstd(all_runs_igd, axis=0)
        
        generations = np.arange(1, min_len + 1)
        
        # 6. Plot HV (Line + Shaded Std Dev)
        ax1.plot(generations, mean_hv, label=algo_name, color=color, linewidth=2)
        ax1.fill_between(generations, mean_hv - std_hv, mean_hv + std_hv, color=color, alpha=0.15)
        
        # 7. Plot IGD+ (Line + Shaded Std Dev)
        ax2.plot(generations, mean_igd, label=algo_name, color=color, linewidth=2)
        ax2.fill_between(generations, mean_igd - std_igd, mean_igd + std_igd, color=color, alpha=0.15)

    # 8. Formatting the HV Subplot
    ax1.set_title(f"Hypervolume Convergence - {problem_name}")
    ax1.set_xlabel("Generations")
    ax1.set_ylabel("Hypervolume (Higher is better)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 9. Formatting the IGD+ Subplot
    ax2.set_title(f"IGD+ Convergence - {problem_name}")
    ax2.set_xlabel("Generations")
    ax2.set_ylabel("IGD+ (Lower is better)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # 10. Save the Figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{problem_name}_convergence_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[{problem_name}] Convergence curves successfully saved to: {save_path}")


def save_boxplots(problem, results, prefix='', generated_path="generated"):
    """
    Generates and saves side-by-side box plots for HV and IGD+ across 30 runs.
    
    :param problem: The pymoo problem instance.
    :param all_algo_results: Dictionary mapping algorithm names to lists of pymoo result objects 
                             e.g., {"NIBEA": [res1...res30], "NSGA2": [res1...res30]}
    """
    assert get_caller_function() == 'run_and_collect', f"save_boxplots() called elsewhere ({get_caller_function()})."

    output_dir = os.path.join(generated_path, f"{prefix}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except:
        print(f"Found dir {output_dir}...")
    problem_name = problem.name() if hasattr(problem, 'name') else problem.__class__.__name__

    algo_names = list(results.keys())
    hv_data = []
    igd_data = []

    # 2. Extract metrics for all 30 runs for each algorithm
    for algo in algo_names:                
        hv_data.append(results[algo]["HV"])
        igd_data.append(results[algo]["IGDPlus"])

    # 3. Create the Figure with 2 Subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 4. Plot HV Boxplot
    # Note: 'showmeans=True' adds a triangle marker for the average, which matches your tables!
    ax1.boxplot(hv_data, labels=algo_names, showmeans=True)
    ax1.set_title(f"Hypervolume (HV) Distribution - {problem_name}")
    ax1.set_ylabel("HV (Higher is better)")
    ax1.tick_params(axis='x', rotation=45) # Rotate labels so they don't overlap
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 5. Plot IGD+ Boxplot
    # We filter out NaNs for IGD+ plotting safely
    clean_igd_data = [[val for val in run if val == val] for run in igd_data] 
    ax2.boxplot(clean_igd_data, labels=algo_names, showmeans=True)
    ax2.set_title(f"IGD+ Distribution - {problem_name}")
    ax2.set_ylabel("IGD+ (Lower is better)")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # 6. Formatting and Saving
    plt.tight_layout() # Ensures labels are not cut off
    save_path = os.path.join(output_dir, f"{problem_name}_boxplots.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[{problem_name}] Box plots successfully saved to: {save_path}")


def plot_median_run(results_list, hvs_list, problem, algo_name, prefix='', generated_path="generated"):
    """
    Identifies and plots the median run based on Hypervolume from a list of pymoo results.

    :param results_list: A list containing the 30 `res` objects from independent runs.
    :param problem: The pymoo problem instance.
    :param algo_name: String name of the algorithm.
    """
    assert get_caller_function() == 'run_and_collect', f"plot_median_run() called elsewhere ({get_caller_function()})."

    output_dir = os.path.join(generated_path, f"{prefix}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except:
        print(f"Found dir {output_dir}...")

    ids = np.argsort(hvs_list)

    # Select the median run (e.g., index 15 out of 30)
    median_idx = ids[len(ids) // 2]

    original_run_number = median_idx
    median_hv = hvs_list[median_idx]
    median_res = results_list[median_idx]

    print(f"Selected run #{original_run_number + 1} as the median. (HV = {median_hv:.5f})")

    # Plot the Pareto front for this specific median run
    problem_name = problem.name() if hasattr(problem, 'name') else problem.__class__.__name__

    plot = Scatter(title=f"{algo_name} on {problem_name} - Median Run")
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7, label="True PF")
    plot.add(median_res.F, facecolor="none", edgecolor="red", label="Median Approx.")

    # Save the plot automatically
    save_path = os.path.join(output_dir, f"{problem_name}_{algo_name}_median.png")
    plot.save(save_path)


def show_perf_graph(problem, res, algo_name):
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD
    indh = HV(ref_point=metric_reference_point(problem))
    indigd = IGD(problem.pareto_front())
    hypervolumes = [indh(entry.pop.get("F")) for entry in res.history]
    igds = [indigd(entry.pop.get("F")) for entry in res.history]
    # Plot the convergence
    plt.plot(hypervolumes, color='blue', label="HV")
    plt.plot(igds, color='red', label="IGD")
    plt.xlabel('Generation')
    plt.ylabel('Indicator')
    plt.title(f"{algo_name}/{problem.name()}: HV and IGD Indicators")
    plt.legend()
    plt.show()
    return np.argmax(hypervolumes), np.argmin(igds)

def show_stats(problem, ref_point, res, algo_name):
    from pymoo.indicators.hv import HV
    indh = HV(ref_point=ref_point)
    from pymoo.indicators.gd import GD
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    pf = pareto_front_for_metrics(problem)
    indgd = GD(pf)
    indigd = IGD(pf)
    indigdplus = IGDPlus(pf)

    if res.opt is not None:
        hv = indh(res.F)
        gd  = indgd(res.F)
        igd = indigd(res.F)
        igdplus = indigdplus(res.F)
        print(f"{algo_name} : Hv = {hv:.{5}f}", f"GD = {gd:.{5}f}", f"IGD = {igd:.{5}f}", f"IGD+ = {igdplus:.{5}f}")
        return (hv, gd, igd, igdplus, res)
    else:
        print("Me - No optimimum found")
    return (0, float('nan'), float('nan'), float('nan'), res)
    #return (None, None, None, None)

def do_graph(problem, res, algo_name):
    F = res.F
    pf = problem.pareto_front()
    if problem.n_obj <= 3:
        plot = Scatter(title=f"{algo_name}, {problem.name()}")
        plot.add(pf, plot_type="line", color="black", alpha=0.7)
        plot.add(F, facecolor="none", edgecolor="red")
        plot.show()
    if problem.n_obj == 4:
        FF = res.history[len(res.history)-1].pop.get("F")
        plot = Scatter(title=f"{algo_name}, {problem.name()}", plot_3d=False, tight_layout=True)
        plot.add(FF, s=20)
        plot.show()
        if False:
            plot = Scatter(title=f"{algo_name}, {problem.name()}, f1 vs f2")#, plot_3d=False)
            plot.add(FF[:, 0:2], s=10)
            plot.show()
            plot = Scatter(title=f"{algo_name}, {problem.name()} f1 vs f3")#, plot_3d=False)
            plot.add(FF[:, [0,2]], s=10)
            plot.show()
            plot = Scatter(title=f"{algo_name}, {problem.name()} f2 vs f3")#, plot_3d=False)
            plot.add(FF[:, [1,2]], s=10)
            plot.show()


def do_iter_graph(problem, res, it, algo_name):
    alg = res.history[it]
    F = alg.pop.get("F")
    pf = problem.pareto_front()
    if problem.n_obj <= 3:
        plot = Scatter(title=f"{algo_name}, {problem.name()}, Iter={it}")
        plot.add(pf, plot_type="line", color="black", alpha=0.7)
        plot.add(F, facecolor="none", edgecolor="red")
    elif problem.n_obj == 4:
        plot = Scatter(title=f"{algo_name}, {problem.name()}, Iter={it}", plot_3d=False, tight_layout=True)
        plot.add(F, s=10)

    if problem.n_obj <= 4:
        plot.show()


class IBEAStats:

    @staticmethod
    def show_final(res, algorithm):
        #print(res.F)
        X = res.X
        F = res.F
        #print("X finale", X)
        #print("F finale", F) # BUG ?
        #print("F finale", algorithm.pop.get("F")[0:4])
        print("F finale:\n", np.array2string(algorithm.pop.get("F")[0:4], precision=2, separator=', ' , suppress_small=True))
        #print("len(pop) finale", len(algorithm.pop))

        #fig = _plotF(F, problem, fig=None, i=0, title='F finale')
        #_plotX(X, fig=fig, i=1, title='X')
        #plt.show()

    @staticmethod
    def stats_1(res):
        print("There is ", len(res.history), " entries in history.\n\n")
        # for each algorithm object in the history
        for entry in res.history:
            print("************ Entry")

            #Fitness array of the whole population
            print("entry Fit:\n", np.array2string(entry.pop.get("Fit"), precision=2, separator=', ' , suppress_small=True))
            if entry.Fmerged_pop is not None:
                print("Fmerged_pop:\n", np.array2string(entry.Fmerged_pop, precision=2, separator=', ' , suppress_small=True))
            # Print F
            print("F:\n", np.array2string(entry.pop.get("F"), precision=2, separator=', ' , suppress_small=True))
            # Print the infills
            if entry.Finfills_archive is not None:
                print("Finfills_archive:\n", np.array2string(entry.Finfills_archive, precision=2, separator=', ' , suppress_small=True))
            #el = np.array(entry.eliminated_tab)
            el = entry.eliminated_tab
            if entry.eliminated_tab is not None:
                print("self.eliminated_tab:\n", np.array2string(el, precision=2, separator=', ' , suppress_small=True))

    # N: If n of gen
    @staticmethod
    def stats_2(res, n_gen, N=10):
        # for each algorithm object in the history
        for entry in res.history:
            if n_gen > N:
                if entry.n_gen <= n_gen - N:
                    continue
            print("************ Entry of gen. n° %s" % entry.n_gen)

            F = entry.pop.get("F")
            # F of eliminated points
            el = entry.eliminated_tab

            sc1 = Scatter(title=("F + infills: Gen %s" % entry.n_gen))
            sc1.add(F, color="blue")

            if entry.Finfills_archive is not None:
                sc1.add(entry.Finfills_archive, color="red", alpha=0.5)

            #if entry.n_gen > 1:
            #    sc1.add(entry.Fmerged_pop[len(entry.pop.get("F")):(2*len(entry.pop.get("F")))], color="red", alpha=0.5)
            sc1.add(entry.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
            sc1.do()
            #sc1.show()


            sc = Scatter(title=("Gen %s" % entry.n_gen))
            #print("History entry: ", dir(entry))
            #print("History entry: output ", str(entry.output))
            sc.add(F, color="blue")
            sc.add(el, color="orange")
            #sc.add(entry.pop.get("X"), color="red", alpha=0.5)
            sc.add(entry.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
            sc.do()
            sc.show()

            del sc1
            del sc

    @staticmethod
    def stats_2a(res, n_gen, N=10):
        pre_pop = None
        #pre_el  = None
        #pre_inf = None
        # for each algorithm object in the history
        for entry in res.history:
            if False:
                if n_gen > N:
                    if entry.n_gen <= n_gen - N:
                        continue
            print("************ Entry of gen. n° %s" % entry.n_gen)

            #entry.pop: pop obtenue après merging et Env Sel(?)
            #elim: non pas de entry.pop mé de la précdt(?)
            if entry.n_gen == 1:
                pre_pop = entry.pop
                continue

            Fp = pre_pop.get("F")
            F = entry.pop.get("F")
            # F of eliminated points
            el = entry.eliminated_tab

            sc1 = Scatter(title=("F + infills: Gen %s" % entry.n_gen))
            sc1.add(Fp, color="blue")

            if entry.Finfills_archive is not None:
                sc1.add(entry.Finfills_archive, color="red", alpha=0.5)

            #if entry.n_gen > 1:
            #    sc1.add(entry.Fmerged_pop[len(entry.pop.get("F")):(2*len(entry.pop.get("F")))], color="red", alpha=0.5)
            sc1.add(entry.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
            sc1.do()
            #sc1.show()


            sc2 = Scatter(title=("Gen %s" % entry.n_gen))
            sc2.add(Fp, color="blue")
            sc2.add(entry.Finfills_archive, color="red", alpha=0.5)
            sc2.add(el, color="yellow", alpha=0.5)
            sc2.add(entry.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
            #sc2.do()


            sc = Scatter(title=("Gen %s" % entry.n_gen))
            sc.add(F, color="blue")
            sc.add(el, color="yellow", alpha=0.5)
            sc.add(entry.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
            sc.do()
            sc.show()

            del sc2
            del sc1
            del sc

            pre_pop = entry.pop

    # Show By 4 subplots. Show 4 generations in one figure.
    @staticmethod
    def stats_3(res, n_gen, N=10):
        n_obj = res.F.shape[1]
        if n_obj == 2:
            fig, axs = plt.subplots(2, 2)
        else:
            fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
        i = 0
        # for each algorithm object in the history
        for entry in res.history:
            if entry.n_gen > 4:
                if n_gen > N:
                    if entry.n_gen <= n_gen - N:
                        continue
            print("************ Entry of gen. n° %s" % entry.n_gen)

            F = entry.pop.get("F")
            pF = entry.problem.pareto_front()
            # Eliminated points in the current generation
            Fel = entry.eliminated_tab
            # Infills points of the current generation
            Finf= entry.Finfills_archive

            # Populate axe n° i / 4 of the figure fig
            ax = fig.axes[i]
            ax.set_title("t = %s" % (entry.n_gen))
            ax.scatter(*F.T, c="blue", marker="o")
            if entry.Finfills_archive is not None:
                ax.scatter(*Finf.T, color="red", alpha=0.5, marker="o")
            if Fel is not None:
                ax.scatter(*Fel.T, c="yellow", marker="v")
            ax.plot(*pF.T, color="black", alpha=0.7)

            if (i+1)%4 == 0:
                plt.tight_layout()
                plt.show()
                # Recreate after closing
                if n_obj == 2:
                    fig, axs = plt.subplots(2, 2)
                else:
                    fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
                #i = (i+1)%5
                i = 0
            else:
                i = i + 1

        pF = res.history[n_gen-1].problem.pareto_front()
        FExact = res.history[n_gen-1].exact_selected_tab
        plot = Scatter(title=f"Exact Elit set")
        plot.add(np.array(FExact), facecolor="none", edgecolor="green")
        plot.add(pF, plot_type="line", color="black", alpha=0.7)
        plot.show()
        FExactGen = res.history[n_gen-1].exact_generated_tab
        plot = Scatter(title=f"Exact generated from Elit points")
        plot.add(np.array(FExactGen), facecolor="none", edgecolor="green")
        plot.add(pF, plot_type="line", color="black", alpha=0.7)
        plot.show()
