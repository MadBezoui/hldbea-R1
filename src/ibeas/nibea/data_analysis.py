import numpy as np
import pandas as pd
from scipy import stats
#import seaborn as sns

TEX_OUTPUT_DIR="./output"

def prepare_data(data: dict) -> pd.DataFrame:
    """
    Converts a dictionary of algorithm results into a Pandas DataFrame suitable for statistical analysis and visualization.

    Parameters:
    ----------
    data : dict
        Dictionary with problem names as keys and dictionaries of algorithm results as values.
        Algorithm result dictionaries have metric names (e.g., 'HV', 'GD', 'IGD') as keys and lists of sampled values as values.

    Returns:
    -------
    df : pandas.DataFrame
        DataFrame with columns 'Problem', 'Algorithm', 'Metric', and 'Value'.
        Each row represents a single measurement of an algorithm on a problem and metric.
    """
    data_list = []

    data_list = [
        {
            'Problem': prob,
            'Algorithm': alg,
            'Metric': metric,
            'Value': value
        }
        for prob, algos in data.items()
        for alg, metrics in algos.items()
        for metric, values in metrics.items()
        for value in values
    ]
    return pd.DataFrame(data_list)

def get_wilcoxon_output(metric_algo1, metric_algo2, compare):
    """
    compare = '+' : higher is better
    compare = '-' : lower is better
    """
    stat, p_value = stats.mannwhitneyu(metric_algo1, metric_algo2, alternative='two-sided')
    #print("Mann-Whitney U Test Statistic:", stat)
    if p_value >= 0.05:
        # Pas de preuve suffisante pour dire qu'ils sont différents
        return ('=', p_value)
    else:
        # Ils sont différents ! Mais qui est le meilleur ?
        # On regarde simplement les moyennes pour trancher
        if np.mean(metric_algo1) > np.mean(metric_algo2):
            return (compare, p_value) #HV: -, Autres: +
        else:
            return ('-' if compare=='+' else '+', p_value)

def calculate_summary_stats(df, wr):
    """
    Calculates and displays summary statistics (mean and standard deviation) for the results in the provided DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'Problem', 'Algorithm', 'Metric', and 'Value', as produced by `prepare_data()`.
    wr : as returned by `perform_wilcoxon_tests()`.

    Returns:
    -------
    None
    """
    # Calculate summary statistics: aggregate on each metric
    summary_stats = df.groupby(['Problem', 'Algorithm', 'Metric']).agg({'Value': ['mean', 'std']}).reset_index()
    summary_stats.columns = ['Problem', 'Algorithm', 'Metric', 'Mean', 'Std']
    #
    #summary_stats.to_csv(f"ignore/output/results_summary.csv", index=False, header=True)
    summary_stats[summary_stats['Metric']=='HV'].to_csv(f"output/hv_raw.csv", index=False, header=True)
    summary_stats[summary_stats['Metric']=='IGDPlus'].to_csv(f"output/igdplus_raw.csv", index=False, header=True)
    #
    # Display the summary statistics
    if False:
        print(summary_stats[summary_stats['Metric']=='HV'])
        print(summary_stats[summary_stats['Metric']=='IGD'])
        print(summary_stats[summary_stats['Metric']=='IGDPlus'])

    stats_hv = summary_stats[summary_stats['Metric']=='HV']
    stats_igdp = summary_stats[summary_stats['Metric']=='IGDPlus']
    if wr:
        #print(wr)
        df_wr = pd.DataFrame(wr)
        ss_hv = df_wr[df_wr['Metric']=='HV']
        ss_igdp = df_wr[df_wr['Metric']=='IGDPlus']

        stats_hv['Result'] = stats_hv.apply(lambda row: f"{row['Mean']:.4f} ({row['Std']:.2e})", axis=1)
        stats_hv = pd.merge(stats_hv, ss_hv, on=['Problem', 'Algorithm', 'Metric'])
        stats_hv['Result'] = stats_hv.apply(lambda row: f"{row['Result']}{row['Sign']}", axis=1)

        stats_igdp['Result'] = stats_igdp.apply(lambda row: f"{row['Mean']:.4f} ({row['Std']:.2e})", axis=1)
        stats_igdp = pd.merge(stats_igdp, ss_igdp, on=['Problem', 'Algorithm', 'Metric'])
        stats_igdp['Result'] = stats_igdp.apply(lambda row: f"{row['Result']}{row['Sign']}", axis=1)
    else:
        #ss['Result'] = ss['Mean'].astype(str) + '(' + ss['Std'].astype(str) + ')'
        stats_hv['Result'] = stats_hv.apply(lambda row: f"{row['Mean']:.4f} ({row['Std']:.2e})", axis=1)
        stats_igdp['Result'] = stats_igdp.apply(lambda row: f"{row['Mean']:.4f} ({row['Std']:.2e})", axis=1)
    stats_hv = stats_hv[['Problem', 'Algorithm', 'Result']]
    stats_igdp = stats_igdp[['Problem', 'Algorithm', 'Result']]

    #print(ss[ss['Metric']=='HV'])
    #print(ss[ss['Metric']=='IGD'])
    #print(ss[ss['Metric']=='IGDPlus'])
    print(stats_hv)
    print(stats_igdp)

    pivot_table_hv = stats_hv.pivot(index='Problem', columns=['Algorithm'], values='Result')
    print(pivot_table_hv)
    latex_hv_code = pivot_table_hv.to_latex(
            escape=False, # IMPORTANT : pour que \textbf{} fonctionne
            column_format='l' + 'c'*len(pivot_table_hv.columns), # ex: lcccccc
            caption="HV Performance Comparison (Mean and Standard Deviation). Best results are in bold.",
            label="tab:results_hv"
        )
    with open(f"{TEX_OUTPUT_DIR}/table_hv.tex", "w") as f:
            f.write("%Generated by calculate_summary_stats()\n\n")
            f.write(latex_hv_code)

    pivot_table_igdp = stats_igdp.pivot(index='Problem', columns=['Algorithm'], values='Result')
    print(pivot_table_igdp)
    latex_igdp_code = pivot_table_igdp.to_latex(
            escape=False, # IMPORTANT : pour que \textbf{} fonctionne
            column_format='l' + 'c'*len(pivot_table_igdp.columns), # ex: lcccccc
            caption="IGD+ Performance Comparison (Mean and Standard Deviation). Best results are in bold.",
            label="tab:results_igdp"
        )
    with open(f"{TEX_OUTPUT_DIR}/table_igdp.tex", "w") as f:
            f.write("%Generated by calculate_summary_stats()\n\n")
            f.write(latex_igdp_code)


    # Pivot the table to get a more readable format
    #pivot_table = summary_stats[summary_stats['Metric']=='HV'].pivot(index='Problem', columns=['Algorithm', 'Metric'], values='Mean')
    #print(pivot_table)
    #pivot_table.to_csv(f"ignore/output/results_summary_pivottable_hv.csv", sep='&', index=True, header=True)



    # Box plots for HV and IGD
    #plt.figure(figsize=(12, 6))
    #sns.boxplot(x='Problem', y='Value', hue='Algorithm', data=df[df['Metric'] == 'HV'])
    #plt.title('Box Plot of Hypervolume (HV) by Problem and Algorithm')
    #plt.show()

    # Bar charts with error bars
    #plt.figure(figsize=(12, 6))
    #sns.barplot(x='Problem', y='Value', hue='Algorithm', data=df[df['Metric'] == 'HV'], capsize=0.1)
    #plt.title('Bar Plot of Hypervolume (HV) by Problem and Algorithm')
    #plt.show()

def perform_wilcoxon_tests(data, Probs, **kwargs):
    """
    Performs Wilcoxon signed-rank tests to compare the results of different algorithms on the specified problems.

    Parameters:
    ----------
    data : dict
        Dictionary with problem names as keys and dictionaries of algorithm results as values.
        Algorithm result dictionaries have metric names (e.g., 'HV', 'GD', 'IGD') as keys and lists of sampled values as values.
    Probs : list
        List of problem names for which to perform the Wilcoxon tests.

    Returns:
    -------
    None
    """
    #https://datagy.io/mann-whitney-u-test-python/
    #https://www.geeksforgeeks.org/machine-learning/mann-whitney-u-test-2/

    algorithms = kwargs.get('algorithms')
    wtests = {'Problem':[], 'Algorithm':[], 'Metric':[], 'Sign':[]}
    for pb_name in Probs:
        pb_name = pb_name if isinstance(pb_name, tuple) else (pb_name,)
        pb_name = pb_name[0]
        #HLDBEA vs Itself (HV)
        wtests['Problem'].extend([pb_name])
        wtests['Algorithm'].extend(['HLDBEA'])
        wtests['Metric'].extend(['HV'])
        wtests['Sign'].extend([' '])
        #HLDBEA vs Itself (IGDPlus)
        wtests['Problem'].extend([pb_name])
        wtests['Algorithm'].extend(['HLDBEA'])
        wtests['Metric'].extend(['IGDPlus'])
        wtests['Sign'].extend([' '])

        hldbea_hv = np.array(data[pb_name]['HLDBEA']['HV'])
        hldbea_igdp = np.array(data[pb_name]['HLDBEA']['IGDPlus'])
        for alg, val in algorithms.items():
            if val and (alg != 'HLDBEA'):

                #HV
                wtests['Problem'].extend([pb_name])
                wtests['Algorithm'].extend([alg])
                wtests['Metric'].extend(['HV'])
                alg_hv = np.array(data[pb_name][alg]['HV'])

                try:
                    sign, p_value = get_wilcoxon_output(hldbea_hv, alg_hv, '-')
                    #print(f"{alg} - {pb_name}, p-value={p_value:.3f}: (sign)")
                    wtests['Sign'].extend([sign])

                except ValueError:
                    print(f"Error for {pb_name}.")

                #IGDPlus
                wtests['Problem'].extend([pb_name])
                wtests['Algorithm'].extend([alg])
                wtests['Metric'].extend(['IGDPlus'])
                alg_igdp = np.array(data[pb_name][alg]['IGDPlus'])

                try:
                    sign, p_value = get_wilcoxon_output(hldbea_igdp, alg_igdp, '+')
                    #print(f"{alg} - {pb_name}, p-value={p_value:.3f}: (sign)")
                    wtests['Sign'].extend([sign])

                except ValueError:
                    print(f"Error for {pb_name}.")

    return wtests


