from ibeas.ibea_t import IBEATest
from ibea_stats import IBEAStats, show_stats, do_graph, show_perf_graph, do_iter_graph
from hldbea.metrics import metric_reference_point


# See ibea_advance_4.py
class IBEATest4:

    @staticmethod
    def construct_filename(NIter, POP, algorithm, problem, step):
        return f"generated/nibea{problem.name()}({problem.n_var},{problem.n_obj})[temp_pop{algorithm.temp_pop}][sel_pop{algorithm.sel_pop}][env{algorithm.env_sel_method}][sc_method{algorithm.sc_method}]({NIter},{POP}){algorithm.ibea_advance.adv_name()}[step{step}]"

    @staticmethod
    def generate_video(res, filename, step):
          from ibea_recorder import IBEARecord
          IBEARecord.record_video(res, filename=filename+'.mp4', step=step)
          #IBEARecord.record_video_1(res, filename=filename+'-1.mp4')
          #IBEARecord.record_video_2(res, g=1, step=10, filename=filename+'-2.mp4')

    @staticmethod
    def test(NIter, POP, problem, ref_point=None, K=None, **kwargs):

        if K is None:
            K = 1

        if ref_point is not None:
            kwargs.setdefault("metric_ref_point", ref_point)
        res, algorithm, n_gen = IBEATest.ibea_4(problem=problem, K=K, pop_size=POP, n_gen=NIter, **kwargs)

        #
        do_g = kwargs.get('do_g')
        if do_g:
            #do_graph(problem, res, 'HLDBEA')
            do_iter_graph(problem, res, n_gen - 3, 'HLDBEA')
            do_iter_graph(problem, res, n_gen - 2, 'HLDBEA')
            do_iter_graph(problem, res, n_gen - 1, 'HLDBEA')
            #IBEAStats.stats_3(res, n_gen)
            show_perf_graph(problem, res, 'HLDBEA')
        #
        do_v = kwargs.get('do_v')
        if do_v:
            stp = 1
            filename = IBEATest4.construct_filename(NIter, POP, algorithm, problem, stp)
            IBEATest4.generate_video(res, filename, stp) #Without the extension

        print("algorithm.count_sel_pop = ", algorithm.count_sel_pop)
        #print("Exact points: ", algorithm.exact_generated_tab)

        if ref_point is None:
            ref_point = metric_reference_point(problem)

        return show_stats(problem, ref_point, res, "HLDBEA")
        #plt.show()
