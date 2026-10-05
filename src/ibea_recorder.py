from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
from pymoo.visualization.scatter import Scatter

import os
import sys

# Set the correct plugin path early (before matplotlib or cv2)
try:
    from PyQt5.QtCore import QLibraryInfo
    qt_plugins = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins
except ImportError:
    pass   # No PyQt5 installed – nothing to fix



class IBEARecord:

    @staticmethod
    def record_video(res, filename, step=1):
        print(f"Generating the video {filename}...")
        writer = Video(filename)

        # use the video writer as a resource
        with Recorder(writer) as rec:

            # for each algorithm object in the history
            for entry in res.history:
                if entry.n_iter % step != 0:
                    continue

                sc = Scatter(title=("Gen %s" % entry.n_gen))
                #print("History entry: ", dir(entry))
                #print("History entry: output ", str(entry.output))
                #if entry.n_iter % 10 == 0:
                #    print("Iter ", entry.n_iter)
                    #print("entry Fit: ", entry.pop.get("Fit"))
                sc.add(entry.pop.get("F"), color="blue")
                #sc.add(entry.pop.get("X"), color="red", alpha=0.5)
                sc.add(entry.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
                sc.do()

                # finally record the current visualization to the video
                rec.record()

                del sc

    @staticmethod
    def record_video_1(res, filename):
        writer = Video(filename)

        # use the video writer as a resource
        with Recorder(writer) as rec:

            # for each algorithm object in the history
            for entry in res.history:

                pF = entry.problem.pareto_front()
                # F points eliminated
                Fel = entry.eliminated_tab

                sc = Scatter(title=("Gen %s" % entry.n_gen))
                sc.add(Fel, color="yellow")
                sc.add(entry.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
                sc.do()

                # finally record the current visualization to the video
                rec.record()

                del sc

    @staticmethod
    def record_video_2(res, g=0, step=10, filename=None):
        writer = Video(filename)

        # use the video writer as a resource
        with Recorder(writer) as rec:

            entry_0 = res.history[g]
            entry = res.history[g+1]

            pF = entry.problem.pareto_front()
            # F points eliminated
            Fel = entry.eliminated_tab
            if Fel is not None:
                Finf= entry.Finfills_archive
                j = 0
                for i in range(len(Fel)):
                    sc = Scatter(title=("Gen %s - point %s" % (entry.n_gen, i)))
                    j = i if i%step == 0 else j
                    sc.add(entry_0.pop.get("F"), color="blue", alpha=0.5)
                    sc.add(Finf, color="red", alpha=0.5)
                    sc.add(Fel[0:j], color="green")
                    sc.add(Fel[i, :], color="yellow", marker="v")
                    sc.add(entry.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
                    sc.do()

                    rec.record()

                    del sc

                F = entry.pop.get("F")
                sc = Scatter(title=("Gen %s - point %s" % (entry.n_gen, i)))
                sc.add(F, color="blue", alpha=0.5)
                sc.do()
                rec.record()
                del sc
