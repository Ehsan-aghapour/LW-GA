import matplotlib.pyplot as plt
import numpy as np

from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTerminationWithRenormalization
from pymoo.visualization.video.callback_video import AnimationCallback

import threading
import os
import pickle

from pymoo.indicators.hv import Hypervolume



class RunningMetric(AnimationCallback):

    def __init__(self,
                 delta_gen,
                 n_plots=4,
                 only_if_n_plots=False,
                 key_press=True,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        print('parent init')
        self.delta_gen = delta_gen
        self.key_press = key_press
        self.only_if_n_plots = only_if_n_plots
        self.term = MultiObjectiveSpaceToleranceTerminationWithRenormalization(n_last=100000,
                                                                               all_to_current=True,
                                                                               sliding_window=False)
        self.hist = []
        self.n_plots = n_plots



    '''def notify(self, algorithm, **kwargs):
        if algorithm.n_gen == 1 or algorithm.n_gen % self.nth_gen == 0:
            try:

                figure = self.do(algorithm.problem, algorithm, **kwargs)
                if self.do_show:
                    if figure is not None:
                        figure.show()
                    else:
                        plt.show()

                if self.recorder is not None:
                    self.recorder.record(fig=figure)

                if self.do_close:
                    plt.close(fig=figure)

                return figure

            except Exception as ex:
                if self.exception_if_not_applicable:
                    raise ex'''

    #def do(self, _, algorithm, force_plot=False, **kwargs):
    #    self._do( _, algorithm, **kwargs)
    def do(self, _, algorithm, force_plot=False, **kwargs):
        self.term.do_continue(algorithm)

        metric = self.term.get_metric()
        metrics = self.term.metrics
        tau = len(metrics)

        fig = None
        '''self.pp.append(self.ii)
        self.ii=self.ii+1
        plt.plot(self.pp)
        plt.pause(10)
        return'''
        if metric is not None and (tau + 1) % self.delta_gen == 0 or force_plot:

            _delta_f = metric["delta_f"]
            _delta_ideal = [m['delta_ideal'] > 0.005 for m in metrics]
            _delta_nadir = [m['delta_nadir'] > 0.005 for m in metrics]

            if force_plot or not self.only_if_n_plots or (self.only_if_n_plots and len(self.hist) == self.n_plots - 1):

                fig, ax = plt.subplots()

                if self.key_press:
                    def press(event):
                        if event.key == 'q':
                            algorithm.termination.force_termination = True

                    fig.canvas.mpl_connect('key_press_event', press)

                for k, f in self.hist:
                    ax.plot(np.arange(len(f)) + 1, f, label="t=%s" % (k+1), alpha=0.6, linewidth=3)
                ax.plot(np.arange(len(_delta_f)) + 1, _delta_f, label="t=%s (*)" % (tau+1), alpha=0.9, linewidth=3)

                for k in range(len(_delta_ideal)):
                    if _delta_ideal[k] or _delta_nadir[k]:
                        ax.plot([k+1, k+1], [0, _delta_f[k]], color="black", linewidth=0.5, alpha=0.5)
                        ax.plot([k+1], [_delta_f[k]], "o", color="black", alpha=0.5, markersize=2)

                ax.set_yscale("symlog")
                ax.legend()

                ax.set_xlabel("Generation")
                ax.set_ylabel("$\Delta \, f$", rotation=0)

                if self.key_press:
                    plt.draw()
                    plt.waitforbuttonpress()
                    plt.close('all')

            self.hist.append((tau, _delta_f))
            self.hist = self.hist[-(self.n_plots - 1):]
            plt.tight_layout()
            plt.savefig('plots/RM'+str(algorithm.n_gen)+'.png', bbox_inches="tight")
            with open ('plots/RM'+str(algorithm.n_gen)+'.pkl', 'wb') as _f:
                pickle.dump(fig, _f)
        #plt.show()
        #plt.pause(2)
        #return fig

    


class _MY_Callback(RunningMetric):
    def __init__(self,
                 delta_gen,
                 n_plots=4,
                 only_if_n_plots=False,
                 key_press=True,
                 **kwargs) -> None:

        
        super().__init__(delta_gen=delta_gen,
            n_plots=n_plots,
            only_if_n_plots=only_if_n_plots,
            key_press=key_press,
            **kwargs)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('checkpoint', exist_ok=True)
        self.data["best"] = []
        self.mydata={"n_evals":[],"Min":[],"Mean":[],"hv":[]}
        self.hv = Hypervolume(ref_point= np.array([1.2, 1.2]),
                     norm_ref_point=False)
    def do(self, _, algorithm, force_plot=False, **kwargs):
        print('Calling Running Metric Callback')
        super().do( _, algorithm, force_plot=False, **kwargs)

        self.data["best"].append(algorithm.pop.get("F").min(axis=0))
        self.mydata["n_evals"].append(algorithm.evaluator.n_eval)
        self.mydata["Min"].append(algorithm.pop.get("F").min(axis=0))
        self.mydata["Mean"].append(algorithm.pop.get("F").mean(axis=0))
        self.mydata["hv"].append(self.hv.do(algorithm.pop.get("F")))

        plt.clf()
        plt.plot(self.mydata["Min"],label=["Min RFPS","Min_Energy"])
        plt.legend()
        
        plt.xlabel("Generation")
        plt.ylabel("Normalized")
        plt.savefig('plots/Min'+str(algorithm.n_gen)+'.png', bbox_inches="tight")

        plt.clf()
        plt.plot(self.mydata["Mean"],label=["Mean RFPS","Mean_Energy"])
        plt.legend()
        plt.xlabel("Generation")
        plt.ylabel("Normalized")
        plt.savefig('plots/Mean'+str(algorithm.n_gen)+'.png', bbox_inches="tight")

        plt.clf()
        plt.plot(self.mydata["hv"],label="Hypervolume")
        plt.legend()
        plt.xlabel("Generation")
        #plt.ylabel("Normalized")
        plt.savefig('plots/hv'+str(algorithm.n_gen)+'.png', bbox_inches="tight")

        np.save("checkpoint/checkpoint"+str(algorithm.n_gen), algorithm)
        with open('checkpoint/GAResult_' + str(algorithm.n_gen) + '.pkl','wb') as pkf:
            pickle.dump(algorithm.result(),pkf)
        
        if os.path.isfile("checkpoint/checkpoint"+str(algorithm.n_gen-1)+ '.npy'):
            os.remove("checkpoint/checkpoint"+str(algorithm.n_gen-1)+ '.npy')
        if os.path.isfile('checkpoint/GAResult_' + str(algorithm.n_gen-1) + '.pkl'):
            os.remove('checkpoint/GAResult_' + str(algorithm.n_gen-1) + '.pkl')



