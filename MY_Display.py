from pymoo.util.display import MultiObjectiveDisplay


class _MY_Display(MultiObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        #self.output.append("metric_a", np.mean(algorithm.pop.get("X")))
        #self.output.append("metric_b", np.mean(algorithm.pop.get("F")))
        # algorithm.opt[0].F      
        self.output.append("ideal",algorithm.opt.get("F").min(axis=0))
        self.output.append("nadir",algorithm.opt.get("F").max(axis=0))