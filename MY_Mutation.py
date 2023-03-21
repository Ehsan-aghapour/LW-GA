import numpy as np

from pymoo.core.mutation import Mutation


class _MY_Mutation(Mutation):

	def __init__(self, prob=None,ProbFreqMutation=None,ProbHostMutation=None,ProbOrderMutation=None,ProbPartsMutation=None):
		super().__init__()
		self.prob = prob
		self.ProbFreqMutation=ProbFreqMutation
		self.ProbHostMutation=ProbHostMutation
		self.ProbOrderMutation=ProbOrderMutation
		self.ProbPartsMutation=ProbPartsMutation


	def _do(self, problem, X, **kwargs):
		if self.prob is None:
			self.prob = 1.0 / problem.n_var

		X = X.astype(np.bool)
		_X = np.full(X.shape, np.inf)

		M = np.random.random(X.shape)
		flip, no_flip = M < self.prob, M >= self.prob

		_X[flip] = (X[flip] + np.random.randint(0, 18)) % 19
        
		_X[no_flip] = X[no_flip]

		return _X.astype(np.bool)




