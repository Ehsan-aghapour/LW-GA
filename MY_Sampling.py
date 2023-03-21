import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.util.normalization import denormalize


def random_by_bounds(n_var, xl, xu, n_samples=1):
	val = np.random.random((n_samples, n_var))
	return denormalize(val, xl, xu)


class int_random(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        val = np.random.randint(problem.xl,problem.xu,size=(n_samples, problem.n_var))
        return val


def random(problem, n_samples=1):
	return random_by_bounds(problem.n_var, problem.xl, problem.xu, n_samples=n_samples)


class FloatRandomSampling(Sampling):
	"""
	Randomly sample points in the real space by considering the lower and upper bounds of the problem.
	"""

	def __init__(self, var_type=np.float64) -> None:
		super().__init__()
		self.var_type = var_type

	def _do(self, problem, n_samples, **kwargs):
		return random(problem, n_samples=n_samples)


class BinaryRandomSampling(Sampling):

	def _do(self, problem, n_samples, **kwargs):
		val = np.random.random((n_samples, problem.n_var))
		return (val < 0.5).astype(bool)


class PermutationRandomSampling(Sampling):

	def _do(self, problem, n_samples, **kwargs):
		X = np.full((n_samples, problem.n_var), 0, dtype=int)
		for i in range(n_samples):
			X[i, :] = np.random.permutation(problem.n_var)
		return X





############################### Decode Chromosome #######################################################
def MY_decode_my_coding_to_chromosome(chromosome, N):
	#check gene range
	for i,gene in enumerate(chromosome):
		if gene+i > N+1:
			print("Error invalid gene value (not in range)")
			return
	#decode, reference order is LBGN
	Ref_Order='LBGN'
	M=list(range(N+1))
	#Partition={'L':0, 'B':0, 'G':0, 'N':0}
	Partition={}
	for i,gene in enumerate(chromosome):
		if gene<len(M):
			Partition[Ref_Order[i]]=M[gene]
			M.pop(gene)
	#sort the dict based on partition value
	Partition=dict(sorted(Partition.items(), key=lambda item: item[1]))

	Order=''
	End_Points=[]
	for comp in Partition:
		if Partition[comp]>N:
			break
		Order=Order+comp
		End_Points.append(Partition[comp])
	
	# Repair 
	End_Points[-1]=N
	#print(f'Order: {Order}, End_Points: {End_Points}')
	Rest=set('LBGN').difference(Order)
	for c in Rest:
		Order=Order+c
		End_Points.append(N)
	P=0
	Parts=[]
	for i in range(len(End_Points)):
		Parts.append(End_Points[i]-P)
		P=End_Points[i]
	_Order=[Ref_Order.index(c) for c in Order]
	#print(f'Order: {Order}, _Order: {_Order}, Parts: {Parts}, _Order+Parts: {_Order+Parts}')
	return (_Order+Parts)
###########################################################################################################

def MY_random_by_bounds(n_var, xl, xu, n_samples=1):
	#Freqs and Host_core 
	n_var1=4
	xl1=xl[:4]
	xu1=xu[:4]
	val1 = np.random.random((n_samples, n_var1))
	#print(val1)
	val1 = denormalize(val1, xl1, xu1).astype(int)
	#print(val1)
	# Partitioning and mapping
	n_var2=4
	N=xu[-1]-xl[-1]
	xl2=[1,1,1,1]
	xu2=[N+1,N,N-1,N-2]
	val2 = np.random.random((n_samples, n_var2))
	#print(val2)
	val2 = denormalize(val2, xl2, xu2).astype(int)
	#print(val2)
	val=[list(val1[i])+MY_decode_my_coding_to_chromosome(val2[i],N) for i in range(len(val2))]
	#print(val)
	return np.array(val)

def MY_random(problem, n_samples=1):
	return MY_random_by_bounds(problem.n_var, problem.xl, problem.xu, n_samples=n_samples)

class _MY_SumFixSampling(Sampling):
	def GenerateParts(self,N,S,U,L,PEs):
		#### SubMethod1 ##########
		Parts=[]
		#print(f'N:{N}, S:{S}, U:{U}, L:{L}, PEs:{PEs}')
		for i in range(PEs-1):
			PartitionPoint=np.random.randint(L,U+1)
			Parts.append(PartitionPoint-L)
			L=PartitionPoint
		Parts.append(S-sum(Parts))
		np.random.shuffle(Parts)
		#print(Parts)
		#### Method2 ##########
		#Parts=[]
		#while sum(Parts) != S: Parts = np.random.randint(L, U, size=(PEs,))

		return Parts

	def _do(self, problem, n_samples, **kwargs):
		############# Method 1 ###########
		#return random(problem, n_samples=n_samples)
		##################################

		############# Method 2 ###########
		# N = Number of layers:
		N=problem.xu[-1]-problem.xl[-1]
		# Sum of Parts:
		S=N
		# Upper and Lower bound of Partition
		U=N
		L=0
		# Number of PEs
		PEs = 4
		#print(f'N is {N}')
		Order = np.full((n_samples, 4), 0, dtype=int)
		Parts = np.full((n_samples, 4), 0, dtype=int)
		#OrderParts = np.full((n_samples, 8), 0, dtype=int)
		for i in range(n_samples):
			Order[i, :] = np.random.permutation(4)
			Parts[i, :] = self.GenerateParts(N,S,U,L,PEs)
			#OrderParts[i, :] = Order[i] + Parts[i] 

		
		_xl=problem.xl[0:4]
		_xu=problem.xu[0:4]
		FreqHost = np.random.randint(_xl, _xu, size=(n_samples,len(_xl)))
		t=np.column_stack((FreqHost, Order, Parts))
		#print(f"eva shape is:{t.shape}")
		for tt in t:
			if any(ttt < 0 for ttt in tt):
				input("evaaa")
		return np.column_stack((FreqHost, Order, Parts))
		##################################
