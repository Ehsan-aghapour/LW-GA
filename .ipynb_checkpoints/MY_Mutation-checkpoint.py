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

		'''for k in kwargs:
			print(f'value for {k}:{kwargs[k]}')
		print(kwargs['algorithm'].n_gen)

		ProbFreqMutation=kwargs.get('ProbFreqMutation',None)
		ProbHostMutation=kwargs.get('ProbHostMutation',None)
		ProbOrderMutation=kwargs.get('ProbOrderMutation',None)
		ProbPartsMutation=kwargs.get('ProbPartsMutation',None)'''
		Algorithm=kwargs.get('algorithm',None)
		Generation=Algorithm.n_gen
		

		M = np.random.random(X.shape)
		print(f'Mutation M shape: {M.shape}')
		'''M[:,0:3] = M[:,0:3] < self.ProbFreqMutation
		M[:,3:4] = M[:,3:4] < self.ProbHostMutation
		M[:,4:8] = M[:,4:8] < self.ProbOrderMutation
		M[:,8:12] = M[:,8:12] < self.ProbPartsMutation'''

		#for Freq in X[0:3]:
		#    if np.random.random() < ProbFreqMutation:
		
		_xl=problem.xl[:]
		_xu=problem.xu[:]
		

		def MutateFreq(x,xl,xu):
			_x=np.array(x[:])
			M=np.random.random(len(x)) < self.ProbFreqMutation
			#Range=np.random.randint(_xl-x,_xu-x)
			print(f'Freq Mutation x:{x} \n and Range:{xu-xl}')
			#print(f'Mutation x Range:{xu-xl}')
			dev=(xu-xl)/(4+(Generation/20))
			print(f'Mutate freq, Dev is:{dev}')
			print(f'x is {x} \nnew x 95% is between {x-(2*dev)} and {x+(2*dev)}')
			#print(f'Mutate freq, x is:{x}')
			x=np.random.normal(x,dev).round()
			print(f'Freq Mutation new x:{x} \n M is {M}')
			#t=input('Next...')
			x=np.minimum(x,xu)
			x=np.maximum(x,xl)
			_x[M]=x[M]
			print(f'Freq Mutation new x:{_x}')
			return _x

		def MutateHost(x,xl,xu):
			_x=x
			M=np.random.random() < self.ProbHostMutation
			if M :
				#Range=np.random.randint(_xl-x,_xu-x)
				dev=(xu-xl)/(4+(Generation/20))
				print(f'Host Mutation x is {x}, Range is {xu-xl} and dev is {dev}')
				print(f'x is {x} \nnew x 95% is between {x-(2*dev)} and {x+(2*dev)}')
				x=np.round(np.random.normal(x,dev))
				x=np.minimum(x,xu)
				x=np.maximum(x,xl)
				_x=x
				print(f'Host Mutation new x is {x}')
			return _x

		def MutateScheduling(x):
			_order=np.array(x[:-4])
			_parts=np.array(x[-4:])
			if np.random.random() < self.ProbOrderMutation :
				print(f'Order mutation, old order: {_order}')
				i,j=np.random.choice(range(len(_order)), size=2, replace=False)
				_order[i],_order[j]=_order[j],_order[i]
				print(f'New order:{_order}')
				return np.concatenate((_order,_parts))
				#_parts[i],_parts[j]=_parts[j],_parts[i]
			if np.random.random() < self.ProbPartsMutation :
				i,j=np.random.choice(range(len(_parts)), size=2, replace=False)
				dev=_parts[j]/(4+(Generation/20))
				print(f'Part mutation dev is: {dev}')
				print(f'x is {x} \nnew x 95% is between {x-(2*dev)} and {x+(2*dev)}')
				_p=abs(np.round(np.random.normal(0,dev)))
				_p=np.minimum(_p,_parts[j])
				_p=np.maximum(_p,-_parts[j])
				print(f'Part mutation parts is: {_parts}, shifting {_p} partiotions from {_parts[j]} to {_parts[i]}')
				_parts[i]=_parts[i]+_p
				_parts[j]=_parts[j]-_p
				print(f'Part mutation new parts is: {_parts}')
				

			return np.concatenate((_order,_parts))

		_x=X.reshape(-1,X.shape[-1])
		print(f'Mutation X shape: {X.shape}')
		print(f'Mutation _x shape: {_x.shape}')
		#t=input('Mutation ...')
		Mutate_happened=False
		total=0
		mutated=0
		mutations=0
		CCF=0
		CCH=0
		CCO=0
		CCP=0
		for ii,x in enumerate(_x):
			#print(f'x is: {x}')
			t=x.copy()
			print('\n\n***********\n')
			x[0:3]=MutateFreq(x[0:3],_xl[0:3],_xu[0:3])
			x[3]=MutateHost(x[3],_xl[3],_xu[3])
			x[4:]=MutateScheduling(x[4:])
			print(f'X befor mutation: {t}')
			print(f'X after mutation: {x}')
			if (t!=x).any():
				mutated=mutated+1
			#mutations=mutations+np.sum(t!=x)
			
			CF=np.sum(x[:3]!=t[:3])
			CH=np.sum(x[3:4]!=t[3:4])
			CO=int((x[4:8]!=t[4:8]).any())
			CP=int((x[8:12]!=t[8:12]).any())
			v=CF+CH+CO+CP
			print(f'Number of changes in mutation: {v} --> {CF},{CH},{CO},{CP}\n****************\n')
			CCF=CCF+CF
			CCH=CCH+CH
			CCO=CCO+CO
			CCP=CCP+CP
			mutations=mutations + v
			total=total+1

		print(f'mutation rate: {100*mutated/total}%, mutated={mutated} and total={total}')
		print(f'mutation rate: {mutations/mutated}, mutated={mutations} and mutated={mutated}')
		print(f'Mutation rate, Freqs: {100*CCF/total}, Host:{100*CCH/total}, Order:{100*CCO/total}, Parts:{100*CCP/total}')		
		#input('press to continue ...')	
		return _x


		'''      
		X = X.astype(np.bool)
		_X = np.full(X.shape, np.inf)

		M = np.random.random(X.shape)
		flip, no_flip = M < self.prob, M >= self.prob

		_X[flip] = np.logical_not(X[flip])
		_X[no_flip] = X[no_flip]

		return _X.astype(np.bool)'''



class _MY_Mutation2(Mutation):

	def __init__(self, prob=None,ProbFreqMutation=2/6,ProbHostMutation=1/6,ProbOrderMutation=1/6,ProbPartsMutation=2/6):
		super().__init__()
		self.prob = prob
		self.ProbFreqMutation=ProbFreqMutation
		self.ProbHostMutation=ProbHostMutation
		self.ProbOrderMutation=ProbOrderMutation
		self.ProbPartsMutation=ProbPartsMutation


	def _do(self, problem, X, **kwargs):
		if self.prob is None:
			self.prob = 1.0 / problem.n_var

		'''for k in kwargs:
			print(f'value for {k}:{kwargs[k]}')
		print(kwargs['algorithm'].n_gen)

		ProbFreqMutation=kwargs.get('ProbFreqMutation',None)
		ProbHostMutation=kwargs.get('ProbHostMutation',None)
		ProbOrderMutation=kwargs.get('ProbOrderMutation',None)
		ProbPartsMutation=kwargs.get('ProbPartsMutation',None)'''
		Algorithm=kwargs.get('algorithm',None)
		Generation=Algorithm.n_gen
		

		M = np.random.random(X.shape)
		print(f'Mutation M shape: {M.shape}')
		M[:,0:3] = M[:,0:3] < self.ProbFreqMutation
		M[:,3:4] = M[:,3:4] < self.ProbHostMutation
		M[:,4:8] = M[:,4:8] < self.ProbOrderMutation
		M[:,8:12] = M[:,8:12] < self.ProbPartsMutation

		#for Freq in X[0:3]:
		#    if np.random.random() < ProbFreqMutation:
		
		_xl=problem.xl[:]
		_xu=problem.xu[:]
		

		def MutateFreq(x,xl,xu):
			_x=np.array(x[:])
			ind=np.random.choice(list(range(len(_x))))
			Range=xu[ind]-xl[ind]
			dev=Range/(4+(Generation/20))
			cc=0
			_dev=dev
			while _x[ind]==x[ind]:
				cc=cc+1
				if cc%1000==0:
					_dev=_dev+dev
				_x[ind]=round(np.random.normal(_x[ind],_dev))
				_x[ind]=np.minimum(_x[ind],xu[ind])
				_x[ind]=np.maximum(_x[ind],xl[ind])

			print(f'Freq Mutation x:{x} \nRange:{xu-xl}\nTarget: {_x[ind]}')
			print(f'Mutate freq, Dev is:{dev}')
			print(f'x is {x} \nnew x 95% is between {x-(2*dev)} and {x+(2*dev)}')
			print(f'Freq Mutation new x:{_x} after {cc} samples')
			return _x

		def MutateHost(x,xl,xu):
			_x=x
			dev=(xu-xl)/(4+(Generation/20))
			print(f'Host Mutation x is {x}, Range is {xu-xl} and dev is {dev}')
			print(f'x is {x} \nnew x 95% is between {x-(2*dev)} and {x+(2*dev)}')
			cc=0
			_dev=dev
			while _x==x:
				cc=cc+1
				if cc%1000==0:
					_dev=_dev+dev
				_x=np.round(np.random.normal(x,_dev))
				_x=np.minimum(_x,xu)
				_x=np.maximum(_x,xl)
				
			print(f'Host Mutation new x is {_x} after {cc} samples')
			return _x

		def MutateOrder(x):
			_order=np.array(x[:-4])
			#_parts=np.array(x[-4:])
			print(f'Order mutation, old order: {_order}')
			i,j=np.random.choice(range(len(_order)), size=2, replace=False)
			_order[i],_order[j]=_order[j],_order[i]
			print(f'New order:{_order}')
			#return np.concatenate((_order,_parts))
			return _order

		def MutateParts(x):
			#_order=np.array(x[:-4])
			_parts=np.array(x[-4:])

			i,j=np.random.choice(range(len(_parts)), size=2, replace=False)
			# if parts[j]==0 then dev=0 and normal rand in next loop always produce mean which 0 so next loop never end
			while(_parts[j]==0):
				i,j=np.random.choice(range(len(_parts)), size=2, replace=False)

			dev=_parts[j]/(4+(Generation/20))
			print(f'Part mutation dev is: {dev}')
			print(f'x is {x} \nnew x 95% is between {x-(2*dev)} and {x+(2*dev)}')
			
			cc=0
			#while (_parts==x[-4:]).all():
			_p=0
			_dev=dev
			while(_p==0):
				cc=cc+1
				if cc%1000==0:
					_dev=_dev+dev
				_p=abs(np.round(np.random.normal(0,_dev)))

			_p=np.minimum(_p,_parts[j])
			_p=np.maximum(_p,-_parts[j])
			print(f'Part mutation parts is: {_parts}, shifting {_p} partiotions from {_parts[j]} to {_parts[i]}')
			_parts[i]=_parts[i]+_p
			_parts[j]=_parts[j]-_p
			print(f'Part mutation new parts is: {_parts} after {cc} samples and (_dev/dev={_dev/dev})')
			
			#return np.concatenate((_order,_parts))
			return _parts

		
				

		_x=X.reshape(-1,X.shape[-1])
		print(f'Mutation X shape: {X.shape}')
		print(f'Mutation _x shape: {_x.shape}')
		#t=input('Mutation ...')
		Mutate_happened=False
		total=0
		mutated=0
		mutations=0
		CCF=0
		CCH=0
		CCO=0
		CCP=0
		RF=0
		RH=0
		RO=0
		RP=0
		CntF=0
		CntH=0
		CntO=0
		CntP=0
		for ii,x in enumerate(_x):
			#print(f'x is: {x}')
			t=x.copy()
			p=[self.ProbFreqMutation,self.ProbHostMutation,self.ProbOrderMutation,self.ProbPartsMutation]
			print(f'Mutation probability for choice parts Freq, Host, Order, Parts\n{p}')
			s=np.random.choice([0,1,2,3],1,p=[self.ProbFreqMutation,self.ProbHostMutation,self.ProbOrderMutation,self.ProbPartsMutation])
			print(f'\n\n***********\nS={s}')
			if s==0:
				x[0:3]=MutateFreq(x[0:3],_xl[0:3],_xu[0:3])
				CntF=CntF+1
			if s==1:
				x[3]=MutateHost(x[3],_xl[3],_xu[3])
				CntH=CntH+1
			if s==2:
				x[4:8]=MutateOrder(x[4:])
				CntO=CntO+1
			if s==3:
				x[8:12]=MutateParts(x[8:])
				CntP=CntP+1


			print(f'X befor mutation: {t}')
			print(f'X after mutation: {x}')
			if (t!=x).any():
				mutated=mutated+1
			#mutations=mutations+np.sum(t!=x)
			
			CF=np.sum(x[:3]!=t[:3])
			CH=np.sum(x[3:4]!=t[3:4])
			CO=int((x[4:8]!=t[4:8]).any())
			CP=int((x[8:12]!=t[8:12]).any())
			v=CF+CH+CO+CP
			print(f'Number of changes in mutation: {v} --> {CF},{CH},{CO},{CP}\n****************\n')
			CCF=CCF+CF
			CCH=CCH+CH
			CCO=CCO+CO
			CCP=CCP+CP

			RF=RF+sum(abs(x[0:3]-t[0:3]))
			RH=RH+sum(abs(x[3:4]-t[3:4]))
			RO=RO+sum(abs(x[4:8]-t[4:8]))
			RP=RP+sum(abs(x[8:12]-t[8:12]))

			mutations=mutations + v
			total=total+1

		print('\n\n\n\n\n\n\n\n-------------------------------------------------------------\n------------------------------------------------------------------\n')
		print(f'mutation rate: {100*mutated/total}%, mutated={mutated} and total={total}')
		print(f'mutation rate: {mutations/mutated}, mutated={mutations} and mutated={mutated}')
		print(f'Mutation rate, Freqs: {100*CCF/total}, Host:{100*CCH/total}, Order:{100*CCO/total}, Parts:{100*CCP/total}')	
		print(f'Mutataion F:{RF/CntF}  H:{RH/CntH}  O:{RO/CntO}  P:{RP/CntP}')
		print(f'Mutataion F:{CntF}  H:{CntH}  O:{CntO}  P:{CntP}')	
		#input('press to continue ...')	
		print('\n--------------------------------------------------------------------\n-----------------------------------------------------------------\n\n\n\n\n\n\n')
		return _x


		'''      
		X = X.astype(np.bool)
		_X = np.full(X.shape, np.inf)

		M = np.random.random(X.shape)
		flip, no_flip = M < self.prob, M >= self.prob

		_X[flip] = np.logical_not(X[flip])
		_X[no_flip] = X[no_flip]

		return _X.astype(np.bool)'''
