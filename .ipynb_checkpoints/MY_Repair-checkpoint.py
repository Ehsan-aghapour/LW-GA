from pymoo.core.repair import Repair

class _MY_Repair(Repair):
    """
    A repair class for minimizing frequency of idle PEs.
    """

    def do(self, problem, pop, **kwargs):
        
        for ind in pop:
            X=ind.get('X')
            
            
        return pop

'''
xx=self.pop.get('X')
    if len(xx)!=len(np.unique(xx,axis=0)):
        print('There is repetative chromosome')
        input()
'''


