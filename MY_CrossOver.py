
import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.util import crossover_mask
import pickle

class _MY_UniformCrossover(Crossover):

    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, _, X, **kwargs):
        _, n_matings, n_var = X.shape
        M = np.random.random((n_matings, n_var)) < 0.5
        print(f'M shape:{M.shape} and X shape:{X.shape}')
        #2*50*12

        # 1:
        for m in M:
            m[7]=m[6]=m[5]=m[4]
            m[11]=m[10]=m[9]=m[8]

        # 2:
        '''for m in M:
            m[7]=m[6]=m[5]=m[4]=False
            m[11]=m[10]=m[9]=m[8]=True'''
        #print(f'M is: {M}')
        _X = crossover_mask(X, M)
        print(f'New X shape:{_X.shape}')
        #t=input("ssss")

        return _X


#class UX(_MY_UniformCrossover):
#    pass


if __name__ == '__main__':
    cc=_MY_UniformCrossover()
    #with open('ff.pkl','rb') as f:
    #    data=pickle.load(f)
    #X=data['X']
    #X=[[[0,1,2,3,4,5,6,7,8,9,10,11,12],[20,21,22,23,24,25,26,27,28,29,30,31]],
    #[[40,41,42,43,44,45,46,47,48,49,50,51],[60,61,62,63,64,65,66,67,68,69,70,71]]]
    #X=np.array(X)
    X=np.random.randint(0,50,(2,4,12))
    
    print(X.shape)
    off=cc._do(X,X=X)

    print(X)
    
    print(off)