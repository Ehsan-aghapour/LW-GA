from pymoo.core.repair import Repair

class _MY_Repair(Repair):
    """
    A repair class for minimizing frequency of idle PEs.
    """

    def do(self, problem, pop, **kwargs):
        Host_Configs=['LL','LB','BL','BB']
        Devices={0:{0:[2,3],1:[2],2:{3},3:[]},
         1:{0:[],1:[3],2:[2],3:[2,3]}}
        for ind in pop:
            X=ind.get('X')
            _X=X.copy()
            HostConfig=X[3]
            # if Parts0(L parts) is zero and L is not in Host (Or is host but corspondeing devices(G/N) --> Set L freq to minimum 
            #if X[8]==0 and X[3]==3:
            if X[8]==0:
                host=False
                for D in Devices[0][HostConfig]:
                    if X[8+D] != 0:
                        host=True
                if not host:
                    X[0]=0

            # if Parts1(B parts) is zero and B is not in Host (Or is host but corspondeing devices(G/N) is not active) --> Set B freq to minimum 
            #if X[9]==0 and X[3]==0:
            if X[9]==0:
                host=False
                for D in Devices[1][HostConfig]:
                    if X[8+D] != 0:
                        host=True
                if not host:
                    X[1]=0

            # if Parts2(G parts) is zero --> Set G freq to minimum 
            if X[10]==0:
                X[2]=0

            
            # When There is Component with Part=0; These components move to end of the Order and sort 
            # (To prevent having multiple chromosome for one config)
            _Order=[]
            for C in X[4:8]:
                if X[8+C] != 0:
                    _Order.append(C)
            for C in range(0,4):
                if C not in _Order:
                    _Order.append(C)
            X[4:8]=_Order

            ind.set('X',X)
            
            if (X != _X).any():
                print('Repair')
                print(_X)
                print(X)
                #input()
            
        return pop

'''
xx=self.pop.get('X')
    if len(xx)!=len(np.unique(xx,axis=0)):
        print('There is repetative chromosome')
        input()
'''


