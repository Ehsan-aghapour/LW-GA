from tkinter import N
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.misc import repair
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback

# from pymoo.termination import get_termination


import numpy as np
import sys
import pickle
from pymoo.core.problem import ElementwiseProblem

from pymoo.util.display import Display

from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
#from pymoo.termination.default import DefaultMultiObjectiveTermination


import time
import pickle

import sys
#sys.stdout = open('somefile.txt', 'w')
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

from pymoo.util.running_metric import RunningMetric
import os


from MY_CrossOver import _MY_UniformCrossover
from MY_Mutation import _MY_Mutation
from MY_Mutation import _MY_Mutation2
from MY_Sampling import _MY_SumFixSampling
from MY_Repair import _MY_Repair
from MY_Profile import Eval
from MY_Profile import set_parameters
from MY_Profile import save_ProfResult

from MY_Display import _MY_Display
from MY_Callback import _MY_Callback


# +
_Ns={'Alex':8,
    'Google':11,
    'Mobile':14,
    'Squeeze':10,
    'Res50':18,}
cmpfreq(v){
    if v<6 :
        return "L",[v]
    else if v<14:
        return "B",[v-6]
    else if v<54:
        v2=v-14
        fgpu=v2/8
        fbig=v2%8
        return "GPU",[fgpu,fbig]
  
#transfer[g][layer][c_dest][c_source]
#data[g][c][f][fbig][layer][m][t/power]
datafiles=["10/data10.pkl","100/data100.pkl"]
data={}
for d in datafiles:
    with open(d,'rb') as f:
        j=pkl.load(f) 
    data.update(j)
    
transfers={}
with open("transfers.pkl",'rb') as f:
    transfers=pkl.load(f)


# -

class MyProblem(ElementwiseProblem):
    g=0
    n=0
    n_eval=0
    #fp=0
    def __init__(self,_graph,Target_Latency):
        targetlatency=Target_Latency
        g=_graph
        n=_Ns[_graph]
        print("Initialize the problem for graph with " + str(n) + " layers.")
        _xl=np.full(n,0)
        _xu=np.full(n,18)
        super().__init__(n_var=n,
                         n_obj=1,
                         n_constr=1,
                         xl=np.array(_xl),
                         xu=np.array(_xu))

        

    def _evaluate(self, x, out, *args, **kwargs):
        C=[]
        F=[]
        for i in x:
            c,f=cmpfreq(i)
            C.append(c)
            F.append(f)
        l=0
        energy=0
        ################## Tasks
        for i,c in enumerate(C):
            d=0
            if c=="G":
                d=data[g][c][F[i][0]][F[i][1]][i]["task"]
            else:
                d=data[g][c][F[i][0]][i]["task"]
            t=d["time"]
            p=d["power"]
            e=p*t
            latency+=t
            energy+=e
            
        ################## Input
        loader=C[0]
        if loader=="GPU":
            t_input=data[g][loader][F[0][0]][F[0][1]][0]["in"]["time"]
            p_input=data[g][loader][F[0][0]][F[0][1]][0]["in"]["power"]
        else:
            t_input=data[g][loader][F[0][0]][0]["in"]["time"]
            p_input=data[g][loader][F[0][0]][0]["in"]["power"]
            
        e_input=t_input*p_input
        latency+=t_input
        energy+=e_input
        
        ################### transfers
        for i in range(1,len(C)):
            if C[i]!=C[i-1]:
                transfer=transfer[g][i][C[i]][C[i-1]]
                transfer_time=transfer
                #approximate:
                transfer_energy=0
                if C[i]=="G"
                    transfer_energy=1.2*(data[g][C[i]][F[i][0]][F[i][1]][i]["in"]["power"]*transfer_time)
                else:
                    transfer_energy=1.2*(data[g][C[i]][F[i][0]][i]["in"]["power"]*transfer_time)
                latency+=transfer_time
                energy+=transfer_energy
                    
        
        #data[g][c][f][fbig][layer][m][t/power] : (m coud be task, in, out, and trans)
        #tranfer[g][layer][c_source][c_dest]
        G1=latency - targetlatency
        out["F"] = [energy]
        out["G"] = [G1]


Trm=200

#https://pymoo.org/interface/termination.html
_termination = MultiObjectiveDefaultTermination(
    #x_tol=1e-8,
    #cv_tol=1e-6,
    f_tol=0.001,
    nth_gen=5,
    n_last=Trm,
    n_max_gen=Trm,
    n_max_evals=Trm*100,
)
#__termination = get_termination("n_gen", 50)
'''termination = DefaultMultiObjectiveTermination(
    xtol=1e-8,
    cvtol=1e-6,
    ftol=0.0025,
    period=30,
    n_max_gen=1000,
    n_max_evals=100000
)'''



def main(_graph='Alex',TargetLatency):
    #graph='Res50'
    graph=_graph
    N=_NS[graph]
    stime=time.time_ns()
    problem = MyProblem(graph,TargetLatency)
    
    _pop_size=1000
    set_parameters(_Graph=graph)
    algorithm = NSGA2(pop_size=_pop_size,
        sampling=int_random(),
        selection=TournamentSelection(func_comp=binary_tournament),
        mutation=_MY_Mutation(1/N),
        crossover=_MY_UniformCrossover(prob=0.5),
    
    
        repair=_MY_Repair(),
    
        eliminate_duplicates=True,
        #n_offsprings=None,
        #display=MultiObjectiveDisplay(),
        )

    res = minimize(problem,
                algorithm,
                #('n_gen', 200),
                seed=1,
                verbose=True,
                return_least_infeasible=True,
                termination=_termination,
                save_history=True,
                display=_MY_Display(),
                callback=_MY_Callback(delta_gen=2,
                        n_plots=4,
                        #only_if_n_plots=True,
                        #do_close=False,
                        key_press=False,
                        do_show=False),
                )
    etime=time.time_ns()
    print(f'GA finished. Duration: {(etime-stime)/10**9} s')
    save_ProfResult()
    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()
    return res

if __name__ == "__main__":
    main()

