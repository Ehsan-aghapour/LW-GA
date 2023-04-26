# +
import numpy as np

from geneticalgorithm2 import geneticalgorithm2 as ga # for creating and running optimization model

from matplotlib import pyplot as plt

import numpy as np

import pickle as pkl

import pandas as pd

import os

import sys

from pathlib import Path

sys.path.append('../../Profiling/')
import P_import as P
#Inference_Cost(_graph='alex',_freq=[[0],[1],[2],[3],[4],[5],[6],[7]],_order=8*'B',_dvfs_delay=3.5, _debug=False)
P.Load_Data()


Target_Latency={"alex":300, "google":450, "mobile":350, "res50":900, "squeeze":400}

NLayers={"alex":8, "google":11, "mobile":14, "res50":18, "squeeze":10, "test_transfer":2}


# -

def decode_gene(v):
    if v<6 :
        return "L",[v]
    elif v<14:
        return "B",[v-6]
    elif v<54:
        v2=v-14
        fgpu=v2//8
        fbig=v2%8
        return "G",[fgpu,fbig]
def decoder(chromosome):
    freqs=[]
    ps=''
    for gene in chromosome:
        p,fs=decode_gene(gene)
        ps+=p
        freqs.append(fs)
    return freqs,ps


def run_ga_LW(_g,_target_latency,comp):
    NL=NLayers[_g]
    graph=_g
    target_latency=_target_latency
    
    if comp=='L':
        varbound=np.array([[0,5]]*NL)
    if comp=='B':
        varbound=np.array([[6,13]]*NL)
    if comp=='G':
        varbound=np.array([[14,53]]*NL)
    if comp=='LBG':
        varbound=np.array([[0,53]]*NL)
        
    
    
    
    def f(X):
        config=decoder(X)
        total_time,total_energy=P.Inference_Cost(_graph=graph,_freq=config[0],_order=config[1],_dvfs_delay='variable')
        
        if total_time < target_latency:
            return total_energy
        else:
            return 1000000000

    algorithm_param = {'max_num_iteration': 5,
                       'population_size':200,
                       'mutation_probability': 0.1,
                       'mutation_discrete_probability': None,
                       'elit_ratio': 0.01,
                       'parents_portion': 0.3,
                       'crossover_type':'one_point',
                       'mutation_type': 'uniform_by_center',
                       'mutation_discrete_type': 'uniform_discrete',
                       'selection_type': 'roulette',
                       'max_iteration_without_improv':None}

    model=ga(function=f,
                dimension=NL,
                variable_type=tuple(['int']*NL),
                variable_boundaries=varbound,
                algorithm_parameters=algorithm_param
            )

    filename='Results/'+_g+'_'+comp+'_last_g.npz'
    model.run(no_plot = True,save_last_generation_as = filename)
    with Path('Results/'+_g+'_'+comp+"_report.pkl").open('wb') as ff:
        pkl.dump(model.report,ff)
    '''with Path(_g+"_result.npz").open('wb') as ff:
        pkl.dump(model.result,ff)
    model.run(start_generation=model.result.last_generation)
    '''
    plt.plot(model.report, label = 'local optimizationion')
    plt.title('Score Graph '+str(graph))
    plt.savefig('Results/Score_'+str(graph)+'_'+comp+'.png', bbox_inches="tight")
    plt.clf()

    
    last_generation=model.result.last_generation
    solutions=last_generation.variables
    scores=last_generation.scores
    Freqs=[]
    Order=[]
    sols=[]
    for solution in solutions:
        freq,comps=decoder(solution)
        freq = [[int(x) for x in inner_list] for inner_list in freq]
        freq=[tuple(f) for f in freq]
        Freqs.append(tuple(freq))
        Order.append(comps)
        sols.append(solution)
    df = pd.DataFrame({
        'graph':graph,
        'order': Order,
        'freq': Freqs,
        'score': scores
    })
    # Save the DataFrame as a CSV file
    df.to_csv('Results/ga_result_'+str(graph)+'_'+comp+'.csv', index=False)
    return model


# +
def main():
    global model_alex,model_google,model_mobile,model_res50,model_squeeze
    os.makedirs("Results", exist_ok=True)
    l=300
    model_alex=run_ga_LW(_g='alex',_target_latency=l,comp='L')
    model_alex=run_ga_LW(_g='alex',_target_latency=l,comp='B')
    model_alex=run_ga_LW(_g='alex',_target_latency=l,comp='G')
    model_alex=run_ga_LW(_g='alex',_target_latency=l,comp='LBG')
    #model_alex.run(start_generation='Results/alex_last_g.npz')


main()


# -

def test():
    g='alex'
    config=decoder([53,53,53,53,53,53,53,13])
    freqs=config[0]
    freqs=tuple([tuple(f) for f in freqs])
    
    print(config)
    t,e=P.Inference_Cost(_graph=g,_freq=config[0],_order=config[1])
    print(g,config[1],str(freqs),e,t)
test()


