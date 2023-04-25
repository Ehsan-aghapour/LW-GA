# +
import numpy as np

from geneticalgorithm2 import geneticalgorithm2 as ga # for creating and running optimization model

from geneticalgorithm2 import Generation, AlgorithmParams#, MiddleCallbackData # classes for comfortable parameters setting and getting

from geneticalgorithm2 import Crossover, Mutations, Selection # classes for specific mutation and crossover behavior

from geneticalgorithm2 import Population_initializer # for creating better start population

from geneticalgorithm2 import np_lru_cache # for cache function (if u want)

from geneticalgorithm2 import plot_pop_scores # for plotting population scores, if u want

from geneticalgorithm2 import Callbacks # simple callbacks (will be deprecated)

from geneticalgorithm2 import Actions, ActionConditions, MiddleCallbacks # middle callbacks

import matplotlib.pyplot as plt

import numpy as np
import pickle as pkl
#from geneticalgorithm import geneticalgorithm as ga
#from genetic_algorithm import GeneticAlgorithm as GA
import pandas as pd
import os
import sys
from pathlib import Path
sys.path.append('../Profiling/')

import P
#Inference_Cost(_graph='alex',_freq=[[0],[1],[2],[3],[4],[5],[6],[7]],_order=8*'B',_dvfs_delay=3.5, _debug=False)
P.Load_Data()


Target_Latency={"alex":300, "google":450, "mobile":350, "res50":900, "squeeze":400}

NLayers={"alex":8, "google":11, "mobile":14, "res50":18, "squeeze":10, "test_transfer":2}


'''
explore_freq=Path('freq_tranfer.csv').resolve()
NFreqs={"L":6, "B":8, "G":5}
Metrics=["in","task","out","trans"]
n=100
params={"alex":(1,1,1), "google":(2,2,1), "mobile":(2,3,1), "res50":(2,4,1), "squeeze":(1,5,1), "test_transfer":(1,0,0)}
data={}
df=None
df2=None
%config Completer.use_jedi = False'''


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


def run_ga(_g='alex',_target_latency=Target_Latency):
    NL=NLayers[_g]
    varbound=np.array([(0,53)]*NL)
    graph=_g
    target_latency=Target_Latency[_g]
    
    
    def f(X):
        config=decoder(X)
        total_time,total_energy=P.Inference_Cost(_graph=graph,_freq=config[0],_order=config[1])
        
        if total_time < target_latency:
            return total_energy
        else:
            return 1000000000

    algorithm_param = {'max_num_iteration': 3000,
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
                variable_type='int',
                variable_boundaries=varbound,
                algorithm_parameters=algorithm_param
            )

    filename='Results/'+_g+'_last_g.npz'
    model.run(no_plot = False,save_last_generation_as = filename)
    with Path('Results/'+_g+"_report.pkl").open('wb') as ff:
        pkl.dump(model.report,ff)
    '''with Path(_g+"_result.npz").open('wb') as ff:
        pkl.dump(model.result,ff)
    model.run(start_generation=model.result.last_generation)
    '''
    plt.plot(model.report, label = f"local optimization")
    last_generation=model.result.last_generation
    solutions=last_generation.variables
    scores=last_generation.scores
    Freqs=[]
    Order=[]
    sols=[]
    for solution in solutions:
        freq,comps=decoder(solution)
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
    df.to_csv('Results/ga_result_'+str(graph)+'.csv', index=False)
    return model


def main():
    global model_alex,model_google,model_mobile,model_res50,model_squeeze
    
    model_alex=run_ga(_g='alex')
    #model_alex.run(start_generation='Results/alex_last_g.npz')

    model_google=run_ga(_g='google')

    model_mobile=run_ga(_g='mobile')

    model_res50=run_ga(_g='res50')

    model_squeeze=run_ga(_g='squeeze')

    for r in model.result.last_generation.variables:
        config=decoder(r)
        print(config)
        print(P.Inference_Cost(_graph=g,_freq=config[0],_order=config[1]))
main()


def test():
    config=decoder([53,53,53,53,53,53,53,13])
    print(config)
    print(P.Inference_Cost(_graph=g,_freq=config[0],_order=config[1]))
test()




