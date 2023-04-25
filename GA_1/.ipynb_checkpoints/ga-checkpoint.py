# +
import numpy as np
import pickle as pkl
from geneticalgorithm import geneticalgorithm as ga
from genetic_algorithm import GeneticAlgorithm as GA
import pandas as pd
import os
import sys


from pathlib import Path
explore_freq=Path('freq_tranfer.csv').resolve()

NLayers={"alex":8, "google":11, "mobile":14, "res50":18, "squeeze":10, "test_transfer":2}
NFreqs={"L":6, "B":8, "G":5}
Metrics=["in","task","out","trans"]
n=100
params={"alex":(1,1,1), "google":(2,2,1), "mobile":(2,3,1), "res50":(2,4,1), "squeeze":(1,5,1), "test_transfer":(1,0,0)}
data={}
df=None
df2=None
# %config Completer.use_jedi = False
# -

def to_df(data):
    lists=[]
    for g in data:
        for c in NFreqs:
            for f in range(NFreqs[c]):
                if c=="G":
                    for fbig in range(NFreqs["B"]):
                        for layer in range(NLayers[g]):
                            for m in Metrics:
                                data_dict={"Graph":g, "Component":c, "Freq":f, "Freq_Host":fbig, "Layer":layer, "Metric":m,
                                           "Time":data[g][c][f][fbig][layer][m].get('time', np.nan),
                                           "Power":data[g][c][f][fbig][layer][m].get('Power', np.nan)}
                                lists.append(data_dict)

                else:
                    for layer in range(NLayers[g]):
                        for m in Metrics:
                            data_dict={"Graph":g, "Component":c, "Freq":f, "Layer":layer, "Metric":m,
                                       "Time":data[g][c][f][layer][m].get('time', np.nan),
                                       "Power":data[g][c][f][layer][m].get('Power', np.nan)}
                            lists.append(data_dict)

    df=pd.DataFrame(lists)
    with open("data_df.pkl","wb") as f:
        pk.dump(df,f)
    return df


def Profiling():
    caching=True
    #df = None
    if os.path.isfile("data.pkl") and caching:
        with open("data.pkl","rb") as f:
            data=pkl.load(f)


    if os.path.isfile("data_df.pkl") and caching:
        with open("data_df.pkl","rb") as f:
            df=pkl.load(f)

    else:
        df=to_df(data)
        df.to_csv("data_df.csv",index=False)
    
    return data,df
    #when reading:
    #test=pd.read_csv("data_df.csv",index_col=0)
    #or you can use df.to_csv with index=False argument


def load_transfer_df():
    #global transfer_df
    if explore_freq.exists():
        df=pd.read_csv(explore_freq)
        #print(f'transfer:\n{transfer_df_freq}')
    first_transfer_time = df.groupby('order')['transfer_time'].first()
    first_transfer_power = df.groupby('order')['transfer_power'].first()
    df['time_ratio'] = df['transfer_time'] / df['order'].map(first_transfer_time)
    df['power_ratio'] = df['transfer_power'] / df['order'].map(first_transfer_power)
    #transfer_df=df
    return df



def load_data():
    global data,df,df2,transfer_df,transfer_times
    data,df=Profiling()
    df2 = df.set_index(['Graph', 'Component', 'Freq', 'Layer', 'Metric', 'Freq_Host'])
    transfer_df=load_transfer_df()
    transfer_times={}
    with open("transfers.pkl",'rb') as f:
        transfer_times=pkl.load(f)


load_data()


def value(graph,comp,freq,layer,metric,attr):
    if len(freq)==1 or comp!='G':
        return df2.loc[(graph, comp, freq[0], layer, metric, np.NaN), attr]
    if len(freq)==2:
        return df2.loc[(graph, comp, freq[0], layer, metric, freq[1]), attr]
    else:
        return -1
value('google','G',[0,0],1,'task','Time')

df2


def comp_cost(g='alex',fn=[[0],[1],[2],[3],[4],[5],[6],[7]],cmps=8*'B',dvfs_delay=3.5, debug=False):
    fn=list(fn)
    fn.insert(0,fn[0])
    cmps=cmps[0]+cmps
    if debug:
        print(f'fn is {fn}')
    
    fc=len(fn)*[None]
    for i in range(len(fc)):
        fc_l=0
        fc_b=0
        fc_g=0
        if cmps[i-1]=='G':
            fc_g=fn[i-1][0]
            fc_b=fn[i-1][1]
        if cmps[i-1]=='B':
            fc_b=fn[i-1][0]
        if cmps[i-1]=='L':
            fc_l=fn[i-1][0]
        
        f={"L":[fc_l], "B":[fc_b], "G":[fc_g,fc_b]}
        fc[i]=f[cmps[i]]
        if debug:
            print(f'i:{i}, previous p:{cmps[i-1]}, current p:{cmps[i]}, curent p freqs:{f}, fc[i]:{fc[i]}')
    
    #just first layer(i=1) current freq is equal to its next freq
    #because next freq is applied before input(i=0) and it is already applied for the first layer
    fc[1]=fn[1]
    if debug:
        print(f'fc is:{fc}')
        print(f'processors:{cmps}')
    tt=0
    ee=0
    tt_nodvfs=0
    ee_nodvfs=0
    
    #comp time
    tfn=value(g,cmps[0],fn[0],0,'in','Time')
    tfc=value(g,cmps[0],fc[0],0,'in','Time')
    t=tfc
    if tfc > dvfs_delay:
        t=tfn - (dvfs_delay/tfc)*tfn + dvfs_delay  
    if debug:
        print(f'in:{0}, next_freq:{fn[0]} time(next_freq):{tfn} cur_freq:{fc[0]} time(cur_freq):{tfc} time:{t}')      
    tt+=t
    tt_nodvfs+=tfn
    
    #comp power
    pfn=value(g,cmps[0],fn[0],0,'in','Power')
    pfc=value(g,cmps[0],fc[0],0,'in','Power') 
    e=t*pfc
    if t > dvfs_delay:
        e=dvfs_delay*pfc + (t-dvfs_delay)*pfn
    e_nodvfs= tfn*pfn
    ee+=e
    ee_nodvfs+=e_nodvfs
    if debug:
        print(f'in:{0}, next_freq:{fn[0]} power(next_freq):{pfn} cur_freq:{fc[0]} power(cur_freq):{pfc} energy:{e}')
        
    for i in range(0,len(fn)-1):
        tfn=value(g,cmps[i+1],fn[i+1],i,'task','Time')
        tfc=value(g,cmps[i+1],fc[i+1],i,'task','Time')
        t=tfc
        if tfc > dvfs_delay:
            t=tfn - (dvfs_delay/tfc)*tfn + dvfs_delay
        if debug:
            print(f'layer:{i}, next_freq:{fn[i+1]} time(next_freq):{tfn} cur_freq:{fc[i+1]} time(cur_freq):{tfc} time:{t}')
        tt+=t
        tt_nodvfs+=tfn
        
        pfn=value(g,cmps[i+1],fn[i+1],i,'task','Power')
        pfc=value(g,cmps[i+1],fc[i+1],i,'task','Power') 
        e=t*pfc
        if t > dvfs_delay:
            e=dvfs_delay*pfc + (t-dvfs_delay)*pfn
        e_nodvfs= tfn*pfn
        if debug:
            print(f'layer:{i}, next_freq:{fn[i+1]} power(next_freq):{pfn} cur_freq:{fc[i+1]} power(cur_freq):{pfc} energy:{e}')
        ee+=e
        ee_nodvfs+=e_nodvfs
        
    if debug:
        print(f'time with dvfs delay: {tt}')
        print(f'time without dvfs delay: {tt_nodvfs}')
        print(f'power with dvfs delay: {ee}')
        print(f'power without dvfs delay: {ee_nodvfs}')
    return tt,ee/1000.0


_fn=[[0],[1],[2],[3],[4],[5],[6],[7]]
comp_cost(fn=_fn[::-1],debug=True)


def transfer_info(p1='B',p2='G',f1=[4],f2=[3,4]):
    '''p1='B'
    p2='G'
    f1=[4]
    f2=[3,4]'''
    f1=[int(i) for i in f1]
    f2=[int(i) for i in f2]
    '''print(p1)
    print(p2)
    print(f1)
    print(f2)'''
    global transfer_df
    order=p1+p2
    if order=='GL':
        order='GB'
        f2[0]=f1[1]
        p2='B'

    if p1=='G':
        f1[0]=0

    else:
        f1=[0]

    if p2=='G':
        f2[0]=0

    if order=='BG':
        f1[0]=f2[1]
    if order=='GB':
        f2[0]=f1[1]
    freqs=tuple([tuple(f1),tuple(f2)])
    #print(freqs)
    row=transfer_df[ (transfer_df['freq']==str(freqs)) & (transfer_df['order']==order)]
    #print(row)
    power=row['transfer_power'].iloc[0]
    coef_t=row['time_ratio'].iloc[0]
    
    return power,coef_t


a,b=transfer_info('G','B',[2.0, 7.0],[7.0])
transfer_df


def comm_cost(g='alex',fn=[[0],[1],[2],[3],[4],[5],[6],[7]],cmps=8*'B',dvfs_delay=3.5, debug=False):
    fn=list(fn)
    fn.insert(0,fn[0])
    cmps=cmps[0]+cmps
    if debug:
        print(f'fn is {fn}')
    
    fc=len(fn)*[None]
    for i in range(len(fc)):
        fc_l=0
        fc_b=0
        fc_g=0
        if cmps[i-1]=='G':
            fc_g=fn[i-1][0]
            fc_b=fn[i-1][1]
        if cmps[i-1]=='B':
            fc_b=fn[i-1][0]
        if cmps[i-1]=='L':
            fc_l=fn[i-1][0]
        
        f={"L":[fc_l], "B":[fc_b], "G":[fc_g,fc_b]}
        fc[i]=f[cmps[i]]
        if debug:
            print(f'i:{i}, previous p:{cmps[i-1]}, current p:{cmps[i]}, curent p freqs:{f}, fc[i]:{fc[i]}')
    
    #just first layer(i=1) current freq is equal to its next freq
    #because next freq is applied before input(i=0) and it is already applied for the first layer
    fc[1]=fn[1]
    if debug:
        print(f'fc is:{fc}')
        print(f'processors:{cmps}')
        
    transfer_t=0
    transfer_e=0
    
    for i in range(1,len(fn)-1):
        if cmps[i]!=cmps[i-1]:
            
            transfer_time=transfer_times[g][i][cmps[i]][cmps[i-1]]
            transfer_power,time_ratio=transfer_info(p1=cmps[i-1],p2=cmps[i],f1=fc[i-1],f2=fc[i])
        
            scaled_time=transfer_time * time_ratio
            transfer_energy=scaled_time * transfer_power
            
            transfer_t+=scaled_time
            transfer_e+=transfer_energy
            if debug:
                print(f"Transfer between layer {i-1} and {i}")
                print(f'transfer_time: {transfer_time}, time_ratio:{time_ratio}, scaled_time:{scaled_time}')
                print(f'transfer_power:{transfer_power}, transfer_energy:{transfer_energy}')
                print(f'total time:{transfer_t}')
                print(f'total energy:{transfer_e}')
    return transfer_t, transfer_e/1000.0
comm_cost(cmps="LLLBBBBB",debug=True)

comp_cost(g="alex",cmps='BBBBBBBB',fn=[ [0],[1],[2],[3],[4],[5],[6],[7] ] )


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


conf=decoder([1,6,8,14,44,6,0,0])
print(conf)
comp_cost(fn=conf[0],cmps=conf[1],debug=True)

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


# -

def run_ga(_g='alex',_target_latency=500):
    NL=NLayers[_g]
    varbound=[(0,53)]*NL
    def f(X):
        target_latency=_target_latency
        config=decoder(X)
        cmp_time,cmp_energy=comp_cost(g=_g, fn=config[0],cmps=config[1])
        comm_time,comm_energy=comm_cost(g=_g, fn=config[0],cmps=config[1])
        time=cmp_time+comm_time
        energy=cmp_energy+comm_energy
        if time<target_latency:
            return energy
        else:
            return 1000000000
        return time

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

    filename=_g+'_last_g.npz'
    res=model.run(no_plot = False,save_last_generation_as = filename)
    with Path(_g+"_report.pkl").open('wb') as ff:
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
        'graph':_g,
        'order': Order,
        'freq': Freqs,
        'score': scores
    })
    # Save the DataFrame as a CSV file
    df.to_csv('ga_result_'+str(_g)+'.csv', index=False)
    return model


def main():
    global model_alex,model_google,model_mobile,model_res50,model_squeeze
    model_alex=run_ga(_g='alex', _target_latency=300)

    model_google=run_ga(_g='google', _target_latency=450)

    model_mobile=run_ga(_g='mobile', _target_latency=350)

    model_res50=run_ga(_g='res50', _target_latency=900)

    model_squeeze=run_ga(_g='squeeze', _target_latency=400)

    for r in model.result.last_generation.variables:
        config=decoder(r)
        print(config)
        print(comp_time(fn=config[0],cmps=config[1],debug=False))

    config=decoder([53,53,53,53,53,53,53,13])
    print(config)
    print(comp_time(fn=config[0],cmps=config[1],debug=False))


main()
