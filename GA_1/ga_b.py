# +
import numpy as np
import pickle as pkl
from geneticalgorithm import geneticalgorithm as ga
from genetic_algorithm import GeneticAlgorithm as GA
import pandas as pd
import os

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

# +
NL=8
varbound=[(0,53)]*NL
target_latency=150

def f(X):
    #print(X)
    global target_latency
    config=decoder(X)
    cmp_time,cmp_energy=comp_cost(fn=config[0],cmps=config[1])
    comm_time,comm_energy=comm_cost(fn=config[0],cmps=config[1])
    time=cmp_time+comm_time
    energy=cmp_energy+comm_energy
    #print(config)
    #print(time)
    if time<target_latency:
        return energy
    else:
        return 1000000000
    return time

algorithm_param = {'max_num_iteration': 3000,
                   'population_size':100,
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

model.run(no_plot = False,)
plt.plot(model.report, label = f"local optimization")


# -

for r in model.result.last_generation.variables:
    config=decoder(r)
    print(config)
    print(comp_time(fn=config[0],cmps=config[1],debug=False))

config=decoder([53,53,53,53,53,53,53,13])
print(config)
print(comp_time(fn=config[0],cmps=config[1],debug=False))

m=model
print(dir(m.result))
#print(dir(m))
#print(f'output:\n{m.output_dict}')
#print(f'best function:\n{m.best_function}')
print(dir(m.result.last_generation))
print(f'best variable:\n{[decoder(i) for i in m.result.last_generation.variables]}')


def _evaluate(self, x, out, *args, **kwargs):
    g=self.g
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
            t=data[g][c][F[i][0]][F[i][1]][i]["task"]["time"]
            p=data[g][c][F[i][0]][F[i][1]][i]["task"]["power"]

        else:
            t=data[g][c][F[i][0]][i]["task"]["time"]
            p=data[g][c][F[i][0]][i]["task"]["power"]

        #t=d["time"]
        #p=d["power"]
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
            if C[i]=="G":
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

# +
from geneticalgorithm import geneticalgorithm as ga
def f1(x):
    return (x[0]-2)**2 + (x[1]-1)**2

def f2(x):
    return (x[0]-3)**2 + (x[1]-2)**2

def constraint(x):
    return x[0] + x[1] - 4

def fitness0(x):
    f1_val = f1(x)
    f2_val = f2(x)
    constraint_val = constraint(x)
    if constraint_val <= 0:
        return (f1_val, f2_val)
    else:
        # if the constraint is not satisfied, return a very high fitness value
        return (np.inf, np.inf)
    
def fitness(x):
    f1_val = f1(x)
    f2_val = f2(x)
    constraint_val = constraint(x)
    if constraint_val <= 0:
        return f1_val + f2_val
    else:
        # if the constraint is not satisfied, return a very high fitness value
        return np.inf

class MyGeneticAlgorithm(ga):
    def __init__(self, function, dimension, variable_type, variable_boundaries, variable_type_mixed=None, function_timeout=10, algorithm_parameters=None, convergence_curve=True, progress_bar=True):
        super().__init__(function,dimension,variable_type,variable_boundaries, variable_type_mixed=variable_type_mixed, function_timeout=function_timeout, convergence_curve=convergence_curve, progress_bar=progress_bar)
        self.best_solutions = []
        if algorithm_parameters is not None:
            self.param.update(algorithm_parameters)
        #print(f'init, pop size:{self.population_size}')
    def crossover(self, parent1, parent2):
        print(f'pop size:{self.population_size}')
        child = []
        for i in range(self.dimension):
            # custom crossover operator that takes the average of the two parents' decision variables
            child.append((parent1[i] + parent2[i]) / 2)
        return child
    
    def mutate(self, chromosome):
        # use the default mutation operator
        super().mutate(chromosome)
        
    


# define the fitness function, variable boundaries, and other parameters as before
varbound = np.array([[0, 5], [0, 3]])
model = MyGeneticAlgorithm(function=fitness, dimension=2, variable_type='real', variable_boundaries=varbound,algorithm_parameters={'population_size':150})
#model = ga(function=fitness, dimension=2, variable_type='real', variable_boundaries=varbound)
print(model.param)

model.run()


print(model.output_dict)
print(model.best_function)
print(model.best_variable)
model.report
# print the results
print('Results:')
for solution in [model.output_dict]:
    print('x1 =', solution['variable'][0], 'x2 =', solution['variable'][1], 'f1 =', solution['function'])


# +


power = pd.read_csv('power.csv')

debug=0
name_layers='layer.pkl'
a_file = open(name_layers, "rb")
profile_data = pickle.load(a_file)


# -

def mapping_time(data_size):
	t = ( (7.72e-4) * data_size - (4.62e-11) * (data_size**2) + 105.71 )*(10**-3)
	if debug:
		print(f'map time:{t}')
	return t

def unmapping_time(data_size):
	t = ( (2.56e-5) * data_size - (1.54e-12) * (data_size**2) + 10.96 )*(10**-3)
	if debug:
		print(f'unmap time:{t}')
	return t

def copy_time(data_size):
	t = ( (1.87e-3) * data_size - (1.12e-10) * (data_size**2) + 68.74 )*(10**-3)
	if debug:
		print(f'copy time:{t}')
	return t

def GB_overhead(data_size):
	if debug:
		print(f'datasize:{data_size}')
	return (mapping_time(data_size)+copy_time(data_size))

def BG_overhead(data_size):
	if debug:
		print(f'datasize:{data_size}')
	return (mapping_time(data_size)+copy_time(data_size)+unmapping_time(data_size))

def BB_overhead(data_size):
	if debug:
		print(f'datasize:{data_size}')
	return (copy_time(data_size))

ovh={'GB':GB_overhead, 'BG':BG_overhead, 'LB':LB_overhead, 'BL':BL_overhead, 'LG':LG_overhead, 'GL':GL_overhead}

Threads=[]
latency=1000
def f_latency(X):
	if debug:
		print(Threads)
		print(X)
	t=0
	overhead=0
	energy=0
	for i in range(len(X)):
		if debug:
			print(i)
		layer_time=0
		switch_overhead=0		
		if X[i] == 0:
			PE='G'
			threads=1
		if X[i] > 0 :
			PE='B'
			threads=Threads[int(X[i])-1]
			
		layer_time=data[PE][str(threads)][str(i)]['time']
		data_size=data[PE][str(threads)][str(i)]['data_size']
		if i<(len(X)-1):
			PE2='B'
			if X[i+1] == 0:
				PE2='G'
				
			if (X[i]) != (X[i+1]):
				switch_overhead=ovh[PE+PE2](data_size)
				'''if X[i] ==0:
					switch_overhead=GB_overhead(data_size)
				else:
					switch_overhead=BG_overhead(data_size)'''
		if debug:
			print(f'layer time:{layer_time},overhead:{switch_overhead}')
		t+=layer_time
		overhead+=switch_overhead
	
	if debug:	
		print(f'Graph process time:{t}, switch overheads:{overhead}')
	return (t+overhead)

'''
from collections import Counter

words = ['a', 'b', 'c', 'a']

Counter(words).keys() # equals to list(set(words))
Counter(words).values()
'''

def f_FPS(X):
	if debug:
		print(Threads)
		print(X)
	#stages=len(set(X))
	stages=int(max(X))+1
	t=[0]*stages
	overhead=[0]*stages
	energy=0
	
	order=[int(X[0])]
	coefs=[]
	energy=0
	
	
	for i in range(1,len(X)):
		if X[i] != X[i-1]:
			order.append(int(X[i]))
		values, counts = np.unique(order, return_counts=True)
		if debug:
			print(values)
			print(counts)
		coefs=[1]*(int(max(values))+1)
		for i,v in enumerate(values):
			coefs[int(v)]=counts[i]
			
	stage_dict={0:'G'}
	for i,td in enumerate(Threads):
		stage_dict[i+1]=f'B{td}'
		
	
	if debug:		
		print(f'mapping is:{X},order is:{order},coef is:{coefs}')
		
	for i in range(len(X)):
		if debug:
			print(i)
		layer_time=0
		switch_overhead=0		
		if X[i] == 0:
			PE='G'
			threads=1
		if X[i] > 0 :
			PE='B'
			threads=Threads[int(X[i])-1]
			
		
		layer_time=data[PE][str(threads)][str(i)]['time']
		data_size=data[PE][str(threads)][str(i)]['data_size']
		
		title=stage_dict[int(X[i])]
		layer_power=power[title][i]
		if debug:
			print(f'title:{title}, layer:{i}, power:{layer_power}')
		layer_energy=layer_power*layer_time
		energy+=layer_energy
		if i<(len(X)-1):
			PE2='B'
			if X[i+1] == 0:
				PE2='G'
				
			if (X[i]) != (X[i+1]):
				switch_overhead=ovh[PE+PE2](data_size)
				'''if X[i] ==0:
					switch_overhead=GB_overhead(data_size)
				else:
					switch_overhead=BG_overhead(data_size)'''
		if debug:
			print(f'layer time:{layer_time},overhead:{switch_overhead}')
			
		t[int(X[i])]+=layer_time*coefs[int(X[i])]
		#Ehsan: it should be modified so that this overhead be counted for CPU clustre including responsible core for GPU processing
		overhead[int(X[i])]+=switch_overhead
	
	if debug:	
		print(f'Graph process time:{t}, switch overheads:{overhead}')
	lat=max(t)+max(overhead)
	penalty=energy
	if(lat>latency):
		penalty+=100000000*(lat-latency)
	return penalty

'''mix:
varbound=np.array([[0.5,1.5],[1,100],[0,1]])
vartype=np.array([['real'],['int'],['int']])
model=ga(function=f,dimension=3,variable_type_mixed=vartype,variable_boundaries=varbound)
'''

Confs={ 1:[[4]],
	2:[[2,2],[1,3],[3,1]],
	3:[[2,1,1],[1,2,1],[1,1,2]],
	4:[[1,1,1,1]] }


graph=list(profile_data.keys())[0]
profile_data2={graph:profile_data[graph]}
#profile_data2[graph]=profile_data[graph]
latencies=[40,50,60,70,80,90,100,110,120,130,140,150,160]
#latencies=[40]
outputs={}
pareto={}
all_best=10000000000e9
all_best_variable=[]
for graph in profile_data2:
	data=profile_data[graph]
	p=1
	for PEs in range(2,6):
		nlayers=len(data['G']['1'])
		varbound=np.array([[0,PEs-1]]*nlayers)
		CPUs=PEs-1
		for conf in Confs[CPUs]:		
			Threads=conf
			for l in latencies:
				latency=l
				model=ga(function=f_FPS,dimension=nlayers,variable_type='int'\
				,variable_boundaries=varbound,progress_bar=False,convergence_curve=False)
				if p:
					print(model.param)
					p=0
				print("\n\t\t\t--------\n")
				print(f'Runnig GA for graph: {graph} with #layers:{nlayers} on #PEs:{PEs},config:{conf},latency:{l}')
				model.run()
				outputs[graph+'-'+str(PEs)+'-'+str(conf)+'-'+str(l)]=model.output_dict
				#mapping=model.best_variable
				e=model.best_function
				best=pareto.setdefault(l,e)
				if pareto[l]>e:
					pareto[l]=e
				if all_best > e:
					all_best=e
					all_best_variable=model.best_variable
				#print("\n\t\t\t--------\n")
				
		print("\n\n*******************************\n")

print(outputs)
print("\n\n\n")		
print(f'pareto:{pareto}')
print(f'all_best:{all_best}, at:{all_best_variable}')


name='ga_outputs.pkl'
a_file = open(name, "wb")
pickle.dump(outputs, a_file)
a_file.close()




'''
def f(X):
	return np.sum(X)
	
m=ga(function=f,dimension=3,variable_type='int',variable_boundaries=np.array([[0,10]]*3),progress_bar=False,convergence_curve=False)	
'''


'''run with modified hyperparameters:
{'max_num_iteration': None, 'population_size': 100, 'mutation_probability': 0.1, 'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type': 'uniform', 'max_iteration_without_improv': None}
algorithm_param = {'max_num_iteration': 3000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
 
model=ga(function=f,dimension=3,variable_type='int',variable_boundaries=varbound,algorithm_parameters=algorithm_param)        
'''          

