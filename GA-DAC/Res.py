# +

graphs=["YOLOv3","MobileV1"]
import sys
target_acc=66
target_graph="YOLOv3"
#target_graph="MobileV1"
#p_dir='Power_Model/'
p_dir='/home/ehsan/UvA/ARMCL/Rock-Pi/LW-ARM-CO-UP/New/Model/test'
sys.path.append(p_dir)
import P

'''if sys.argv[1]=="y":
    target_graph="YOLOv3"
if sys.argv[1]=="m":
    target_graph="MobileV1"
target_acc=float(sys.argv[2])'''

print(f'Running Ga for model:{target_graph} for target accuracy:{target_acc}')

# +
import pymoo
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling



import sys
from tensorflow.keras import layers, models
# -



# +
P.Load_Data()
model=None




# + endofcell="--"
Target_Acc={"YOLOv3":[66], "MobileV1":[]}

NLayers={"YOLOv3":75, "MobileV1":14}
model_names = { "MobileV1":"Mobile.h5", "YOLOv3":"YOLOv3.h5" }

# -

def decode_gene(v):
    if v==0:
        return "N",[v]
    elif v<6 :
        return "G",[v-1,7]
    elif v<14:
        return "B",[v-6]
    elif v<20:
        return "L",[v-14]
def decoder(chromosome):
    freqs=[]
    ps=''
    for gene in chromosome:
        p,fs=decode_gene(gene)
        ps+=p
        freqs.append(fs)
    return freqs,ps
# --


#np.set_printoptions(threshold=np.inf)
#class MyProblem(ElementwiseProblem):
class MyProblem(Problem):

    def __init__(self,_graph,target_accuracy):
        self.target_accuracy=target_accuracy
        self.g=_graph
        self.n=NLayers[_graph]
        print("Initialize the problem for graph with " + str(self.n) + " layers.")
        _xl=np.full(self.n,0)
        _xu=np.full(self.n,19)
        super().__init__(n_var=self.n,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array(_xl),
                         xu=np.array(_xu),
                         vtype=int
                        )
        #self.integer = np.arange(20)

    
        
    def _evaluate(self, X, out, *args, **kwargs):
        
        X = np.round(X).astype(int)
        #print(X)
        configs=[decoder(x1) for x1 in X]
        
        inference_time = np.zeros(X.shape[0])
        avg_power = np.zeros(X.shape[0])

        # Iterate over each solution for the second function
        for i,config in enumerate(configs):
            inference_time[i],avg_power[i],_ = P.Inference_Cost(_graph=self.g,_freq=config[0],_order=config[1],_dvfs_delay='variable')
            if np.isnan(inference_time[i]):
                print(X[i])
                print(config)
                input("nan")
            
        x_quantization=np.where(X==0,1,0)
        predicted_accuracy = model.predict(x_quantization).flatten()
        #print(predicted_accuracy)
        #G= predicted_accuracy - self.target_accuracy
        G= self.target_accuracy - predicted_accuracy
        #print(G)
        #input()
        #print(f'time:{inference_time}')
        #print(f'power:{avg_power}')

        out["F"] = [inference_time, avg_power]
        out["G"] = [G]

# +


algorithm = NSGA2(
    pop_size=100,
    eliminate_duplicates=True
)


# +
import pygmo as pg
import numpy as np
import pandas as pd

model_name=model_names[target_graph]
model=models.load_model(model_name)
problem = MyProblem(target_graph,66)

'''res = minimize(problem,
               algorithm,
               ("n_gen", 200),
               verbose=False,
               seed=1,
                save_history=True,
              )

import pickle
with open(f'{target_graph}-{target_acc}', "wb") as f:
    pickle.dump(res, f)'''

Name_GA="YOLOv3-66.0"
Name_BL='YOLOv3_ParotoFront68.2711766072.csv'
Name_GA="R/"+Name_GA
Name_BL="R/2/"+Name_BL
import pickle
with open(Name_GA,'rb') as f:
    res=pickle.load(f)
    
df = pd.read_csv(Name_BL)  # Replace with your CSV file path
pareto_frontier_baseline = df[['Time', 'Power']].to_numpy()
pareto_frontier_ga=res.F
#display(pareto_frontier_baseline)
#display(pareto_frontier_ga)
plot = Scatter()
plot.add(res.F, edgecolor="red", facecolor="none")
plot.add(pareto_frontier_baseline, edgecolor="blue", facecolor="none")
plot.show()
ref_point = [max(pareto_frontier_ga[:, 0].max(), pareto_frontier_baseline[:, 0].max()) + 1,
             max(pareto_frontier_ga[:, 1].max(), pareto_frontier_baseline[:, 1].max()) + 1]

hv_ga = pg.hypervolume(pareto_frontier_ga)
hv_baseline = pg.hypervolume(pareto_frontier_baseline)

volume_ga = hv_ga.compute(ref_point)
volume_baseline = hv_baseline.compute(ref_point)

print("Hypervolume GA:", volume_ga)
print("Hypervolume Baseline:", volume_baseline)
d=pd.DataFrame(pareto_frontier_ga,columns=["Time","Power"])
d.to_csv("mot.csv")
# -

res.



for i in range(len(res.X)):
    if res.F[i][0]!=np.nan:
        x = res.X[i]
        y = res.F[i]
        print(f"Solution {i+1}: Decision Variables , Objective Values = {y}")












if False:
    #inference_time,avg_power,_=P.Inference_Cost(_graph=graph,_freq=config[0],_order=config[1],_dvfs_delay='variable')
    x='[ 8 14  0  6  3  2  4  7  8 10  8 13  4 17]'
    x='[ 2  7 16 11  1  5  5  7 18  4  9  4  8 13 16  6  9  6  7 18  2 13 13  4\
       5  1 14 14  1  3  1 13 16  9 18 12  1 15  3  3 11  4  6  1  9 10  2  3\
      10  6  1  5 10  8  7  2  1  1  3  9  0  6 10 14 16 15  8  9 16  3 10  5\
      12 17 18]'
    # Remove the square brackets and split the string by spaces
    values = x.strip('[]').split()

    # Convert the values to integers
    x = np.array([int(value) for value in values])
    config=decoder(x)
    inference_time,avg_power,_=P.Inference_Cost(_debug=False,_graph=graph,_freq=config[0],_order=config[1],_dvfs_delay='variable')
    x=np.where(x==0,1,0)
    print(x)
    #model.predict(x.reshape(1,NLayers[graph]))
    print(inference_time,avg_power)
    import math
    np.isnan(inference_time)




