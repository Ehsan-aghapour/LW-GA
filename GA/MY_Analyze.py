#python MY_Analyze
#python MY_Analyze 0 Res50 3(need GAResult_Res50.pkl) (first argument is new_run and second is thee name(postfix) of target cnn, 
# and last argument is number of clusters for clutering GA front face)
#from copyreg import pickle
#https://pymoo.org/getting_started/part_4.html
from __future__ import annotations
from cProfile import label
from re import L
import MY_GA as GA
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import Hypervolume
from pymoo.util.running_metric import RunningMetric
import pickle
import sys
import time

import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter
from sklearn.cluster import KMeans
import MY_Profile
import math


New_Run=1
if len(sys.argv) >1:
    New_Run=int(sys.argv[1])



n_evals = []             # corresponding number of function evaluations\
hist_F = []              # the objective space values in each generation
hist_cv = []             # constraint violation in each generation
hist_cv_avg = []         # average constraint violation in the whole population

global res
def GARun(_graph='Alex'):
    global res
    try:
        res=GA.main(_graph)
        DumpRes()
    except:
        DumpRes()
    
    

def His():
    hist=res.history
    k=res.history
    global X
    global F
    X, F = res.opt.get("X", "F")
    for algo in hist:

        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algo.opt

        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())

        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])

def An_CV():
    vals = hist_cv_avg

    k = np.where(np.array(vals) <= 0.0)[0].min()
    print(f"Whole population feasible in Generation {k} after {n_evals[k]} evaluations.")

    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, vals,  color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, vals,  facecolor="none", edgecolor='black', marker="p")
    plt.axvline(n_evals[k], color="red", label="All Feasible", linestyle="--")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Constraint Violation")
    plt.legend()
    plt.show()

def An_HV():
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    
    metric = Hypervolume(ref_point= np.array([1.1, 1.1]),
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal,
                     nadir=approx_nadir)

    hv = [metric.do(_F) for _F in hist_F]

    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.show()

def An_RM():
    running = RunningMetric(delta_gen=4,
                        n_plots=4,
                        only_if_n_plots=False,
                        #do_close=False,
                        key_press=False,
                        do_show=True)

    for algorithm in res.history:
        p=running.notify(algorithm)
    

def DumpRes():
    with open('GAResult.pkl','wb') as pkf:
        pickle.dump(res,pkf)

def LoadRes(Graph=''):
    global res
    if Graph:
        Graph='_'+Graph
    
    with open('GAResult'+Graph+'.pkl','rb') as pkf:
        res=pickle.load(pkf)
    return res


def An_CNV():
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt0 = np.array([e.opt[0].F[0] for e in res.history])
    opt1 = np.array([e.opt[0].F[1] for e in res.history])

    plt.title("Convergence")
    plt.plot(n_evals, opt0, "--")
    #plt.yscale("log")
    plt.show()

    plt.plot(n_evals, opt1, "-")
    #plt.yscale("log")
    plt.show()
    j=input("next...")

def plot_objects():
    #f=res.history[-1].reslut().F
    f=res.F
    # denormalization:
    ff=np.multiply(f,[100,300])

    #1
    #plot=Scatter()
    #plot.add(ff, facecolor="none", edgecolor="red")
    #plot.show()

    #2:
    plt.scatter(ff[:,0],ff[:,1],facecolors='none', edgecolors='r')
    plt.xlabel('Frame time (ms)')
    plt.ylabel('Energy (mj)')
    plt.show()

def plot_clustering(Graph='Alex',k=3):
    LoadRes(Graph)
    MY_Profile.set_parameters(_Graph=Graph)
    c_ref=["green","red", "yellow", "blue"]
    c_ref=np.array(c_ref)

    #sort normalized front
    sorted_F=res.F[np.argsort(res.F[:,0])]

    #cluster based on sorted normalized front
    model = KMeans(n_clusters=k).fit(res.F)

    #calc energy and time by denormalizing
    ff=np.multiply(res.F,[100,300])

    #index of sorted F based on first column (object) that is time
    sort_indexes=np.argsort(ff[:,0])

    #sorted ff
    sorted_ff=ff[sort_indexes]

    plt.figure(figsize=(8, 6))
    plt.scatter(ff[:,0], ff[:,1], c=c_ref[model.labels_.astype(int)])
    #plt.scatter(sorted_F[:,0], sorted_F[:,1], c=c_ref[model.labels_.astype(int)])
    plt.xlabel('Execution Time Per Frame (ms)')
    plt.ylabel('Energy Per Frame (mj)')
    plt.savefig(Graph+'_Time.jpg', dpi=1000)

    plt.figure(figsize=(8, 6))
    plt.scatter(1000/ff[:,0], ff[:,1], c=c_ref[model.labels_.astype(int)])
    plt.xlabel('Throughput (FPS)')
    plt.ylabel('Energy Per Frame (mj)')
    plt.savefig(Graph+'_FPS.jpg', dpi=1000)

    columns=['Index','Order','Host (G N)','Freq [L B G]','Time (ms)','Energy (mj)']
    columns2=['Index','X','Time (ms)','Energy (mj)']
    
    d=[]
    d2=[]
    colors=[]
    colors2=[]
    Xf=open(Graph+'_Xs.csv','w')
    for i in range(len(sort_indexes)):
        t=[]
        t2=[]
        t.append(i)
        
        t2.append(i)
        t2.append(res.X[sort_indexes[i]])
        
        chromos=res.X[sort_indexes[i]]
        
        t.append(MY_Profile.Decode(chromos)[0])
        t.append(MY_Profile.Decode(chromos)[1])
        t.append(MY_Profile.Decode(chromos)[2])
        t.append(round(ff[sort_indexes[i]][0],2))
        t.append(round(ff[sort_indexes[i]][1],2))
        d.append(t)
        for e in t:
            Xf.write(str(e)+',')
        Xf.write(str(model.labels_.astype(int)[sort_indexes[i]])+',')
        Xf.write(str(res.X[sort_indexes[i]])+'\n')

        t2.append(round(ff[sort_indexes[i]][0],2))
        t2.append(round(ff[sort_indexes[i]][1],2))
        d2.append(t2)

        tt=[]
        for j in range(len(columns)):
            tt.append(c_ref[model.labels_.astype(int)[sort_indexes[i]]])
        colors.append(tt)

        tt2=[]
        for j in range(len(columns2)):
            tt2.append(c_ref[model.labels_.astype(int)[sort_indexes[i]]])
        colors2.append(tt2)
    Xf.close()
    
    plt.figure()
    ax=plt.table(cellText=d, cellColours=colors,colLabels=columns, loc='center',cellLoc='center')
    plt.axis('off')
    #ax.auto_set_font_size(True)
    ax.auto_set_font_size(False)
    ax.set_fontsize(7)
    ax.auto_set_column_width(col=list(range(len(columns))))
    plt.savefig(Graph+'_Table.jpg', dpi=1000)

    plt.figure()
    ax=plt.table(cellText=d2, cellColours=colors2,colLabels=columns2, loc='center',cellLoc='center')
    plt.axis('off')
    ax.auto_set_font_size(False)
    ax.set_fontsize(8)
    ax.auto_set_column_width(col=list(range(len(columns2))))
    plt.savefig(Graph+'_TableX.jpg', dpi=1000)

    plt.show()

'''
from sklearn.cluster import KMeans


1:
kmeans = KMeans(n_clusters= 3)
label = kmeans.fit_predict(ff)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(ff[label == i , 0] , ff[label == i , 1] , label = i)
plt.legend()
plt.show()


2:
model = KMeans(n_clusters=5).fit(X)
# Visualize it:
#plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))


colors = np.array(range(0,len(ff)*10,10))
plt.scatter(ff[:,0],ff[:,1],c=colors)

for i in range(len(ff)):
    plt.annotate(r.X[i],(ff[i][0],ff[i][1]))
'''


def test():
    global points
    global colors
    points=[]
    colors=[]
    g=0
    for a in res.history:
        for p in a.opt:
            if not any(np.array_equal(p.F,y) for y in points) and g:
                print(f'{p.F} not exist in {points} appear in generation {g}--{p.data["n_gen"]}')
                #input()
                points.append(p.F)
                #colors.append(p.data['n_gen'])
                colors.append(g)
        g=g+1

    points=np.array(points)
    colors=np.array(colors)

    global last_points
    global last_colors
    last_points=[]
    last_colors=[]
    for k,p in enumerate(points):
        if any(np.array_equal(p,y) for y in res.F):
            last_points.append(p)
            last_colors.append(colors[k])
    last_points=np.array(last_points)
    last_colors=np.array(last_colors)

    '''for i in range(len(last_colors)):
        plt.scatter(last_points[:,0],last_points[:,1],c=last_colors)
        plt.annotate(last_colors[i],last_points[i])'''
    for i in range(len(colors)):
        plt.scatter(points[:,0],points[:,1],c=colors)
        plt.annotate(colors[i],points[i])

    sorted_F=res.F[np.argsort(res.F[:,0])]
    print(sorted_F[:,0])
    plt.plot(sorted_F[:,0],sorted_F[:,1],'-o')
    plt.show()


def plot_fronts(k=0,max=0,ignore=0):
    H=res.history
    global F1s
    global P_F1s
    F1s=[H[0].result().F]
    P_F1s=[H[0].result().opt]
    mapping={0:0}
    for i in range(1,len(H)):
        print(f'{H[i].result().F.shape} and {F1s[-1].shape} and {H[i].result().F.shape == F1s[-1].shape}')
        if not (H[i].result().F.shape == F1s[-1].shape):
            F1s.append(H[i].result().F)
            P_F1s.append(H[i].result().opt)
            mapping[len(F1s)-1]=i
        else:
            if not (H[i].result().F==F1s[-1]).all():
                F1s.append(H[i].result().F)
                P_F1s.append(H[i].result().opt)
                mapping[len(F1s)-1]=i
        
    
    for i in range(len(F1s)):
        P_F1s[i]=P_F1s[i][np.argsort(F1s[i][:,0])]
        F1s[i]=F1s[i][np.argsort(F1s[i][:,0])]
    
    _k=5
    if k:
        _k=k
    step=int(math.ceil((len(F1s)/_k)))
    print(f'step is {step}')
    #input()

    target_index=len(F1s)-1
    n=1
    while target_index>0:
        if n<=ignore:
            target_index=target_index-step
            n=n+1
            continue
        target_F=F1s[target_index]
        target_p=P_F1s[target_index]
        plt.plot(target_F[:,0],target_F[:,1],'-o',label=f'G:{mapping[target_index]} Points:{len(target_F)}')
        plt.legend()
        for i in range(len(target_F)):      
            plt.annotate(target_p[i].data['n_gen'],target_F[i])
        target_index=target_index-step
        if max and n>=max:
            break
        n=n+1
    
    plt.show()







    




def main():
    if New_Run:
        Graph=''
        if len(sys.argv) > 2:
            Graph=sys.argv[2]
        GARun(Graph)
        His()
        An_CNV()
        An_CV()
        An_HV()
        An_RM()
    else:
        global res
        Graph=''
        if len(sys.argv) > 2:
            Graph=sys.argv[2]
        LoadRes(Graph)
        His() 
        An_CNV()
        An_CV()
        An_HV()
        An_RM()
        k=1
        if len(sys.argv) > 3:
            k=sys.argv[3]
        plot_clustering(Graph,int(k))
if __name__ == "__main__":
    main()
    