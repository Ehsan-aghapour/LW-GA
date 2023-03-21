# +
import re
import Arduino_read
import os
import time
import threading
import subprocess
import itertools
import pickle as pk

cnn={
    "alex":"graph_alexnet_n_pipe_npu_lw",
    "google":"graph_googlenet_n_pipe_npu_lw",
    "mobile":"graph_mobilenet_n_pipe_npu_lw",
    "res50":"graph_resnet50_n_pipe_npu_lw",
    "squeeze":"graph_squeezenet_n_pipe_npu_lw",
}

graphs=["alex", "google", "mobile", "res50", "squeeze"]
NLayers={"alex":8, "google":11, "mobile":14, "res50":18, "squeeze":10}
C=["L","B", "G"]
NFreqs={"L":6, "B":8, "G":5}
Metrics=["transfer"]
n=10
params={"alex":(1,1), "google":(2,2), "mobile":(2,3), "res50":(2,4), "squeeze":(1,5)}

transfer={}
for g in graphs:
    transfer.setdefault(g,{})
    for layer in range(1,NLayers[g]):
        transfer[g].setdefault(layer,{})
        for dest in C:
            transfer[g][layer].setdefault(dest,{})
            for source in C:
                if source!=dest:
                    transfer[g][layer][dest].setdefault(source,{})
                        
                
 
#transfer[g][layer][c_dest][c_source][t/pwr]
                
print(transfer)                    
                            
# -


#transfer[g][layer][c_dest][c_source][t/pwr]
def Parse(timefile,graph="alex",order="BGBGBGBG"):
    with open(timefile) as ff:
        lines=ff.readlines()
    #order="BBBGBBBB"
    #freqs=[[0],[1],[2],[3,6],[4],[5],[6],[7]]
    trans={}
    for l in lines:     
        match = re.search(r"transfer_time of layer (\d+) : (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            trans[k]=value
            if k>0:
                transfer[graph][k][order[k]][order[k-1]]=value
            
    print(trans)   


# +

########################## Run a Config on board ############################
def Run_Graph(ALL_Freqs, run_command, myoutput, blocking=True):
    print(run_command)
    p = subprocess.Popen(run_command.split(),stdout=myoutput,stderr=myoutput, stdin=subprocess.PIPE, text=True)
    time.sleep(8)
    for Freqs in ALL_Freqs:       
        p.stdin.write(f'{Freqs}\n')
        p.stdin.flush()
        time.sleep(8)
    
    p.stdin.write("end\n")
    p.stdin.flush()
    if blocking:
        p.wait()


# -

def format_freqs(fs=[ [ [7],[6],[4],[3,6],[4],[5],[6],[7] ], [] ]):
        formated_fs=[]
        for f in fs:
            ff = '-'.join(['[' + str(sublist[0]) + ',' + str(sublist[1]) + ']' if len(sublist) > 1 else str(sublist[0]) for sublist in f])
            #print(ff)
            formated_fs.append(ff)
        return formated_fs


def profile(ff=["7-6-4-[3,6]-4-5-6-7"],_n=n,order='BBBGBBBB',graph="alex",tme="temp.txt"):
    if os.path.isfile(tme):
        print("loading existed files")
        return 
    
    rr=f"PiTest build/examples/LW/{cnn[graph]} test_graph/ CL {params[graph][0]} {params[graph][1]} 1 {_n} 0 0 100 100 {order} 1 2 4 Alex B B"
    oo=open(tme,'w+')
    Run_Graph(ff,rr,oo,True)
    #time.sleep(2)
    oo.close()


# +
def profile_task_time(graph="alex"):
    
    #os.system(f"PiPush /home/ehsan/UvA/ARMCL/Rock-Pi/ComputeLibrary_64_CPUGPULW/build/examples/LW/{cnn[graph]} test_graph/")
    #time.sleep(5)
    NL=NLayers[graph]
    
    C=["G","B","L"]
    combinations = list(itertools.combinations(C, 2))
    orders=[]
    
    for combination in combinations:
        order1=""
        order2=""
        for i in range(NL):
            order1=order1+combination[i%2]
            order2=order2+combination[(i+1)%2]
        orders.append(order1)
        orders.append(order2)
    print(orders)
    
    for _order in orders:
        print(f'graph:{graph} order:{_order} ')       
        timefile=f'./transfer_{graph}_'+_order+'.txt'
        profile(["min"],n,_order,graph,timefile)
        time.sleep(2)
        Parse(timefile,graph,_order)
        



# +
for graph in graphs:
    profile_task_time(graph)

k=input("ok?")
if k=='y':
    with open("./transfers.pkl","wb") as f:
        pk.dump(transfer,f)
# -


