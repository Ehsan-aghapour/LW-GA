# +
import re
import Arduino_read
import os
import time
import threading
import subprocess
import pickle as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
import select
from pathlib import Path

cnn_dir="/home/ehsan/UvA/ARMCL/Rock-Pi/ComputeLibrary_64_CPUGPULW/"

cnn={
    "alex":"graph_alexnet_n_pipe_npu_lw",
    "google":"graph_googlenet_n_pipe_npu_lw",
    "mobile":"graph_mobilenet_n_pipe_npu_lw",
    "res50":"graph_resnet50_n_pipe_npu_lw",
    "squeeze":"graph_squeezenet_n_pipe_npu_lw",
    "test_transfer":"graph_test_transfer_n_pipe_npu_lw"
}


graphs=["alex", "google", "mobile", "res50", "squeeze"]
NLayers={"alex":8, "google":11, "mobile":14, "res50":18, "squeeze":10, "test_transfer":2}
NFreqs={"L":6, "B":8, "G":5}
Metrics=["in","task","out","trans"]
n=100
params={"alex":(1,1,1), "google":(2,2,1), "mobile":(2,3,1), "res50":(2,4,1), "squeeze":(1,5,1), "test_transfer":(1,0,0)}
C=["L","B", "G"]

Layers_csv=Path('Layers.csv').resolve()
Transfers_csv=Path('Transfers.csv').resolve()
Transfer_Freq_csv=Path('Transfer_Freq.csv').resolve()
Transfer_Data_Size_Min_Freq_csv=Path('Transfer_Data_Size_Min_Freq.csv').resolve()
Transfer_Data_Size_Max_Freq_csv=Path('Transfer_Data_Size_Max_Freq.csv').resolve()
Evaluations_csv=Path("Evaluations.csv").resolve()
Layers_logs=Path("./Layers/").resolve()
Transfers_logs=Path("./Transfers/").resolve()
Synthetic_Tranfer_logs=Path("./Synthetic_Transfers/").resolve()
Layers_Percentage_csv=Path("Layers_Percentage.csv").resolve()
Layers_With_Percentage_csv=Path("Layers_With_Percentage.csv").resolve()


Layers_df=pd.DataFrame(columns=["Graph", "Component", "Freq", "Freq_Host", "Layer", "Metric", "Time", "Power"])
Transfers_df=pd.DataFrame(columns=["Graph", "Layer", "Dest", "Src", "Time"])
Transfer_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'SenderFreq','RecFreq' 'transfer_time', 'transfer_power'])
Transfer_Data_Size_Min_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'transfer_time', 'transfer_power'])
Transfer_Data_Size_Max_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'transfer_time', 'transfer_power'])
Evaluations_df=pd.DataFrame(columns=['graph','order','freq','input_time','task_time','total_time', 'input_power','task_power'])


# -


def Load_Data():
    global Layers_df, Transfers_df, Transfer_Freq_df,\
        Transfer_Data_Size_Min_Freq_df,Transfer_Data_Size_Max_Freq_df
    #### Load data of layer times with different freqs
    if Layers_csv.exists():
        Layers_df=pd.read_csv(Layers_csv)
    
    # set index to enable access with list of indexes (in value function)
    Layers_df_indexed = Layers_df.set_index(['Graph', 'Component', 'Freq', 'Layer', 'Metric', 'Freq_Host'])
    
    #### Load transfer times of real layers
    if Transfers_csv.exists():
        Transfers_df=pd.read_csv(Transfers_csv)

       
    ### Load time and power of tranfer with syntethic layers for different layers
    if Transfer_Freq_csv.exists():
        Transfer_Freq_df=pd.read_csv(Transfer_Freq_csv)
    if Transfer_Freq_df.shape[0]:
        first_transfer_time = Transfer_Freq_df.groupby('order')['transfer_time'].first()
        first_transfer_power = Transfer_Freq_df.groupby('order')['transfer_power'].first()
        Transfer_Freq_df['time_ratio'] = Transfer_Freq_df['transfer_time'] / Transfer_Freq_df['order'].map(first_transfer_time)
        Transfer_Freq_df['power_ratio'] = Transfer_Freq_df['transfer_power'] / Transfer_Freq_df['order'].map(first_transfer_power)   
    
    ### Load tranfering VS data size with min freq
    if Transfer_Data_Size_Min_Freq_csv.exists():
        Transfer_Data_Size_Min_Freq_df=pd.read_csv(Transfer_Data_Size_Min_Freq_csv)
    
    ### Load tranfering VS data size with max freq
    if Transfer_Data_Size_Max_Freq_csv.exists():
        Transfer_Data_Size_Max_Freq_df=pd.read_csv(Transfer_Data_Size_Max_Freq_csv)

Load_Data()



print(Layers_df[(Layers_df['Graph']=="alex") & (Layers_df['Component']=="B") & 
          (Layers_df['Freq']==0) & (Layers_df['Freq_Host']==-1)])
print(Layers_df[(Layers_df['Graph']=="alex") & (Layers_df['Component']=="B") & 
          (Layers_df['Freq']==0) & (Layers_df['Freq_Host']==-1)]["Time"].sum())
i=0
100*Layers_df[(Layers_df['Graph']=="alex") & (Layers_df['Component']=="B") & 
          (Layers_df['Freq']==0) & (Layers_df['Freq_Host']==-1) & (Layers_df['Layer']==i) ]["Time"]/Layers_df[(Layers_df['Graph']=="alex") & (Layers_df['Component']=="B") & 
          (Layers_df['Freq']==0) & (Layers_df['Freq_Host']==-1)]["Time"].sum()

#def Compute_Layer_Percentage():
if True:
    sum_time_per_graph_component = Layers_df.groupby(['Graph', 'Component', 'Freq', 'Freq_Host'])['Time'].sum().reset_index()
    pd.set_option('display.max_rows', 1000)
    #print(sum_time_per_graph_component)
    Layers_With_Percentage_df=Layers_df.merge(sum_time_per_graph_component, on=['Graph', 'Component', 'Freq', 'Freq_Host'], suffixes=('', '_sum'))
    Layers_With_Percentage_df['Time_Percentage'] = Layers_With_Percentage_df['Time'] / Layers_With_Percentage_df['Time_sum'] * 100
    print(Layers_With_Percentage_df[(Layers_With_Percentage_df["Graph"]=="alex") & (Layers_With_Percentage_df["Freq"]==0) & (Layers_With_Percentage_df["Component"]=="G") & (Layers_With_Percentage_df["Freq_Host"]==0)]["Time_Percentage"].sum())
    Layers_With_Percentage_df.to_csv(Layers_With_Percentage_csv, index=False)
    Layers_Percentage_df=Layers_With_Percentage_df.groupby(['Graph', 'Component','Layer','Metric'])['Time_Percentage'].mean().reset_index()
    print(Layers_Percentage_df)
    #Layers_Percentage_df.to_csv(Layers_Percentage_csv, index=False)
    pivot_df = Layers_Percentage_df.pivot_table(index=['Graph', 'Layer', 'Metric'], columns='Component', values='Time_Percentage')
    pivot_df.columns = ['Time_Percentage_{}'.format(col) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    
    pivot_df['Time_Percentage_Average'] = pivot_df[['Time_Percentage_B', 'Time_Percentage_G', 'Time_Percentage_L']].mean(axis=1)
    
    pivot_df = pivot_df.groupby(['Graph', 'Layer']).sum().reset_index()
    pivot_df.to_csv(Layers_Percentage_csv, index=False)
    print(pivot_df)


def ab():
    rr='ab'
    print(f'Command is: {rr}')
    p = subprocess.Popen(rr.split())
    p.communicate()
    while(p.returncode):
        print('ab not successful next try after 10s ...')
        time.sleep(10)
        p = subprocess.Popen(rr.split())
        p.communicate()  


########################## Run a Config on board ############################
def Run_Graph(ALL_Freqs, run_command, myoutput, blocking=True):
    print(run_command)
    p = subprocess.Popen(run_command.split(),stdout=myoutput,stderr=myoutput, stdin=subprocess.PIPE, text=True)
    time.sleep(5)
    for Freqs in ALL_Freqs:       
        p.stdin.write(f'{Freqs}\n')
        p.stdin.flush()
        
        '''while p.poll() is None:
            # check if the subprocess is ready to accept input
            rlist, _, _ = select.select([p.stdin], [], [], 1)
            if rlist:
                break'''
        
        time.sleep(8)
    
    p.stdin.write("end\n")
    p.stdin.flush()
    if blocking:
        p.wait()


############################# Parse power file #################################
def Read_Power(file_name):#(graph,file_name,frqss):
    f=open(file_name)
    lines=f.readlines()
    f.close()
    #print(len(lines))
    powers=[]
    pin_last=0
    c=0
    tts=[]
    for l in lines:
        c=c+1
        #print(f'line: {l}')
        try:
            values=l.split(',')
            if len(values) < 3 :
                powers=[]
                pin_last=0
                print(f'Ignoring line {c}: {values}')
                continue
            if not values[0].isnumeric():
                powers=[]
                pin_last=0
                print(f'Ignoring line {c}: {values}')
                continue
            v=float(values[0].strip())  
            if v!=pin_last:
                #print(f'pin value changed to {v}')
                if len(powers):
                    tts.append(len(powers[-1]))
                    powers[-1]=sum(powers[-1])/len(powers[-1])
                powers.append([float(values[2].strip())])
                pin_last=v
                #print(f'appending {float(values[2].strip())} in line {c}')
                #input('salam')
            else: 
                if len(powers):
                    #print(f'Adding {float(values[2].strip())}')
                    powers[-1].append(float(values[2].strip()))
        except:
            print(f'Error in parse power line {c}')
    #print(f'Power first run was {powers[0]}')
    #powers=powers[2:-1:2]
    #without first try run in armcl (So no need to remove first power data)
    #print(f'powers before last aggregation:{powers}')
    tts.append(len(powers[-1]))
    powers[-1]=sum(powers[-1])/len(powers[-1])
    #print(f'powers:{powers}')
    #powers=powers[0:-1:2]
    print(f'number of intervals: {len(tts)}')
    print(f'number of samples in each interval: {tts}')
    
    return powers,tts


## Convert freqs list to string
def format_freqs(fs=[ [ [7],[6],[4],[3,6],[4],[5],[6],[7] ], [] ]):
        formated_fs=[]
        for f in fs:
            ff = '-'.join(['[' + str(sublist[0]) + ',' + str(sublist[1]) + ']' if len(sublist) > 1 else str(sublist[0]) for sublist in f])
            #print(ff)
            formated_fs.append(ff)
        return formated_fs


### This is common function to run a case
## Remember to modify ARMcL code based on your desire
def Profile(_ff=[[[0],[1],[2],[3,6],[4],[5],[6],[7]]],_n=n,order='BBBGBBBB',graph="alex",pwr="pwr.csv",tme="temp.txt", caching=True, kernel_c=96):
    if os.path.isfile(pwr) and os.path.isfile(tme) and caching:
        print("loading existed files")
        return 
    
    ff=format_freqs(_ff)
    print(f'\n\nformatted freqs:\n {ff}')
    os.system(f"PiPush {cnn_dir}/build/examples/LW/{cnn[graph]} test_graph/")
    os.system('adb shell "echo 0 > /sys/class/gpio/gpio157/value"')
    time.sleep(4)
    Power_monitoring = threading.Thread(target=Arduino_read.run,args=(pwr,))
    Power_monitoring.start()
    rr=f"PiTest build/examples/LW/{cnn[graph]} test_graph/ CL {params[graph][0]} {params[graph][1]} {params[graph][2]} {_n} 0 0 100 100 {order} 1 2 4 Alex B B --kernel_c={kernel_c}"
    oo=open(tme,'w+')
    Run_Graph(ff,rr,oo,True)
    time.sleep(2)
    Power_monitoring.do_run = False
    oo.close()


# +
#### Function for parse the log output of armcl for extracting transfer time between components
#### Be careful to change ARMCL code so that there is no wait between layers (to be real tranfer time)
#transfer[g][layer][c_dest][c_source][t/pwr]
def Parse_Transfer_Layers(timefile,graph="alex",order="BGBGBGBG"):
    trans_df=pd.DataFrame(columns=["Graph", "Layer", "Dest", "Src", "Time"])
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
                #transfer[graph][k][order[k]][order[k-1]]=value
                trans_df.loc[len(trans_df)]={"Graph":graph, "Layer":k, "Dest":order[k], "Src":order[k-1],"Time":value}       
    print(trans)   
    return trans_df
 
#### Run a graph for measuring trasfer times of real layers
#### As transfer time of real layers is small, it does not profile power
def Profile_Transfer_Layers(ff=["7-6-4-[3,6]-4-5-6-7"],_n=n,order='BBBGBBBB',graph="alex",tme="temp.txt",caching=False):
    if os.path.isfile(tme) and caching:
        print("loading existed files")
        return 
    
    rr=f"PiTest build/examples/LW/{cnn[graph]} test_graph/ CL {params[graph][0]} {params[graph][1]} 1 {_n} 0 0 100 100 {order} 1 2 4 Alex B B"
    oo=open(tme,'w+')
    Run_Graph(ff,rr,oo,True)
    #time.sleep(2)
    oo.close()


# -

### Run different order configuration to profile transfer time of real layers with min freqs
### It calls profile_Transfer_Layers and Parse_Transfer_Layers functions
def Profile_Transfer_Time(graph="alex"):
    os.system(f"PiPush /home/ehsan/UvA/ARMCL/Rock-Pi/ComputeLibrary_64_CPUGPULW/build/examples/LW/{cnn[graph]} test_graph/")
    time.sleep(5)
    global Transfers_df
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
        Transfers_logs.mkdir(parents=True, exist_ok=True)
        timefile=f'{Transfers_logs}/transfer_{graph}_'+_order+'.txt'
        Profile_Transfer_Layers(["min"],n,_order,graph,timefile,caching=True)
        time.sleep(2)
        trans_df=Parse_Transfer_Layers(timefile,graph,_order)
        print(trans_df)
        Transfers_df=pd.concat([Transfers_df,trans_df],ignore_index=True)
        #Transfers_df=Transfers_df.append(trans_df,ignore_index=True)
        Transfers_df.to_csv(Transfers_csv,index=False)


#### Run the profile_transfer_time function to profile transfer time of real layers with minimum freqs
def Run_Profile_Transfer_Time():
    global Transfers_df
    for graph in graphs:
        if Transfers_df[Transfers_df["Graph"]==graph].shape[0]==0:
            Profile_Transfer_Time(graph)


### Parse the log results to extract timing paramters for layer times
def Parse(timefile,graph,order,frqss):
    time_df=pd.DataFrame(columns=["Graph", "Component", "Freq", "Freq_Host", "Layer", "Metric", "Time"])
    with open(timefile) as ff:
        lines=ff.readlines()
    freq_indx=0
    freqs=frqss[0]
    t={}
    ins={}
    outs={}
    trans={}
    parts={} 
    for l in lines:     
        if "Profiling these DVFS settings finised" in l:
            print(f'Tasks:{t}')
            print(f'Inputs:{ins}')
            print(f'trans:{trans}')
            print(f'outs:{outs}')
            for layer in t:
                cmp=order[layer]
                freq=freqs[layer]
                Host_freq=-1
                if order[layer]=="G":
                    Host_freq=freq[1]
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"task","Time":t[layer]}
                if layer in ins:
                    time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"in","Time":ins[layer]}
                if layer in outs:
                    time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"out","Time":outs[layer]}
                if layer in trans:
                    time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"trans","Time":trans[layer]}
            t={}
            ins={}
            outs={}
            trans={}
            parts={}
            
        pattern = r".* Running Graph with .* LW DVFS"
        if re.match(pattern,l):
            freqs=frqss[freq_indx]
            print(f'Next freq:{freqs}')
            freq_indx=freq_indx+1
        match = re.search(r"Layer Number: (\d+) \t time: (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            t[k]=value 
        match = re.search(r"input time of layer (\d+) : (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value =float(match.group(2))
            ins[k]=value
        match = re.search(r"output time of layer (\d+) : (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            outs[k]=value
        match = re.search(r"transfer_time of layer (\d+) : (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            trans[k]=value
            
        match = re.search(r"total(\d+)_time:(\d+\.\d+)", l)
        if match:
            k = match.group(1)
            value = float(match.group(2))
            parts[k]=value
    
    return time_df  


## This is like Parse but for syntethic (test_transfer) graph 
## In this graph task() is comment and just transfer time and power is explored 
def Parse_transfer_graph(timefile,graph,order,frqss):
    with open(timefile) as ff:
        lines=ff.readlines()
    freq_indx=0
    freqs=frqss[0]
    t={}
    ins={}
    outs={}
    trans={}
    parts={}
    prof_trans=[]
    transfer_df_time = pd.DataFrame(columns=['order', 'freq', 'transfer_time'])
    
    for l in lines:     
        if "Profiling these DVFS settings finised" in l:
            print(f'Tasks:{t}')
            print(f'Inputs:{ins}')
            print(f'trans:{trans}')
            print(f'outs:{outs}')
            prof_trans=trans
            transfer_df_time.loc[len(transfer_df_time)]={'order':order, 'freq': tuple(freqs), 'transfer_time':trans[1]}
            t={}
            ins={}
            outs={}
            trans={}
            parts={}
            
        pattern = r".* Running Graph with .* LW DVFS"
        if re.match(pattern,l):
            freqs=frqss[freq_indx]
            print(f'Next freq:{freqs}')
            freq_indx=freq_indx+1
        match = re.search(r"Layer Number: (\d+) \t time: (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            t[k]=value
            
        
        match = re.search(r"input time of layer (\d+) : (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value =float(match.group(2))
            ins[k]=value
        match = re.search(r"output time of layer (\d+) : (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            outs[k]=value
        match = re.search(r"transfer_time of layer (\d+) : (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            trans[k]=value
            
        match = re.search(r"total(\d+)_time:(\d+\.\d+)", l)
        if match:
            k = match.group(1)
            value = float(match.group(2))
            parts[k]=value
    return prof_trans,transfer_df_time


### This is for parse power for layers of real graphs 
### So the in ARMCL you need to add a sleep between tasks to be catched with power setup
### As here transfer power is not capture adding this sleep does not affect these data
### but it is neccessary as transfering is less than 1.4 ms (sample interval of power setup)
def Parse_Power(file_name,graph,order,frqss):
    pwr_df=pd.DataFrame(columns=["Graph", "Component", "Freq", "Freq_Host", "Layer", "Metric", "Power"])
    NL=NLayers[graph]
    powers,tts=Read_Power(file_name)
    input_pwrs=[]
    task_pwrs={}
    #for each freq: NL*2(which is input-layer pairs)
    #after each freq we have an excess [0] and [1] interval, so:
    nn=((2*NL*n)+2)
    nnn=nn*len(frqss)
    if len(powers)!=nnn:
        print(f"bad power size:{len(powers)}")
        print(f'Expected size is:NFreqx((2xNLxn)+2) which is {len(frqss)}x((2x{NL}x{n})+2)=nnn')
        input("what")
        return
    print(f'len powers is {len(powers)}')
    #data[g][c][f][fbig][layer][m]
    for i,freq in enumerate(frqss):
        pwrs=powers[i*nn:(i+1)*nn-2]
        input_pwrs=pwrs[0::2*NL]
        print(f'\n\n\n************\nInput powers with len {len(input_pwrs)}')
        input_pwrs=sum(input_pwrs)/len(input_pwrs)
        for layer,j in enumerate(range(1,2*NL,2)):
            Host_freq=-1
            if order[layer]=="G":
                Host_freq=freq[layer][1]
                
            if layer==0:
                pwr_df.loc[len(pwr_df)]={"Graph":graph, "Component":order[layer],"Freq":freq[layer][0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"in","Power":input_pwrs}
                print(f'setting power for {graph}-{order}-{layer}-{freq[layer][0]}-in-power-->{input_pwrs}')
            
            task_pwrs[layer]=pwrs[j::2*NL]
            print(f'len layer power {len(task_pwrs[layer])}')
            task_pwrs[layer]=sum(task_pwrs[layer])/len(task_pwrs[layer])
            pwr_df.loc[len(pwr_df)]={"Graph":graph, "Component":order[layer],"Freq":freq[layer][0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"task","Power":task_pwrs[layer]}
            print(f'setting power for {graph}-{order}-{layer}-{freq[layer][0]}-task-power->{task_pwrs[layer]}')
    return pwr_df


### This is for parsing the power of syntethic (test_transfer) graph to measure transferig power
### For this it is necessary to remove sleep in transfer to be real one
def Parse_Power_Transfer_graph(file_name,graph,order,frqss):
    NL=NLayers[graph]
    powers,tts=Read_Power(file_name)
    input_pwrs=[]
    task_pwrs={}
    trans_pwrs={}
    transfer_df_pwr = pd.DataFrame(columns=['order', 'freq', 'transfer_power'])
    #for each freq: NL*2(which is input-layer pairs)
    #after each freq we have a excess [0]and[1]interval so:
    nn=((2*NL*n)+2)
    nnn=nn*len(frqss)
    if len(powers)!=nnn:
        print(f"bad power size: {len(powers)}")
        input("what")
        return
    print(f'len powers is {len(powers)}')
    #data[g][c][f][fbig][layer][m]
    for i,freq in enumerate(frqss):
        pwrs=powers[i*nn:(i+1)*nn-2]
        #print(f'powers for freq :{freq}: {powers}')
        input_pwrs=pwrs[0::2*NL]
        print(f'\n\n\n************\nInput powers with len {len(input_pwrs)}')
        input_pwrs=sum(input_pwrs)/len(input_pwrs)
        #input_pwrs=sum(input_pwrs)
        for layer,j in enumerate(range(1,2*NL,2)):
            task_pwrs[layer]=pwrs[j::2*NL]
            print(f'len layer {layer} power {len(task_pwrs[layer])}')
            task_pwrs[layer]=sum(task_pwrs[layer])/len(task_pwrs[layer])
            if layer>0:
                trans_pwrs[layer]=pwrs[j-1::2*NL]
                print(f'len layer {layer} trans power {len(trans_pwrs[layer])}')
                trans_pwrs[layer]=sum(trans_pwrs[layer])/len(trans_pwrs[layer])
            if layer==0:
                #d[layer]["in"]["Power"]=input_pwrs
                print(f'setting power for {graph}-{order}-{layer}-{freq[layer][0]}-in-power-->{input_pwrs}')
            else:
                #d[layer]["trans"]["Power"]=trans_pwrs[layer]
                print(f'setting power for {graph}-{order}-{layer}-{freq[layer][0]}-trans-power->{trans_pwrs[layer]}')
                transfer_df_pwr.loc[len(transfer_df_pwr)]={'order':order, 'freq': tuple(freq), 'transfer_power':trans_pwrs[layer]}
            #d[layer]["task"]["Power"]=task_pwrs[layer]
            print(f'setting power for {graph}-{order}-{layer}-{freq[layer][0]}-task-power->{task_pwrs[layer]}')
    return trans_pwrs,transfer_df_pwr


### This function is for profiling time and power of tasks in real graphs
### In ARMCL you need to sleep between tasks 
### As transfer time for most cases is less than 1.4 ms (sample interval of power measurement setup)
def Profile_Task_Time(graph):
    global Layers_df
    NL=NLayers[graph]
    orders=["B","G","L"]
    for _order in orders:
        frqss=[]
        NF=NFreqs[_order]
        if _order=="G":
            Nbig=NFreqs["B"]
            for f in range(NF):
                for fbig in range(Nbig):
                    layer_f=[f,fbig]
                    layers_f=NL*[layer_f]
                    frqss.append(layers_f)
        else:
            for f in range(NF):
                layer_f=[f]
                layers_f=NL*[layer_f]
                frqss.append(layers_f)
        print(f'graph:{graph} order:{_order} freqs:{frqss}')
        
        
        order=NL*_order
        Layers_logs.mkdir(parents=True, exist_ok=True)
        pwrfile=f'{Layers_logs}/power_{graph}_'+order+'.csv'
        timefile=f'{Layers_logs}/time_{graph}_'+order+'.txt'
        Profile(frqss,n,order,graph,pwrfile,timefile,caching=True)
        time.sleep(10)
        time_df=Parse(timefile,graph,order,frqss)
        power_df=Parse_Power(pwrfile,graph,order,frqss)
        #time_df['Freq'] = time_df['Freq'].apply(lambda x: tuple(x))
        #power_df['Freq'] = power_df['Freq'].apply(lambda x: tuple(x))
        merged_df = pd.merge(power_df, time_df, on=['Graph', 'Component', 'Freq','Freq_Host','Layer','Metric'])
        Layers_df=pd.concat([Layers_df,merged_df], ignore_index=True)
        Layers_df.to_csv(Layers_csv,index=False)


#when reading:
#test=pd.read_csv("data_df.csv",index_col=0)
#or you can use df.to_csv with index=False argument
def Profiling_Layers():
    for graph in graphs[::1]:
        if Layers_df[Layers_df["Graph"]==graph].shape[0]==0:
            Profile_Task_Time(graph)   


def Analyze(graph_name=graphs,metric=['task','in','out','trans'],comp=['G','B','L'],
            freq_h=[float('nan')],f=range(10),layers=range(40),index=['Layer'],columns=['Freq'],parameter='Time'):

    # Group the filtered DataFrame by the 'Layer' and 'Freq' columns, and aggregate the 'Time' column using the 'mean()' function
    grouped_df = Layers_df[(Layers_df['Graph'].isin(graph_name)) & 
                    (Layers_df['Metric'].isin(metric)) & 
                    (Layers_df['Component'].isin(comp)) & 
                    (Layers_df['Freq_Host'].isin(freq_h))&
                    (Layers_df['Layer'].isin(layers)) ].groupby(index+columns)['Time','Power'].sum().reset_index()
    grouped_df['Energy']=grouped_df['Power']*grouped_df['Time']/1000.0
    grouped_df['Energy-Efficiency']=1000.0/(grouped_df['Energy'])
    # Create a pivot table to rearrange the data for plotting
    pivot_table = pd.pivot_table(grouped_df, values=parameter, index=index, columns=columns)
    pivot_table.plot(kind='bar', stacked=False, figsize=(30, 6))
    plt.title(f'{metric} {parameter} vs {columns} for {graph_name}')
    plt.xlabel(f'{index}')
    plt.ylabel(f'{metric} {parameter}')
    plt.show()
    return pivot_table
Analyze(graph_name=['alex'],metric=['task'],comp=['G'],freq_h=[0],index=['Layer'],columns=['Freq'])


def _Analyze_Components():
    grouped_df = Layers_df[(Layers_df['Graph'].isin(['alex'])) & 
                        (Layers_df['Metric'].isin(['in','task'])) & 
                        (Layers_df['Freq'].isin([0])) & 
                        (Layers_df['Freq_Host'].isin([0,-1]))&
                        (Layers_df['Layer'].isin(range(10))) ].groupby(['Component','Layer','Metric'])\
                        ['Time','Power'].sum().reset_index()

    grouped_df['Energy']=grouped_df['Time']*grouped_df['Power']/1000.0
    print(grouped_df)

    '''grouped_df['Layer'] = grouped_df['Layer'].where(grouped_df['Metric'] != 'in', 'input')
    grouped_df = grouped_df.drop('Metric', axis=1)
    print(grouped_df)'''

    aggregations = {
        'Time': 'sum',
        'Power': 'mean',
        'Energy': 'sum'
    }
    grouped_df = grouped_df.groupby(['Component', 'Layer']).agg(aggregations).reset_index()
    print(grouped_df)

    pivot_df = grouped_df.pivot_table(index=['Layer'], columns='Component', values='Energy')
    pivot_df.columns = ['Energy_{}'.format(col) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    print(pivot_df)
    pivot_df.plot(x='Layer',y=['Energy_G','Energy_B','Energy_L'],kind='bar')


def Analyze_Components(g=['alex']):

    grouped_df = Layers_df[(Layers_df['Graph'].isin(g)) & 
                        (Layers_df['Metric'].isin(['in','task'])) &  
                        (Layers_df['Freq_Host'].isin([0,-1]))&
                        (Layers_df['Layer'].isin(range(10))) ].groupby(['Freq','Component','Layer','Metric'])\
                        ['Time','Power'].sum().reset_index()

    grouped_df['Energy']=grouped_df['Time']*grouped_df['Power']/1000.0
    print(grouped_df)

    '''grouped_df['Layer'] = grouped_df['Layer'].where(grouped_df['Metric'] != 'in', 'input')
    grouped_df = grouped_df.drop('Metric', axis=1)
    print(grouped_df)'''

    aggregations = {
        'Time': 'sum',
        'Power': 'mean',
        'Energy': 'sum'
    }
    grouped_df = grouped_df.groupby(['Freq','Component', 'Layer']).agg(aggregations).reset_index()
    print(grouped_df)

    pivot_df = grouped_df.pivot_table(index=['Layer','Freq'], columns='Component', values=['Time', 'Energy'])
    pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
    #pivot_df = grouped_df.pivot_table(index=['Layer','Freq'], columns='Component', values='Energy')
    #pivot_df.columns = ['Energy_{}'.format(col) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    print(pivot_df)
    pivot_df.to_csv("Components.csv",index=False)
    
   # Group by 'Layer' and get the maximum valid frequency for each parameter
    max_freq_B = pivot_df[pivot_df['Energy_B'].notna()].groupby('Layer')['Freq'].max()
    max_freq_G = pivot_df[pivot_df['Energy_G'].notna()].groupby('Layer')['Freq'].max()
    max_freq_L = pivot_df[pivot_df['Energy_L'].notna()].groupby('Layer')['Freq'].max()

    # Extract the values at the maximum valid frequency for each parameter
    freq_df = pd.DataFrame({
        'Layer': max_freq_B.index,
        'Energy_B_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_B.values)]['Energy_B'].values,
        'Time_B_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_B.values)]['Time_B'].values,
        'Energy_G_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_G.values)]['Energy_G'].values,
        'Time_G_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_G.values)]['Time_G'].values,
        'Energy_L_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_L.values)]['Energy_L'].values,
        'Time_L_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_L.values)]['Time_L'].values})
    display(freq_df)
    energy_cols = ['Energy_G_MaxFreq', 'Energy_B_MaxFreq', 'Energy_L_MaxFreq']
    energy_plot = freq_df.plot(x='Layer', y=energy_cols, kind='bar', title='Energy for Freq Max')
    energy_plot.set_xlabel('Layer')
    energy_plot.set_ylabel('Energy')
    plt.show()

    # Plot Time columns
    time_cols = ['Time_G_MaxFreq', 'Time_B_MaxFreq', 'Time_L_MaxFreq']
    time_plot = freq_df.plot(x='Layer', y=time_cols, kind='bar', title='Time for Freq Max')
    time_plot.set_xlabel('Layer')
    time_plot.set_ylabel('Time')
    plt.show()
    
    for freq in pivot_df['Freq'].unique():
        # Filter dataframe for the current Freq value
        freq_df = pivot_df[pivot_df['Freq'] == freq]
        display(freq_df)

        # Plot Energy columns
        energy_cols = ['Energy_G', 'Energy_B', 'Energy_L']
        energy_plot = freq_df.plot(x='Layer', y=energy_cols, kind='bar', title='Energy for Freq {}'.format(freq))
        energy_plot.set_xlabel('Layer')
        energy_plot.set_ylabel('Energy')
        plt.show()

        # Plot Time columns
        time_cols = ['Time_G', 'Time_B', 'Time_L']
        time_plot = freq_df.plot(x='Layer', y=time_cols, kind='bar', title='Time for Freq {}'.format(freq))
        time_plot.set_xlabel('Layer')
        time_plot.set_ylabel('Time')
        plt.show()
Analyze_Components2(g=['google'])

# +
import random
def generate_random_strings(n, num_strings):
    chars = ['L', 'B', 'G']
    random_strings = []
    for _ in range(num_strings):
        random_string = ''.join(random.choice(chars) for _ in range(n))
        random_strings.append(random_string)
    return random_strings

n = 8  # replace with desired length of the random strings
num_strings = 1000  # replace with desired number of random strings
random_strings = generate_random_strings(n, num_strings)
for i, random_string in enumerate(random_strings):
    print(f"Random String {i + 1}: {random_string}")


# -

def Analyze2(graph_name = 'alex'):
    graph_df = Layers_df[Layers_df['Graph'] == graph_name]
    # Group the filtered DataFrame by the 'Layer' and 'Freq' columns, and aggregate the 'Time' column using the 'mean()' function
    #grouped_df = graph_df[graph_df['Metric'] == 'task'].groupby(['Graph', 'Component', 'Freq', 'Layer'])['Time'].sum()
    grouped_df = graph_df[graph_df['Metric'] == 'task'].groupby(['Graph', 'Component', 'Layer', 'Freq'])['Time'].sum().reset_index()
    print(grouped_df)
    # Create a pivot table to rearrange the data for plotting
    pivot_table = pd.pivot_table(grouped_df,index=['Graph', 'Component', 'Layer'], columns='Freq', values='Time')
    # Generate a line plot to visualize the effect of frequency on task timing for different layers
    pivot_table.plot(kind='bar', stacked=False, figsize=(10, 6))
    plt.title(f'Task Timing vs Frequency for {graph_name}')
    plt.xlabel('Layer')
    plt.ylabel('Task Timing (ms)')
    plt.show()
    return pivot_table
#Analyze2()


def Value(graph,comp,freq,layer,metric,attr):
    if len(freq)==1 or comp!='G':
        return Layers_df_indexed.loc[(graph, comp, freq[0], layer, metric, np.NaN), attr]
    if len(freq)==2:
        return Layers_df_indexed.loc[(graph, comp, freq[0], layer, metric, freq[1]), attr]
    else:
        return -1


def Comp_Cost(g='alex',fn=[[0],[1],[2],[3],[4],[5],[6],[7]],cmps=8*'B',dvfs_delay=3.5, debug=False):
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
    tfn=Value(g,cmps[0],fn[0],0,'in','Time')
    tfc=Value(g,cmps[0],fc[0],0,'in','Time')
    t=tfc
    if tfc > dvfs_delay:
        t=tfn - (dvfs_delay/tfc)*tfn + dvfs_delay  
    if debug:
        print(f'in:{0}, next_freq:{fn[0]} time(next_freq):{tfn} cur_freq:{fc[0]} time(cur_freq):{tfc} time:{t}')      
    tt+=t
    tt_nodvfs+=tfn
    
    #comp power
    pfn=Value(g,cmps[0],fn[0],0,'in','Power')
    pfc=Value(g,cmps[0],fc[0],0,'in','Power') 
    e=t*pfc
    if t > dvfs_delay:
        e=dvfs_delay*pfc + (t-dvfs_delay)*pfn
    e_nodvfs= tfn*pfn
    ee+=e
    ee_nodvfs+=e_nodvfs
    if debug:
        print(f'in:{0}, next_freq:{fn[0]} power(next_freq):{pfn} cur_freq:{fc[0]} power(cur_freq):{pfc} energy:{e}')
        
    for i in range(0,len(fn)-1):
        tfn=Value(g,cmps[i+1],fn[i+1],i,'task','Time')
        tfc=Value(g,cmps[i+1],fc[i+1],i,'task','Time')
        t=tfc
        if tfc > dvfs_delay:
            t=tfn - (dvfs_delay/tfc)*tfn + dvfs_delay
        if debug:
            print(f'layer:{i}, next_freq:{fn[i+1]} time(next_freq):{tfn} cur_freq:{fc[i+1]} time(cur_freq):{tfc} time:{t}')
        tt+=t
        tt_nodvfs+=tfn
        
        pfn=Value(g,cmps[i+1],fn[i+1],i,'task','Power')
        pfc=Value(g,cmps[i+1],fc[i+1],i,'task','Power') 
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


def Transfer_Info(p1='B',p2='G',f1=[4],f2=[3,4]):
    global Transfer_Freq_df
    f1=[int(i) for i in f1]
    f2=[int(i) for i in f2]
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
    row=Transfer_Freq_df[ (Transfer_Freq_df['freq']==str(freqs)) & (Transfer_Freq_df['order']==order)]
    #print(row)
    power=row['transfer_power'].iloc[0]
    coef_t=row['time_ratio'].iloc[0]  
    return power,coef_t
#a,b=transfer_info('G','B',[2.0, 7.0],[7.0])
#Transfer_Freq_df


def Comm_Cost(g='alex',fn=[[0],[1],[2],[3],[4],[5],[6],[7]],cmps=8*'B',dvfs_delay=3.5, debug=False):
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
            #transfer_time=transfer_times[g][i][cmps[i]][cmps[i-1]]
            transfer_time=Transfers_df[(Transfers_df["Graph"]==g) &
                                       (Transfers_df["Layer"]==i) &
                                       (Transfers_df["Dest"]==cmps[i]) &
                                       (Transfers_df["Source"]==cmps[i-1])]["Time"]
            transfer_power,time_ratio=Transfer_Info(p1=cmps[i-1],p2=cmps[i],f1=fc[i-1],f2=fc[i])
        
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


def Test():
    _fs=[ [ [0],[1],[2],[3],[4],[5],[6],[7] ],
         [ [7],[6],[5],[4],[3],[2],[1],[0] ] ]
    ord='BBBBBBBB'
    _g="alex"
    for fs in _fs:
        Profile(_ff=[fs], _n=n, order=ord, graph=_g, pwr="pwr.csv", tme="temp.txt",caching=False)
        time=Parse(timefile="temp.txt", graph=_g, order=ord, frqss=[fs])
        power=Parse_Power(pwrfile="pwr.csv", graph=_g, order=ord, frqss=[fs])
        _dvfs_delay=3.5
        #np.reshape(fs,-1)
        cmp=Comp_Cost(g=_g,fn=fs,cmps=ord,dvfs_delay=_dvfs_delay, debug=False)
        cmm=Comm_Cost(g=_g,fn=fs,cmps=ord,dvfs_delay=_dvfs_delay, debug=False)
        print(time)
        print(power)
        print(cmp)
        print(cmm)
#test()


def Parse_Power_total(file_name,graph,order,frqss):
    powers,tts=Read_Power(file_name)
    power_df = pd.DataFrame(columns=['graph', 'order', 'freq', 'input_power','task_power'])
    NL=1
    nn=((2*NL*n)+2)
    nnn=nn*len(frqss)
    if len(powers)!=nnn:
        print(f"bad power size: {len(powers)}")
        input("what")
        return
    print(f'len powers is {len(powers)}')
     
    for i,freq in enumerate(frqss):
        pwrs=powers[i*nn:(i+1)*nn-2]
        input_pwrs=pwrs[0::2*NL]
        task_pwrs=pwrs[1::2*NL]
        input_pwrs=sum(input_pwrs)/len(input_pwrs)
        task_pwrs=sum(task_pwrs)/len(task_pwrs)   
        print(f'\n\n\n************\nInput powers: {input_pwrs}')
        print(f'setting power for {graph}-{order}-{freq}-task-power->{task_pwrs}')
        power_df.loc[len(power_df)]={'graph':graph, 'order':order, 'freq': tuple(freq), 'input_power':input_pwrs, 'task_power':task_pwrs}
    return power_df


def Parse_total(timefile,graph,order,frqss):
    with open(timefile) as ff:
        lines=ff.readlines()
    freq_indx=0
    freqs=frqss[0]
    input_time=-1
    parts=[]
    df_time = pd.DataFrame(columns=['graph', 'order', 'freq', 'input_time', 'task_time', 'total_time'])
    for l in lines:        
        if "Profiling these DVFS settings finised" in l:
            print(f'Input_time:{input_time}')
            s=sum(parts)
            print(f'parts:{parts}, sum:{s}')            
            
            df_time.loc[len(df_time)]={'graph':graph, 'order':order, 'freq': tuple(freqs), 'input_time':input_time, 'task_time':s-input_time, 'total_time':s}
            input_time=-1
            parts=[]
            
        pattern = r".* Running Graph with .* LW DVFS"
        if re.match(pattern,l):
            freqs=frqss[freq_indx]
            print(f'Next freq:{freqs}')
            freq_indx=freq_indx+1
            
        match = re.search(r"input0_time:(\d+\.\d+)", l)
        if match:
            value = float(match.group(1))
            input_time=value
            
        match = re.search(r"total(\d+)_time:(\d+\.\d+)", l)
        if match:
            k = match.group(1)
            value = float(match.group(2))
            parts.append(value)

    return df_time


def Real_Evaluation(g="alex",ord='GBBBBBBB',_fs=[ [ [0,0],[0],[0],[0],[0],[0],[0],[0] ] ]):   
    pf="pwr_whole.csv"
    tf="temp_whole.txt"
    global Evaluations_df
    if Evaluations_csv.exists():
        Evaluations_df=pd.read_csv(Evaluations_csv)
    else:
        Evaluations_df=pd.DataFrame(columns=['graph','order','freq','input_time','task_time','total_time', 'input_power','task_power'])
    
    new_fs=[]
    repeat=False
    #print(evaluations)
    for ff in _fs:
        tpl_f=(tuple(tuple(i) for i in ff))
        #print(f"f:{tpl_f}\nhey:{evaluations['freq'][0]}")
        #input()
        if repeat==False and Evaluations_df[(Evaluations_df['order']==ord) & (Evaluations_df['freq']==str(tpl_f)) & (Evaluations_df['graph']==g)].shape[0]==0:
            new_fs.append(ff)
        else:
            print(f'{ord}, Freq:{ff} already evaluated')

    if len(new_fs)==0:
        return evaluations
    Profile(_ff=new_fs, _n=n, order=ord, graph=g, pwr=pf, tme=tf,caching=False,kernel_c=96*50)
    time_df=Parse_total(timefile=tf, graph=g, order=ord, frqss=new_fs)
    power_df=Parse_Power_total(file_name=pf,graph=g,order=ord,frqss=new_fs)
    power_df['freq'] = power_df['freq'].apply(lambda x: tuple(tuple(i) for i in x))
    time_df['freq'] = time_df['freq'].apply(lambda x: tuple(tuple(i) for i in x))
    merged_df = pd.merge(power_df, time_df, on=['graph', 'order', 'freq'])


    input_time=time_df['input_time'].iloc[0]
    task_time=time_df['task_time'].iloc[0]
    input_power=power_df['input_power'].iloc[0]
    task_power=power_df['task_power'].iloc[0]
    input_e=input_power*input_time
    task_e=task_power*task_time
    total_e=input_e+task_e
    merged_df['input_e']=input_e/1000.0
    merged_df['task_e']=task_e/1000.0
    merged_df['total_e']=total_e/1000.0

    Evaluations_df=pd.concat([Evaluations_df,merged_df], ignore_index=True)
    Evaluations_df.to_csv(Evaluations_csv,index=False)
    return Evaluations_df
#Real_Evaluation(g="alex",ord='GBBBBBBB',_fs=[ [ [4,6],[6],[6],[6],[6],[6],[6],[6] ] ])
#Real_Evaluation(g="alex",ord='BBBBBBBB',_fs=[ [ [0],[1],[2],[3],[4],[5],[6],[7] ] ])

def Transfer_Cost(_order,fs,_kernel_c=96*100):   
    g="test_transfer"
    trans=[]
    trans_pwr=[]
    
    Synthetic_Tranfer_logs.mkdir(parents=True, exist_ok=True)
    pwrfile=f'{Synthetic_Tranfer_logs}/power_{g}_{_order}_{str(_kernel_c)}_{str(fs)}.csv'
    timefile=f'{Synthetic_Tranfer_logs}/time_{g}_{_order}_{str(_kernel_c)}_{str(fs)}.txt'
    
    Profile(_ff=fs, _n=n, order=ord, graph=g, pwr=pwrfile, tme=timefile,caching=True,kernel_c=_kernel_c)
    
    trans,transfer_df_time=Parse_transfer_graph(timefile=timefile, graph=g, order=ord, frqss=fs)

    trans_pwr,trans_pwr_df=Parse_Power_Transfer_graph(file_name=pwrfile,graph=g,order=ord,frqss=fs)
    
    return trans,trans_pwr,trans_pwr_df,transfer_df_time


def Explore_Freq_on_Transfering():

    _fs={'BL':[[[0],[i]] for i in range(NFreqs['L'])],
            'LB':[[[0],[i]] for i in range(NFreqs['B'])],
            'GB':[[[0,i],[i]] for i in range(NFreqs['B'])],
            'LG':[[[0],[0,i]] for i in range(NFreqs['B'])],
            'BG':[[[i],[0,i]] for i in range(NFreqs['B'])]}
    global Transfer_Freq_df
    if Transfer_Freq_df.exists():
        Transfer_Freq_df=pd.read_csv(Transfer_Freq_csv)
    else:
        Transfer_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'SenderFreq','RecFreq' 'transfer_time', 'transfer_power'])
    kernel_cs=[150]
    kernel_cs=[96*i for i in kernel_cs]
    
    #global trans,trans_pwr,trans_pwr_df,transfer_df_time
    for kernel_c in kernel_cs:
        for order in _fs:
            print(f'order:{order}, kernels:{kernel_c}, shape: {Transfer_Freq_df[(Transfer_Freq_df["order"]==orders[order]) & (Transfer_Freq_df["kernels"]==kernel_c)].shape[0]}')
            if Transfer_Freq_df[(Transfer_Freq_df['order']==orders[order]) & (Transfer_Freq_df['kernels']==kernel_c)].shape[0]==0:
                try:
                    trans,trans_pwr,trans_pwr_df,transfer_df_time=Transfer_Cost(_order=order,fs=_fs[order],_kernel_c=kernel_c)
                    #transfer_df_freq.loc[len(transfer_df_freq)] = {"kernels":kernel_c, "c":orders[_c], "transfer_time":transfer_df_time, "transfer_power":trans_pwr_df}
                    trans_pwr_df['freq'] = trans_pwr_df['freq'].apply(lambda x: tuple(tuple(i) for i in x))
                    transfer_df_time['freq'] = transfer_df_time['freq'].apply(lambda x: tuple(tuple(i) for i in x))
                    merged_df = pd.merge(trans_pwr_df, transfer_df_time, on=['order', 'freq'])
                    merged_df['kernels']=kernel_c
                    Transfer_Freq_df=pd.concat([Transfer_Freq_df,merged_df], ignore_index=True)
                    print(f'merged is:\n{merged_df}')
                    print(f'accumulated result is:\n{Transfer_Freq_df}')
                    Transfer_Freq_df.to_csv(Transfer_Freq_csv,index=False)
                    time.sleep(10)
                except:
                    ab()
#Explore_Freq_on_Transfering()


def Plot_Transfer_VS_Data_size(order,freq_mode):
    if freq_mode=="max":
        trans_df=Transfer_Data_Size_Max_Freq_df
    if freq_mode=="min":
        trans_df=Transfer_Data_Size_Min_Freq_df
    
    t=trans_df[trans_df['order']==order].groupby(['kernels']).sum(['transfer_time', 'transfer_power'])
    #p=trans_df[trans_df['c']==order].groupby(['kernels'])['trans_power'].sum()
    print(f'results for {order}:\n{t}')
    #print(p)
    pivot_table_time = pd.pivot_table(t,index=['kernels'], values=['transfer_time'])
    pivot_table_time.plot(figsize=(5,3))
    plt.title(f'{order} time vs Data size')
    plt.xlabel('# Kernels')
    plt.ylabel('time (ms)')
    plt.show()
    pivot_table_power = pd.pivot_table(t,index=['kernels'], values=['transfer_power'])
    pivot_table_power.plot(ylim=[0, 6000],figsize=(5, 3))
    plt.title(f'{order} Power vs Data size')
    plt.xlabel('# Kernels')
    plt.ylabel('Power (mW)')
    plt.show()


# set sleep time between tasks to 0 in ARMCL src/graph/detail/ExecuionHelpers.cpp 
#(check graphmanager.cpp for sure that there is no sleep )
def Explore_Data_Size_on_Transfering(freq_mode="max"):
    global Transfer_Data_Size_Max_Freq_df, Transfer_Data_Size_Min_Freq_df
    g="test_transfer"
    _fs={"max":{"BL":[ [ [0],[5] ] ],
                "LB":[ [ [0],[7] ] ],
                "GB":[ [ [0,7],[7] ] ],
                "LG":[ [ [0],[0,7] ] ],
                "BG":[ [ [7],[0,7] ] ]},
         "min":{"BL":[ [ [0],[0] ] ],
                "LB":[ [ [0],[0] ] ],
                "GB":[ [ [0,0],[0] ] ],
                "LG":[ [ [0],[0,0] ] ],
                "BG":[ [ [0],[0,0] ] ]}
        }
    if freq_mode=="max":
        if Transfer_Data_Size_Max_Freq_csv.exists():
            Transfer_Data_Size_Max_Freq_df=pd.read_csv(Transfer_Data_Size_Max_Freq_csv)
            print(f'max freq trans data:\n{Transfer_Data_Size_Max_Freq_df}')
        else:
            Transfer_Data_Size_Max_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'transfer_time', 'transfer_power'])
        kernel_cs=[10,20,30,40,50,60,70,80,90,100,150,200,250,300,400,500]
        kernel_cs=[96*i for i in kernel_cs]

        for kernel_c in kernel_cs:
            for order in _fs["max"]:
                if Transfer_Data_Size_Max_Freq_df[(Transfer_Data_Size_Max_Freq_df['order']==orders[order]) & 
                                                  (Transfer_Data_Size_Max_Freq_df['kernels']==kernel_c)].shape[0]==0:
                    try:
                        if orders[order][1]=='G' and kernel_c > 150*96:
                            continue
                                                    
                        _fs=_fs["max"][order]
                        trans,trans_pwr,trans_pwr_df,transfer_df_time=Transfer_Cost(_order=order,fs=_fs,_kernel_c=kernel_c)
                        Transfer_Data_Size_Max_Freq_df.loc[len(Transfer_Data_Size_Max_Freq_df)] = {"kernels":kernel_c, "order":orders[_c], "transfer_time":trans[1], "transfer_power":trans_pwr[1]}
                        Transfer_Data_Size_Max_Freq_df.to_csv(Transfer_Data_Size_Max_Freq_csv,index=False)
                        time.sleep(10)
                    except:
                        ab()               

        return Transfer_Data_Size_Max_Freq_df
    
    if freq_mode=="min":
        if os.path.isfile(Transfer_Data_Size_Min_Freq_csv):
            Transfer_Data_Size_Min_Freq_df=pd.read_csv(Transfer_Data_Size_Min_Freq_csv)
            print(f'min freq trans data:\n{Transfer_Data_Size_Min_Freq_df}')
            #return transfer_df_min
        else:
            Transfer_Data_Size_Min_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'transfer_time', 'transfer_power'])
        kernel_cs=[10,20,30,40,50,60,70,80,90,100,150,200,250,300,400,500]
        kernel_cs=[96*i for i in kernel_cs]


        for kernel_c in kernel_cs:
            for order in _fs["min"]:
                if Transfer_Data_Size_Min_Freq_df[(Transfer_Data_Size_Min_Freq_df['order']==orders[order]) & 
                                                  (Transfer_Data_Size_Min_Freq_df['kernels']==kernel_c)].shape[0]==0:
                    try:
                        if orders[order][1]=='G' and kernel_c > 150*96:
                            continue
                        _fs=_fs["min"][order]
                        trans,trans_pwr,trans_pwr_df,transfer_df_time=Transfer_Cost(_order=order,fs=_fs,_kernel_c=kernel_c)
                        Transfer_Data_Size_Min_Freq_df.loc[len(Transfer_Data_Size_Min_Freq_df)] = {"kernels":kernel_c, "order":orders[_c], "transfer_time":trans[1], "transfer_power":trans_pwr[1]}
                        Transfer_Data_Size_Min_Freq_df.to_csv(Transfer_Data_Size_Min_Freq_csv,index=False)
                        time.sleep(10)
                    except:
                        ab()               

        return Transfer_Data_Size_Min_Freq_df


def Run_Explore_Data_Size_on_Transfering(_freq_mode="max"):
    orders={0:'BL', 1:'LB', 2:'GB', 3:'LG', 4:'BG'}
    Explore_Data_Size_on_Transfering(freqs_mode=_freq_mode)
    for i in orders:
        Plot_Transfer_VS_Data_size(order=orders[i],freq_mode=_freq_mode)
#Run_Explore_Data_Size_on_Transfering(_freq_mode="max")
#Run_Explore_Data_Size_on_Transfering(_freq_mode="min")


#def main():
if True:
    Load_Data()
    
    print('\n\n\n\n***************Run_Profile_Transfer_Time\n')
    input('Make sure to set profile mode to PROFILE_MODE_TRANSFER_TIMES in ExecutionHelpers.cpp')
    Run_Profile_Transfer_Time()
    
    print('\n\n\n\n***************Profiling_Layers\n')
    input('Make sure to set profile mode to PROFILE_MODE_LAYERS in ExecutionHelpers.cpp')
    Profiling_Layers()
    
    
    print('\n\n\n\n***************Explore_Freq_on_Transfering\n')
    input('Make sure to set profile mode to PROFILE_MODE_SYNTHETIC_TRANSFERS in ExecutionHelpers.cpp')
    Explore_Freq_on_Transfering()
    
    print('\n\n\n\n***************Run_Explore_Data_Size_on_Transfering(Max)\n')
    Run_Explore_Data_Size_on_Transfering(_freq_mode="max")
    
    print('\n\n\n\n***************Run_Explore_Data_Size_on_Transfering(min)\n')
    Run_Explore_Data_Size_on_Transfering(_freq_mode="min")
    
    
    print('\n\n\n\n***************Value function for indexing\n')
    Value('alex','B',[7],7,'out','Time')
    Value('alex','G',[0,0],0,'task','Time')
    [Value('alex','B',[i],i,'task','Time') for i in range(0,8)]
    [Value('alex','B',[i-1],i,'task','Time') for i in range(1,8)]
    
    print('\n\n\n\n***************Analyze\n')
    Analyze(graph_name=['alex'],metric=['task'],comp=['G'],freq_h=[0],index=['Layer'],columns=['Freq'])
    Analyze(graph_name=['alex'],metric=['task'],comp=['L'],index=['Layer'],columns=['Freq'],parameter='Energy-Efficiency')
    
    print('\n\n\n\n***************Analyze2\n')
    Analyze2()
    
    print('\n\n\n\n***************Test\n')
    print(f'Real Run time is: 334.5 ms')
    print(f'Real Run time is: 192.7 ms')
    Test()
    
    
    
    print('\n\n\n\n***************Transfer_Info\n')
    a,b=Transfer_Info('G','B',[2.0, 7.0],[7.0])
    print(a,b)
    
    print('\n\n\n\n***************Comm_Cost\n')
    _fn=[[0],[1],[2],[3],[4],[5],[6],[7]]
    Comm_Cost(cmps="LLLBBBBB",debug=True)
    
    print('\n\n\n\n***************Comp_Cost\n')
    Comp_Cost(g="alex",cmps='BBBBBBBB',fn=_fn[::-1])
    
    print('\n\n\n\n***************Real_Evaluation\n')
    input('Make sure to set profile mode to PROFILE_MODE_WHOLE_NETWORK in ExecutionHelpers.cpp')
    Real_Evaluation(g="alex",ord='GBBBBBBB',_fs=[ [ [4,6],[6],[6],[6],[6],[6],[6],[6] ] ])
    Real_Evaluation(g="alex",ord='BBBBBBBB',_fs=[ [ [0],[1],[2],[3],[4],[5],[6],[7] ] ])


