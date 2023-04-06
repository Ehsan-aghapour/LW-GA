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

maxtransfile="transfer_max.csv"
mintransfile="transfer_min.csv"
cnn={
    "alex":"graph_alexnet_n_pipe_npu_lw",
    "google":"graph_googlenet_n_pipe_npu_lw",
    "mobile":"graph_mobilenet_n_pipe_npu_lw",
    "res50":"graph_resnet50_n_pipe_npu_lw",
    "squeeze":"graph_squeezenet_n_pipe_npu_lw",
    "test_transfer":"graph_test_transfer_n_pipe_npu_lw"
}
cnn_dir="/home/ehsan/UvA/ARMCL/Rock-Pi/ComputeLibrary_64_CPUGPULW/"

graphs=["alex", "google", "mobile", "res50", "squeeze"]
NLayers={"alex":8, "google":11, "mobile":14, "res50":18, "squeeze":10, "test_transfer":2}
NFreqs={"L":6, "B":8, "G":5}
Metrics=["in","task","out","trans"]
n=100
params={"alex":(1,1,1), "google":(2,2,1), "mobile":(2,3,1), "res50":(2,4,1), "squeeze":(1,5,1), "test_transfer":(1,0,0)}

data={}
df=None
df2=None

for g in graphs:
    data.setdefault(g,{})
    for layer in range(NLayers[g]):
        data[g].setdefault(layer,{})
        for c in NFreqs:
            data[g][layer].setdefault(c,{})
            for m in Metrics:
                data[g][layer][c].setdefault(m,{})
                for f in range(NFreqs[c]):
                    data[g][layer][c][m].setdefault(f,{})
                    if c=="G":
                        for fbig in range(NFreqs["B"]):
                            data[g][layer][c][m][f].setdefault(fbig,{})
 




data={}
for g in graphs:
    data.setdefault(g,{})
    for c in NFreqs:
        data[g].setdefault(c,{})
        for f in range(NFreqs[c]):
            data[g][c].setdefault(f,{})
            if c=="G":
                for fbig in range(NFreqs["B"]):
                    data[g][c][f].setdefault(fbig,{})
                    for layer in range(NLayers[g]):
                        data[g][c][f][fbig].setdefault(layer,{})
                        for m in Metrics:
                            data[g][c][f][fbig][layer].setdefault(m,{})
                            
            else:
                for layer in range(NLayers[g]):
                    data[g][c][f].setdefault(layer,{})
                    for m in Metrics:
                        data[g][c][f][layer].setdefault(m,{})
                
#print(data)                    
                            


# +
def Parse(timefile,graph,order,frqss):
    with open(timefile) as ff:
        lines=ff.readlines()
    
    
    #order="BBBGBBBB"
    #freqs=[[0],[1],[2],[3,6],[4],[5],[6],[7]]
    freq_indx=0
    freqs=frqss[0]
    n=-1
    t={}
    ins={}
    outs={}
    trans={}
    parts={}
    prof_trans=[]
    transfer_df_time = pd.DataFrame(columns=['order', 'freq', 'transfer_time'])
    
    for l in lines:     
        '''if "Layer Number:" in l:
            n=int(l.split(" ")[2].strip())
            print(f'layer {n}')
            t[str(n)]=float(l.split(" ")[-1].strip())'''
        if "Profiling these DVFS settings finised" in l:
            print(f'Tasks:{t}')
            print(f'Inputs:{ins}')
            print(f'trans:{trans}')
            prof_trans=trans
            transfer_df_time.loc[len(transfer_df_time)]={'order':order, 'freq': tuple(freqs), 'transfer_time':trans[1]}
            print(f'outs:{outs}')
            sss=0
            for iii in t:
                sss+=t[iii]
            for iii in ins:
                sss+=ins[iii]
            print(f'sum tasks+input:{sss}')
            print(parts)
            #data[g][c][f][fbig][layer][m]
            if graph!="test_transfer":
                for layer in t:
                    cmp=order[layer]
                    freq=freqs[layer]
                    #print(data[graph][cmp])
                    if order[layer]=="G":
                        d=data[graph][cmp][freq[0]][freq[1]]
                    else:
                        d=data[graph][cmp][freq[0]]
                    d[layer]["task"]['time']=t[layer]
                    if layer in ins:
                        d[layer]["in"]['time']=ins[layer]
                    if layer in outs:
                        d[layer]["out"]['time']=outs[layer]
                    if layer in trans:
                        d[layer]["trans"]['time']=trans[layer]
            #print(data["squeeze"]["G"])
            ss=0
            for part in parts:
                ss=ss+parts[part]
            print(f"latency is {ss}")
            n=-1
            t={}
            ins={}
            outs={}
            trans={}
            parts={}
            
        pattern = r".* Running Graph with .* LW DVFS"
        if re.match(pattern,l):
            freqs=frqss[freq_indx]
            print(f'Next freq:{freqs}')
            #input()
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
        
      
    #data[g][c][f][fbig][layer][m]
    return prof_trans,transfer_df_time
        
            
    
    
# -


import select
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


def format_freqs(fs=[ [ [7],[6],[4],[3,6],[4],[5],[6],[7] ], [] ]):
        formated_fs=[]
        for f in fs:
            ff = '-'.join(['[' + str(sublist[0]) + ',' + str(sublist[1]) + ']' if len(sublist) > 1 else str(sublist[0]) for sublist in f])
            #print(ff)
            formated_fs.append(ff)
        return formated_fs


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


def Parse_Power(file_name,graph,order,frqss):
    NL=NLayers[graph]
    powers,tts=Read_Power(file_name)
    input_pwrs=[]
    task_pwrs={}
    #for each freq: NL*2(which is input-layer pairs)
    #after each freq we have a excess [0]and[1]interval so:
    nn=((2*NL*n)+2)
    nnn=nn*len(frqss)
    if len(powers)!=nnn:
        print(f"bad power size:{len(powers)}")
        input("what")
        return
    print(f'len powers is {len(powers)}')
    #data[g][c][f][fbig][layer][m]
    for i,freq in enumerate(frqss):
        pwrs=powers[i*nn:(i+1)*nn-2]
        input_pwrs=pwrs[0::2*NL]
        print(f'\n\n\n************\nInput powers with len {len(input_pwrs)}')
        input_pwrs=sum(input_pwrs)/len(input_pwrs)
        #input_pwrs=sum(input_pwrs)
        for layer,j in enumerate(range(1,2*NL,2)):
            task_pwrs[layer]=pwrs[j::2*NL]
            print(f'len layer power {len(task_pwrs[layer])}')
            task_pwrs[layer]=sum(task_pwrs[layer])/len(task_pwrs[layer])
            if order[layer]=="G":
                d=data[graph][order[layer]][freq[layer][0]][freq[layer][1]]
            else:
                d=data[graph][order[layer]][freq[layer][0]]
            if layer==0:
                d[layer]["in"]["Power"]=input_pwrs
                print(f'setting power for {graph}-{order}-{layer}-{freq[layer][0]}-in-power-->{input_pwrs}')
            d[layer]["task"]["Power"]=task_pwrs[layer]
            print(f'setting power for {graph}-{order}-{layer}-{freq[layer][0]}-task-power->{task_pwrs[layer]}')


def profile(_ff=[[[0],[1],[2],[3,6],[4],[5],[6],[7]]],_n=n,order='BBBGBBBB',graph="alex",pwr="pwr.csv",tme="temp.txt", caching=True, kernel_c=96):
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
def profile_task_time(graph="alex"):
    
    
    NL=NLayers[graph]
    
    orders=["G","B","L"]
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
        pwrfile=f'./power_{graph}_'+order+'.csv'
        timefile=f'./time_{graph}_'+order+'.txt'
        profile(frqss,n,order,graph,pwrfile,timefile)
        time.sleep(10)
        Parse(timefile,graph,order,frqss)
        Parse_Power(pwrfile,graph,order,frqss)
        


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
            data=pk.load(f)

    else:
        for graph in graphs[::-1]:
            profile_task_time(graph)
        with open("data.pkl","wb") as f:
            pk.dump(data,f)

    if os.path.isfile("data_df.pkl") and caching:
        with open("data_df.pkl","rb") as f:
            df=pk.load(f)

    else:
        df=to_df(data)
        df.to_csv("data_df.csv",index=False)
    
    return data,df
    #when reading:
    #test=pd.read_csv("data_df.csv",index_col=0)
    #or you can use df.to_csv with index=False argument


def analyze(graph_name=graphs,metric=['task','in','out','trans'],comp=['G','B','L'],
            freq_h=range(10),f=range(10),layers=range(40),index=['Layer'],columns=['Freq'],parameter='Time'):
    '''graph_name = graphs
    metric=['task','in','out','trans']
    comp=['G','B','L']
    freq_h=range(10)
    f=range(10)
    layers=range(40)

    indexes=['Layer','Freq']'''

    # Group the filtered DataFrame by the 'Layer' and 'Freq' columns, and aggregate the 'Time' column using the 'mean()' function
    grouped_df = df[(df['Graph'].isin(graph_name)) & 
                    (df['Metric'].isin(metric)) & 
                    (df['Component'].isin(comp)) & 
                    (df['Freq_Host'].isin(freq_h))].groupby(index+columns)['Time'].sum().reset_index()

    # Create a pivot table to rearrange the data for plotting
    pivot_table = pd.pivot_table(grouped_df, values=parameter, index=index, columns=columns)
    pivot_table.plot(kind='bar', stacked=False, figsize=(30, 6))
    plt.title(f'{metric} {parameter} vs {columns} for {graph_name}')
    plt.xlabel(f'{index}')
    plt.ylabel(f'{metric} {parameter}')
    plt.show()
    return pivot_table
#analyze(graph_name=['alex'],metric=['task'],comp=['G'],freq_h=[0],index=['Layer'],columns=['Freq'])


# +
def analyze2():
    graph_name = 'alex'
    graph_df = df[df['Graph'] == graph_name]

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

#analyze2()

# +

def value(graph,comp,freq,layer,metric,attr):
    if len(freq)==1 or comp!='G':
        return df2.loc[(graph, comp, freq[0], layer, metric, np.NaN), attr]
    if len(freq)==2:
        return df2.loc[(graph, comp, freq[0], layer, metric, freq[1]), attr]
    else:
        return -1
    


# +
def _comp_time(fn=[0,1,2,3,4,5,6,7],dvfs_delay=3.5):
    fn=list(fn)
    fn.insert(0,fn[0])
    print(f'fn is {fn}')
    fc=len(fn)*[None]
    fc[0]=fn[-1]
    fc[1]=fn[1]
    for i in range(2,len(fn)):
        fc[i]=fn[i-1]
    tt=0
    tt_nodvfs=0
    tfn=value('alex','B',[fn[0]],0,'in','Time')
    tfc=value('alex','B',[fc[0]],0,'in','Time')
    t=tfn - (dvfs_delay/tfc)*tfn + dvfs_delay
    print(f'in:{0}, next_freq:{fn[0]} time(next_freq):{tfn} cur_freq:{fc[0]} time(cur_freq):{tfc} time:{t}')
    tt+=t
    tt_nodvfs+=tfn
    for i in range(0,8):
        tfn=value('alex','B',[fn[i+1]],i,'task','Time')
        tfc=value('alex','B',[fc[i+1]],i,'task','Time')
        t=tfn - (dvfs_delay/tfc)*tfn + dvfs_delay
        print(f'layer:{i}, next_freq:{fn[i+1]} time(next_freq):{tfn} cur_freq:{fc[i+1]} time(cur_freq):{tfc} time:{t}')
        tt+=t
        tt_nodvfs+=tfn
    print(f'time with dvfs delay: {tt}')
    print(f'time without dvfs delay: {tt_nodvfs}')
    return tt

def comp_time(g='alex',fn=[[0],[1],[2],[3],[4],[5],[6],[7]],cmps=8*'B',dvfs_delay=3.5):
    fn=list(fn)
    fn.insert(0,fn[0])
    cmps=cmps[0]+cmps
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
        print(f'i:{i}, previous p:{cmps[i-1]}, current p:{cmps[i]}, curent p freqs:{f}, fc[i]:{fc[i]}')
    
    #just first layer(i=1) current freq is equal to its next freq
    #because next freq is applied before input(i=0) and it is already applied for the first layer
    fc[1]=fn[1]
        
    print(f'fc is:{fc}')
    print(f'processors:{cmps}')
    tt=0
    tt_nodvfs=0
    tfn=value(g,cmps[0],fn[0],0,'in','Time')
    tfc=value(g,cmps[0],fc[0],0,'in','Time')
    t=tfn - (dvfs_delay/tfc)*tfn + dvfs_delay
    print(f'in:{0}, next_freq:{fn[0]} time(next_freq):{tfn} cur_freq:{fc[0]} time(cur_freq):{tfc} time:{t}')
    tt+=t
    tt_nodvfs+=tfn
    for i in range(0,8):
        tfn=value(g,cmps[i+1],fn[i+1],i,'task','Time')
        tfc=value(g,cmps[i+1],fc[i+1],i,'task','Time')
        t=tfn - (dvfs_delay/tfc)*tfn + dvfs_delay
        print(f'layer:{i}, next_freq:{fn[i+1]} time(next_freq):{tfn} cur_freq:{fc[i+1]} time(cur_freq):{tfc} time:{t}')
        tt+=t
        tt_nodvfs+=tfn
    print(f'time with dvfs delay: {tt}')
    print(f'time without dvfs delay: {tt_nodvfs}')
    return tt


# -

def load_data():
    global data,df,df2
    data,df=Profiling()
    df2 = df.set_index(['Graph', 'Component', 'Freq', 'Layer', 'Metric', 'Freq_Host'])


# +
def predict(inc=True):
    _fn=[[0],[1],[2],[3],[4],[5],[6],[7]]
    if not inc:
        _fn=_fn[::-1]
        
        
    _dvfs_delay=3.5
    comp_time(fn=_fn, dvfs_delay=_dvfs_delay)
    
def main():
    load_data()
    
    analyze(graph_name=['alex'],metric=['task'],comp=['G'],freq_h=[0],index=['Layer'],columns=['Freq'])
    analyze2()
    
    value('alex','B',[7],7,'out','Time')
    value('alex','G',[0,0],0,'task','Time')
    [value('alex','B',[i],i,'task','Time') for i in range(0,8)]
    [value('alex','B',[i-1],i,'task','Time') for i in range(1,8)]
    
    print(f'Real Run time is: 334.5 ms')
    predict()
    print(f'Real Run time is: 192.7 ms')
    predict(False)


# -

main()


def test():
    load_data()
    _fs=[ [ [0],[1],[2],[3],[4],[5],[6],[7] ],
         [ [7],[6],[5],[4],[3],[2],[1],[0] ] ]
    ord='BBBBBBBB'
    g="alex"
    for fs in _fs:
        profile(_ff=[fs], _n=n, order=ord, graph=g, pwr="pwr.csv", tme="temp.txt",caching=False)
        Parse(timefile="temp.txt", graph=g, order=ord, frqss=[fs])
        _dvfs_delay=3.5
        comp_time(fn=np.reshape(fs,-1), dvfs_delay=_dvfs_delay)


# +
#test()
# -

def Parse_Power_T(file_name,graph,order,frqss):
    NL=NLayers[graph]
    powers,tts=Read_Power(file_name)
    input_pwrs=[]
    task_pwrs={}
    trans_pwrs={}
    transfer_df_pwr = pd.DataFrame(columns=['order', 'freq', 'transfer_power'])
    #transfer_df_pwr.loc[len(transfer_df_pwr)]={'order':order, 'freq': freq[layer], 'transfer_power':trans_pwrs[layer]}
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
            '''if order[layer]=="G":
                d=data[graph][order[layer]][freq[layer][0]][freq[layer][1]]
            else:
                d=data[graph][order[layer]][freq[layer][0]]'''
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


def Parse_Power_total(file_name,graph,order,frqss):
    powers,tts=Read_Power(file_name)
    power_df = pd.DataFrame(columns=['order', 'freq', 'input_power','task_power','power'])
   
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
        total_power=
        input_pwrs=sum(input_pwrs)/len(input_pwrs)
        
        task_pwrs=sum(task_pwrs)/len(task_pwrs)   
        print(f'\n\n\n************\nInput powers: {input_pwrs}')
        print(f'setting power for {graph}-{order}-{freq}-task-power->{task_pwrs}')
        power_df.loc[len(power_df)]={'order':order, 'freq': tuple(freq), 'input_power':input_pwrs, 'task_power':task_pwrs, 'power':input_pwrs+task_pwrs}
    return power_df

# +

    
def Parse_total(timefile,graph,order,frqss):
    with open(timefile) as ff:
        lines=ff.readlines()
    freq_indx=0
    freqs=frqss[0]
    input_time=-1
    parts=[]
    df_time = pd.DataFrame(columns=['order', 'freq', 'input_time', 'task_time', 'total_time'])
    for l in lines:        
        if "Profiling these DVFS settings finised" in l:
            print(f'Input_time:{input_time}')
            s=sum(parts)
            print(f'parts:{parts}, sum:{s}')            
            
            df_time.loc[len(df_time)]={'order':order, 'freq': tuple(freqs), 'input_time':input_time, 'task_time':s-input_time, 'toatal_time':s}
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
        
            
    
    
# -

_fs=[ [ [0,0],[0],[0],[0],[0],[0],[0],[0] ] ]
ord='GBBBBBBB'
g="alex"
pf="pwr_whole.csv"
tf="temp_whole.txt"
profile(_ff=_fs, _n=n, order=ord, graph=g, pwr=pf, tme=tf,caching=True,kernel_c=96*50)
time_df=Parse_total(timefile=tf, graph=g, order=ord, frqss=_fs)
power_df=Parse_Power_total(file_name=pf,graph=g,order=ord,frqss=_fs)
print(time_df)
print(power_df)


# +
# set sleep time between tasks to 0 in ARMCL src/graph/detail/ExecuionHelpers.cpp 
#(check graphmanager.cpp for sure that there is no sleep )
def test_T_max_freq(c,_kernel_c=96):
    load_data()
    
    if c==0:
        _fs=[ [ [0],[5] ] ]
        ord='BL'
    if c==1:
        _fs=[ [ [0],[7] ] ]
        ord='LB'
    if c==2:
        _fs=[ [ [0,7],[7] ] ]
        ord='GB'
    if c==3:
        _fs=[ [ [0],[0,7] ] ]
        ord='LG'
    if c==4:
        _fs=[ [ [7],[0,7] ] ]
        ord='BG'
        
    g="test_transfer"
    trans=[]
    trans_pwr=[]
    for fs in _fs:
        profile(_ff=[fs], _n=n, order=ord, graph=g, pwr="pwr.csv", tme="temp.txt",caching=False,kernel_c=_kernel_c)
        trans,trans_df=Parse(timefile="temp.txt", graph=g, order=ord, frqss=[fs])
        #_dvfs_delay=3.5
        #cal_time(fn=np.reshape(fs,-1), dvfs_delay=_dvfs_delay)
        trans_pwr,trans_pwr_df=Parse_Power_T(file_name="pwr.csv",graph=g,order=ord,frqss=[fs])
    return trans,trans_pwr

def test_T_min_freq(c,_kernel_c=96):
    load_data()
    
    if c==0:
        _fs=[ [ [0],[0] ] ]
        ord='BL'
    if c==1:
        _fs=[ [ [0],[0] ] ]
        ord='LB'
    if c==2:
        _fs=[ [ [0,0],[0] ] ]
        ord='GB'
    if c==3:
        _fs=[ [ [0],[0,0] ] ]
        ord='LG'
    if c==4:
        _fs=[ [ [0],[0,0] ] ]
        ord='BG'
        
    g="test_transfer"
    trans=[]
    trans_pwr=[]
    for fs in _fs:
        profile(_ff=[fs], _n=n, order=ord, graph=g, pwr="pwr.csv", tme="temp.txt",caching=False,kernel_c=_kernel_c)
        trans,trans_df=Parse(timefile="temp.txt", graph=g, order=ord, frqss=[fs])
        #_dvfs_delay=3.5
        #cal_time(fn=np.reshape(fs,-1), dvfs_delay=_dvfs_delay)
        trans_pwr,trans_pwr_df=Parse_Power_T(file_name="pwr.csv",graph=g,order=ord,frqss=[fs])
    return trans,trans_pwr


# +
#deprecated
def _transfer_max_freq():
    if os.path.isfile(maxtransfile):
        transfer_df_max=pd.read_csv(maxtransfile)
        print(f'max freq trans data:\n{transfer_df_max}')
        return transfer_df_max
    kernel_cs=[10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600]
    kernel_cs=[96*i for i in kernel_cs]
    cs=[0,1,2,3,4]
    orders={0:'BL', 1:'LB', 2:'GB', 3:'LG', 4:'BG'}
    global res
    for kernel_c in kernel_cs:
        #evaluated=False
        for _c in cs:
            evaluated=False
            for r in res:
                if r["kernels"]==kernel_c and r["c"]==orders[_c]:
                    evaluated=True
            if not evaluated:
                trans,trans_pwr=test_T_max_freq(_c,kernel_c)
                res.append({"kernels":kernel_c, "c":orders[_c], "transfer_time":trans[1], "transfer_power":trans_pwr[1]})
                print(res[-1])
                time.sleep(3)
    transfer_df_max=pd.DataFrame(res)    
    transfer_df_max.to_csv(maxtransfile,index=False)
    return transfer_df_max

def _transfer_min_freq():
    if os.path.isfile(mintransfile):
        transfer_df_min=pd.read_csv(mintransfile)
        print(f'min freq trans data:\n{transfer_df_min}')
        return transfer_df_min
    kernel_cs=[10,20,30,40,50,60,70,80,90,100,150,200,250,300]
    kernel_cs=[96*i for i in kernel_cs]
    cs=[0,1,2,3,4]
    orders={0:'BL', 1:'LB', 2:'GB', 3:'LG', 4:'BG'}
    global res_min
    for kernel_c in kernel_cs:
        #evaluated=False
        for _c in cs:
            evaluated=False
            for r in res_min:
                if r["kernels"]==kernel_c and r["c"]==orders[_c]:
                    evaluated=True
            if not evaluated:
                trans,trans_pwr=test_T_min_freq(_c,kernel_c)
                res_min.append({"kernels":kernel_c, "c":orders[_c], "transfer_time":trans[1], "transfer_power":trans_pwr[1]})
                print(res_min[-1])
                time.sleep(3)
    transfer_df_min=pd.DataFrame(res_min)    
    transfer_df_min.to_csv(mintransfile,index=False)
    return transfer_df_min


# -

def Explore_Freq(c,_kernel_c=96*100):
    load_data()
    
    if c==0:
        _fs=[[[0],[i]] for i in range(NFreqs['L'])]
        ord='BL'
    if c==1:
        _fs=[[[0],[i]] for i in range(NFreqs['B'])]
        ord='LB'   
    if c==2:
        _fs=[[[0,i],[i]] for i in range(NFreqs['B'])]
        ord='GB'
    if c==3:
        _fs=[[[0],[0,i]] for i in range(NFreqs['B'])]
        ord='LG'
    if c==4:
        _fs=[[[i],[0,i]] for i in range(NFreqs['B'])]
        ord='BG'
        
    g="test_transfer"
    trans=[]
    trans_pwr=[]
    
    profile(_ff=_fs, _n=n, order=ord, graph=g, pwr="pwr.csv", tme="temp.txt",caching=False,kernel_c=_kernel_c)
    trans,transfer_df_time=Parse(timefile="temp.txt", graph=g, order=ord, frqss=_fs)
    #print(trans,transfer_df_time)
    #input()
    #_dvfs_delay=3.5
    #cal_time(fn=np.reshape(fs,-1), dvfs_delay=_dvfs_delay)
    trans_pwr,trans_pwr_df=Parse_Power_T(file_name="pwr.csv",graph=g,order=ord,frqss=_fs)
    
    return trans,trans_pwr,trans_pwr_df,transfer_df_time


explore_freq='freq_tranfer.csv'
def run_explore_freq():
    global transfer_df_freq
    if os.path.isfile(explore_freq):
        transfer_df_freq=pd.read_csv(explore_freq)
        print(f'max freq trans data:\n{transfer_df_freq}')
        #return transfer_df_max
    else:
        transfer_df_freq = pd.DataFrame(columns=['kernels', 'order', 'freq', 'transfer_time', 'transfer_power'])
    kernel_cs=[150]
    kernel_cs=[96*i for i in kernel_cs]
    cs=[0,1,2,3,4]
    #cs=[0]
    orders={0:'BL', 1:'LB', 2:'GB', 3:'LG', 4:'BG'}
    global trans,trans_pwr,trans_pwr_df,transfer_df_time
    for kernel_c in kernel_cs:
        for _c in cs:
            print(f'c:{_c}, order:{orders[_c]}, kernels:{kernel_c}, shape: {transfer_df_freq[(transfer_df_freq["order"]==orders[_c]) & (transfer_df_freq["kernels"]==kernel_c)].shape[0]}')
            if transfer_df_freq[(transfer_df_freq['order']==orders[_c]) & (transfer_df_freq['kernels']==kernel_c)].shape[0]==0:
                print(f"inside, c is:{_c}, order:{orders[_c]}, kernels:{kernel_c}, shape:{transfer_df_freq[(transfer_df_freq['order']==orders[_c]) & (transfer_df_freq['kernels']==kernel_c)].shape[0]}")
                #print(transfer_df_freq)
                #input()
                try:
                    trans,trans_pwr,trans_pwr_df,transfer_df_time=Explore_Freq(_c,kernel_c)
                    #transfer_df_freq.loc[len(transfer_df_freq)] = {"kernels":kernel_c, "c":orders[_c], "transfer_time":transfer_df_time, "transfer_power":trans_pwr_df}
                    trans_pwr_df['freq'] = trans_pwr_df['freq'].apply(lambda x: tuple(tuple(i) for i in x))
                    transfer_df_time['freq'] = transfer_df_time['freq'].apply(lambda x: tuple(tuple(i) for i in x))
                    merged_df = pd.merge(trans_pwr_df, transfer_df_time, on=['order', 'freq'])
                    merged_df['kernels']=kernel_c
                    transfer_df_freq=pd.concat([transfer_df_freq,merged_df], ignore_index=True)
                    print(f'merged is:\n{merged_df}')
                    print(f'accumulated result is:\n{transfer_df_freq}')
                    transfer_df_freq.to_csv(explore_freq,index=False)
                    time.sleep(10)
                except:
                    ab()


run_explore_freq()


# +
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
        
def transfer_max_freq():
    if os.path.isfile(maxtransfile):
        transfer_df_max=pd.read_csv(maxtransfile)
        print(f'max freq trans data:\n{transfer_df_max}')
        #return transfer_df_max
    else:
        transfer_df_max = pd.DataFrame(columns=['kernels', 'order', 'transfer_time', 'transfer_power'])
    kernel_cs=[10,20,30,40,50,60,70,80,90,100,150,200,250,300,400,500]
    kernel_cs=[96*i for i in kernel_cs]
    cs=[0,1,2,3,4]
    orders={0:'BL', 1:'LB', 2:'GB', 3:'LG', 4:'BG'}
    for kernel_c in kernel_cs:
        for _c in cs:
            if transfer_df_max[(transfer_df_max['order']==orders[_c]) & (transfer_df_max['kernels']==kernel_c)].shape[0]==0:
                try:
                    if orders[_c][1]=='G' and kernel_c > 150*96:
                        continue
                    trans,trans_pwr=test_T_max_freq(_c,kernel_c)
                    print(trans,trans_pwr)
                    #input()
                    transfer_df_max.loc[len(transfer_df_max)] = {"kernels":kernel_c, "order":orders[_c], "transfer_time":trans[1], "transfer_power":trans_pwr[1]}
                    transfer_df_max.to_csv(maxtransfile,index=False)
                    time.sleep(10)
                except:
                    ab()               
    
    return transfer_df_max

def transfer_min_freq():
    if os.path.isfile(mintransfile):
        transfer_df_min=pd.read_csv(mintransfile)
        print(f'min freq trans data:\n{transfer_df_min}')
        #return transfer_df_min
    else:
        transfer_df_min = pd.DataFrame(columns=['kernels', 'order', 'transfer_time', 'transfer_power'])
    kernel_cs=[10,20,30,40,50,60,70,80,90,100,150,200,250,300,400,500]
    kernel_cs=[96*i for i in kernel_cs]
    cs=[0,1,2,3,4]
    orders={0:'BL', 1:'LB', 2:'GB', 3:'LG', 4:'BG'}
    for kernel_c in kernel_cs:
        for _c in cs:
            if transfer_df_min[(transfer_df_min['order']==orders[_c]) & (transfer_df_min['kernels']==kernel_c)].shape[0]==0:
                try:
                    if orders[_c][1]=='G' and kernel_c > 150*96:
                        continue
                    trans,trans_pwr=test_T_min_freq(_c,kernel_c)
                    print(trans,trans_pwr)
                    transfer_df_min.loc[len(transfer_df_min)] = {"kernels":kernel_c, "order":orders[_c], "transfer_time":trans[1], "transfer_power":trans_pwr[1]}
                    transfer_df_min.to_csv(mintransfile,index=False)
                    time.sleep(10)
                except:
                    ab()
    
    return transfer_df_min


# -

def Explore_Data_size(order,transfer_df):
    #order='LB'
    t=transfer_df[transfer_df['order']==order].groupby(['kernels']).sum(['transfer_time', 'transfer_power'])
    #p=transfer_df[transfer_df['c']==order].groupby(['kernels'])['transfer_power'].sum()
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


def run_Explore_Data_size():
    orders={0:'BL', 1:'LB', 2:'GB', 3:'LG', 4:'BG'}
    transfer_df_max_freq=transfer_max_freq()
    print(f'transfer df max:\n{transfer_df_max_freq}')
    for i in orders:
        Explore_Data_size(order=orders[i],transfer_df=transfer_df_max_freq)
    transfer_df_min_freq=transfer_min_freq()
    print(f'transfer df min:\n{transfer_df_min_freq}')
    for i in orders:
        Explore_Data_size(order=orders[i],transfer_df=transfer_df_min_freq)


run_Explore_Data_size()


