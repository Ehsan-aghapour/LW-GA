# +
import re
import Arduino_read
import os
import time
import threading
import subprocess

cnn={
    "alex":"graph_alexnet_n_pipe_npu_lw",
    "google":"graph_googlenet_n_pipe_npu_lw",
    "mobile":"graph_mobilenet_n_pipe_npu_lw",
    "res50":"graph_resnet50_n_pipe_npu_lw",
    "squeeze":"graph_squeezenet_n_pipe_npu_lw",
}

graphs=["alex", "google", "mobile", "res50", "squeeze"]
NLayers={"alex":8, "google":11, "mobile":14, "res50":18, "squeeze":10}
NFreqs={"L":6, "B":8, "G":5}
Metrics=["in","task","out","trans"]
n=10
params={"alex":(1,1), "google":(2,2), "mobile":(2,3), "res50":(2,4), "squeeze":(1,5)}

data={}
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
    for l in lines:     
        '''if "Layer Number:" in l:
            n=int(l.split(" ")[2].strip())
            print(f'layer {n}')
            t[str(n)]=float(l.split(" ")[-1].strip())'''
        if "Profiling these DVFS settings finised" in l:
            print(f'Tasks:{t}')
            print(f'Inputs:{ins}')
            print(f'trans:{trans}')
            print(f'outs:{outs}')
            print(parts)
            #data[g][c][f][fbig][layer][m]
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
            
        if "Running Graph with LW DVFS" in l:
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
        
      
    #data[g][c][f][fbig][layer][m]
    
        
            
    
    


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
    print(f'number of intervals{len(tts)}')
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


def profile(ff=["7-6-4-[3,6]-4-5-6-7"],_n=n,order='BBBGBBBB',graph="alex",pwr="pwr.csv",tme="temp.txt"):
    if os.path.isfile(pwr) and os.path.isfile(tme):
        print("loading existed files")
        return 
    
    os.system('adb shell "echo 0 > /sys/class/gpio/gpio157/value"')
    time.sleep(1)
    Power_monitoring = threading.Thread(target=Arduino_read.run,args=(pwr,))
    Power_monitoring.start()
    rr=f"PiTest build/examples/LW/{cnn[graph]} test_graph/ CL {params[graph][0]} {params[graph][1]} 1 {_n} 0 0 100 100 {order} 1 2 4 Alex B B"
    oo=open(tme,'w+')
    Run_Graph(ff,rr,oo,True)
    time.sleep(2)
    Power_monitoring.do_run = False
    oo.close()


# +
def profile_task_time(graph="alex"):
    
    os.system(f"PiPush ../build/examples/LW/{cnn[graph]} test_graph/")
    time.sleep(5)
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
        formatted_frqss=format_freqs(frqss)
        print(f'\n\nformatted freqs:\n {formatted_frqss}')
        order=NL*_order
        pwrfile=f'./power_{graph}_'+order+'.csv'
        timefile=f'./time_{graph}_'+order+'.txt'
        profile(formatted_frqss,n,order,graph,pwrfile,timefile)
        time.sleep(10)
        Parse(timefile,graph,order,frqss)
        Parse_Power(pwrfile,graph,order,frqss)
        


# -

for graph in graphs:
    if graph=="res50":
        profile_task_time(graph)


