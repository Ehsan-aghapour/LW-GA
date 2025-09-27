# +
from ast import Num
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
import traceback
import random
import math
import pprint
from scipy.stats import norm


import queue
import select




Test=5


cnn_dir="/home/ehsan/UvA/ARMCL/Rock-Pi/ComputeLibrary_64_Yolov3/"

cnn={
    "Alex":"graph_alexnet_pipeline",
    "Google":"graph_googlenet_pipeline",
    "MobileV1":"graph_mobilenet_pipeline",
    "ResV1_50":"graph_resnet50_pipeline",
    "SqueezeV1":"graph_squeezenet_pipeline",
    "YOLOv3":"graph_yolov3_pipeline",
    "test_transfer":"graph_test_transfer_pipeline"
}


graphs=["Alex", "Google", "MobileV1", "ResV1_50", "SqueezeV1", "YOLOv3"]
NLayers={"Alex":8, "Google":11, "MobileV1":14, "ResV1_50":18, "SqueezeV1":10, "YOLOv3":75, "test_transfer":2}
NFreqs={"L":6, "B":8, "G":5}
Metrics=["in","task","out","trans"]
Num_frames=10
Num_Warm_up=3



try:
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    Layers_csv = script_dir / 'Layers.csv'
    Transfers_csv = script_dir / 'Transfers.csv'
    Transfer_Freq_csv = script_dir / 'Transfer_Freq.csv'
    Transfer_Data_Size_Min_Freq_csv = script_dir / 'Transfer_Data_Size_Min_Freq.csv'
    Transfer_Data_Size_Max_Freq_csv = script_dir / 'Transfer_Data_Size_Max_Freq.csv'
    Evaluations_csv = script_dir / 'Evaluations.csv'
    Layers_logs = script_dir / 'Layers'
    Transfers_logs = script_dir / 'Transfers'
    Synthetic_Tranfer_logs = script_dir / 'Synthetic_Transfers'
    Layers_Percentage_csv = script_dir / 'Layers_Percentage.csv'
    Layers_With_Percentage_csv = script_dir / 'Layers_With_Percentage.csv'
    #Freq_Transition_Dealy_csv = script_dir/'..'/'DVFS-Delay'/'Perf2'/'Data'/'FreqMeasurements2_5.csv'
    Freq_Transition_Dealy_csv = script_dir/'FreqMeasurements2_5.csv'
    GA_Results_PELSI = script_dir / 'ga_result.csv'
    GA_Results_LW = script_dir / 'ga_result_LW.csv'
except:
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
    #Freq_Transition_Dealy_csv = Path("../DVFS-Delay/Perf2/Data/FreqMeasurements2_5.csv").resolve()
    Freq_Transition_Dealy_csv = Path("FreqMeasurements2_5.csv").resolve()
    GA_Results_PELSI = Path('ga_result.csv').resolve()
    GA_Results_LW = Path('ga_result_LW.csv').resolve()

Layers_df=pd.DataFrame(columns=["Graph", "Component", "Freq", "Freq_Host", "Layer", "Metric", "Time", "Power"])
Layers_df_indexed=pd.DataFrame()
Transfers_df=pd.DataFrame(columns=["Graph", "Layer", "Dest", "Src", "Time"])
Transfer_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'SenderFreq','RecFreq' 'transfer_time', 'transfer_power'])
Transfer_Data_Size_Min_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'transfer_time', 'transfer_power'])
Transfer_Data_Size_Max_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'transfer_time', 'transfer_power'])
Evaluations_df=pd.DataFrame(columns=['graph','order','freq','input_time','task_time','total_time', 'input_power','task_power'])
Freq_Transition_Dealy_df=None


# +
def Load_Data():
    global Layers_df, Transfers_df, Transfer_Freq_df,\
        Transfer_Data_Size_Min_Freq_df,Transfer_Data_Size_Max_Freq_df,Layers_df_indexed,Freq_Transition_Dealy_df
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
        
    ## Loading frequency transmition delay times 
    if Freq_Transition_Dealy_csv.exists():
        Freq_Transition_Dealy_df = pd.read_csv(Freq_Transition_Dealy_csv)
        Freq_Transition_Dealy_df.replace({'Little': 'L', 'Big': 'B', 'GPU': 'G'}, inplace=True)
        
if Test:
    Load_Data()


# -

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



# +
########################## Run a Config on board ############################
#def Run_Graph(ALL_Freqs, run_command, myoutput, blocking=True):

def Run_Graph(ALL_Freqs, prepare_command, output_filename, blocking=True,Power_monitoring=None):
    #This will prepare everything and write run command in run_command.sh
    with open(output_filename+"_log_", 'w') as myoutput_log:
        print(f'prepare command is:{prepare_command}')
        #p = subprocess.Popen(prepare_command.split(),shell=True,text=True)
        p = subprocess.Popen(prepare_command.split(), stdout=myoutput_log, stderr=myoutput_log, stdin=subprocess.PIPE, text=True)
        p.wait()
    print("\n*********************************************\n\
    Preparation Finished\n***********************************************\n\n")
    time.sleep(3)
    run_command=f"{cnn_dir}/run_command.sh"
    Power_monitoring.start()
    with open(output_filename, 'w') as myoutput:
        print(f'run command is:{run_command}')
        p = subprocess.Popen(run_command.split(),stdout=myoutput,stderr=myoutput, stdin=subprocess.PIPE, text=True)
        #time.sleep(50)
        time.sleep(5)
        for Freqs in ALL_Freqs:             
            '''while p.poll() is None:
                # check if the subprocess is ready to accept input
                _, wlist, xlist = select.select([], [p.stdin], [p.stdin], 1)
                if wlist:  # Ready for writing
                    print("Ready to write freq to stdin of the process")
                    break
                if xlist:  # Exceptional condition
                    raise Exception("Exceptional condition on subprocess stdin")'''
            p.stdin.write(f'{Freqs}\n')
            p.stdin.flush()

            '''while p.poll() is None:
                # check if the subprocess is ready to accept input
                rlist, _, _ = select.select([p.stdin], [], [], 1)
                if rlist:
                    break'''

            time.sleep(12)

        p.stdin.write("end\n")
        p.stdin.flush()
        if blocking:
            p.wait()

# # +

# +

        

    
def Run_Graph_1(ALL_Freqs, run_command, output_filename, blocking=True,Power_monitoring=None):
    with open(output_filename, 'w') as myoutput:
        print(run_command)
        p = subprocess.Popen(run_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

        in_text_power="setup finished"
        in_text="Please Enter the desired Frequency setttings:"
        freq_index = 0
        while freq_index < len(ALL_Freqs) or p.poll() is None:
            # Use select to wait for output
            readable, _, _ = select.select([p.stdout, p.stderr], [], [], 1)
            for stream in readable:
                line = stream.readline()
                if line:
                    print('Output:', line, end='')
                    myoutput.write(line)
                    myoutput.flush()
                    if in_text_power in line:
                        Power_monitoring.start()
                        print('start pm in Run_graph function\n')
                        time.sleep(4)
                    if in_text in line and freq_index < len(ALL_Freqs):
                        p.stdin.write(f'{ALL_Freqs[freq_index]}\n')
                        p.stdin.flush()
                        freq_index += 1

            # Adjust as needed for your use case
            #time.sleep(1)

        p.stdin.write("end\n")
        p.stdin.flush()
        if blocking:
            p.wait()
    
    
def enqueue_output(out, queue, file):
    for line in iter(out.readline, ''):
        file.write(line)
        file.flush()
        if(queue):
            queue.put(line)
    print("\n\n\n\n\n\n\n\n\n\n\n\nTAmam\n\n\n\n\n\n")
    out.close()
        
def Run_Graph_2(ALL_Freqs, run_command, output_filename, blocking=True,Power_monitoring=None):
    with open(output_filename, 'w') as myoutput:
        with open(output_filename+'_log', 'w') as myoutput_log:
            # Start the subprocess with stdout and stderr redirected to pipes
            p = subprocess.Popen(run_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

            # Queues for stdout and stderr
            stdout_queue = queue.Queue()
            #stderr_queue = queue.Queue()

            # Threads for stdout and stderr
            stdout_thread = threading.Thread(target=enqueue_output, args=(p.stdout, stdout_queue, myoutput))
            #stderr_thread = threading.Thread(target=enqueue_output, args=(p.stderr, None, myoutput_log))
            stdout_thread.daemon = True
            #stderr_thread.daemon = True
            stdout_thread.start()
            #stderr_thread.start()

            freq_index = 0
            in_text_power="setup finished"
            in_text="Please Enter the desired Frequency setttings:"
            while freq_index < len(ALL_Freqs):
                # Check stdout and stderr
                for q in [stdout_queue]:#, stderr_queue]:
                    try:
                        line = q.get_nowait()
                    except queue.Empty:
                        continue  # No output yet, keep checking
                    else:
                        if in_text_power in line:
                            Power_monitoring.start()
                            print("Starting power monitoring in run_graph func\n")
                            time.sleep(2)
                        print('Output:', line)
                        
                        if in_text in line:
                            if freq_index < len(ALL_Freqs):
                                p.stdin.write(f'{ALL_Freqs[freq_index]}\n')
                                p.stdin.flush()
                                freq_index += 1
                #time.sleep(0.0001)  # Adjust sleep time as needed

            p.stdin.write("end\n")
            p.stdin.flush()
            if blocking:
                p.wait()

            stdout_thread.join()
            #stderr_thread.join()



# -

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


# +
## Convert freqs list to string
def format_freqs(fs=[ [ [7],[6],[4],[3,6],[4],[5],[6],[7] ], [] ]):
        formated_fs=[]
        for f in fs:
            if f[0]=="min":
                formated_fs.append(f)
                continue
            if type(f)==str:
                f=[[int(j) for j in re.findall(r"\b\d+\b", l)] for l in f.split('),')]
            ff = '-'.join(['[' + str(sublist[0]) + ',' + str(sublist[1]) + ']' if len(sublist) > 1 else str(sublist[0]) for sublist in f])
            #print(ff)
            formated_fs.append(ff)
        return formated_fs

def format_to_list(fs):
    formated_fs=[]
    for f in fs:
        t=[[int(j) for j in re.findall(r"\b\d+\b", l)] for l in f.split('),')]
        formated_fs.append(t)
    return formated_fs


# +
### This is common function to run a case
## Remember to modify ARMcL code based on your desire
def Profile(_ff=[[[0],[1],[2],[3,6],[4],[5],[6],[7]]],_Num_frames=Num_frames,order='BBBGBBBB',graph="alex",pwr="pwr.csv",tme="temp.txt", caching=True, kernel_c=96, _power_profie_mode='whole'):
    #caching=False
    if os.path.isfile(pwr) and os.path.isfile(tme) and caching:
        print("loading existed files")
        return 
    
    ab()
    ff=format_freqs(_ff)
    print(f'\n\nformatted freqs:\n {ff}')
    os.system(f"adb push {cnn_dir}/build/examples/Pipeline/{cnn[graph]} /data/local/ARM-CO-UP/test_graph/")
    os.system('adb shell "echo 0 > /sys/class/gpio/gpio157/value"')
    time.sleep(5)
    Power_monitoring = threading.Thread(target=Arduino_read.run,args=(pwr,))
    #Power_monitoring.start()
    rr=f"{cnn_dir}/Run_CO-UP model={graph} --n={_Num_frames} --order={order}  push=0 compile=0 --power_profile_mode={_power_profie_mode} --kernel_c={kernel_c}"
    print(f'run command is {rr}')
    #oo=open(tme,'w+')
    
    # if you want to set freqs with cin:
    Run_Graph(ff,rr,tme,True,Power_monitoring)
    
    # if you want to set with run command
    #run_command=rr + f'--freqs=ff[0]'
    #p = subprocess.Popen(run_command.split(),stdout=oo,stderr=oo, stdin=subprocess.PIPE, text=True)
    #time.sleep(5)
    #p.wait()
    
    Power_monitoring.do_run = False
    #oo.close()
    
#Profile(caching=False,_Num_frames=10)
# -


# Profile(caching=False,_Num_frames=10)


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
    starting_layer={}
    ending_layer={}
    for l in lines:         
        pattern = r'Adding Graph(\d+)\s+target \d+ PE: [A-Z]\s+Host PE: [A-Z]\s+Layers: (\d+)-(\d+)'
        matches = re.findall(pattern, l)
        # Iterate through the matches and extract the start and end layers
        if matches:
            graph_number, start_layer, end_layer = matches[0]
            starting_layer[graph_number]=start_layer
            ending_layer[graph_number]=end_layer
        
        pattern = r'Graph(\d+)\s+Input:\s+([\d.]+)\s+Task:\s+([\d.]+)\s+send:\s+([\d.]+)\s+Out:\s+([\d.]+)\s+Process:\s+([\d.]+)'
        matches = re.findall(pattern, l)

        if matches:
            graph_number, input_value, task_value, send_value, out_value, process_value = matches[0]    
            #print(graph_number,input_value,starting_layer[graph_number])
            #input()
            k=int(ending_layer[graph_number])
            value=float(send_value)
            if k<(len(order)-1):
                trans_df.loc[len(trans_df)]={"Graph":graph, "Layer":k, "Dest":order[k+1], "Src":order[k],"Time":value}
            
            
            
            
    print(trans)   
    return trans_df
 
#### Run a graph for measuring trasfer times of real layers
#### As transfer time of real layers is small, it does not profile power
def Profile_Transfer_Layers(ff=["7-6-4-[3,6]-4-5-6-7"],_Num_frames=Num_frames,order='BBBGBBBB',graph="alex",tme="temp.txt",caching=False):
    if os.path.isfile(tme) and caching:
        print("loading existed files")
        return 
    
    #rr=f"PiTest build/examples/LW/{cnn[graph]} test_graph/ CL {params[graph][0]} {params[graph][1]} 1 {_Num_frames} 0 0 100 100 {order} 1 2 4 Alex B B"
    
    rr=f"{cnn_dir}/Run_CO-UP model={graph} --n={_Num_frames} --order={order}  push=0 compile=0 --power_profile_mode=transfers"
    print(f'run command is {rr}')
    #oo=open(tme,'w+')
    Run_Graph(ff,rr,tme,True)
    #time.sleep(2)
    #oo.close()


# -

### Run different order configuration to profile transfer time of real layers with min freqs
### It calls profile_Transfer_Layers and Parse_Transfer_Layers functions
def Profile_Transfer_Time(graph="alex"):
    ab()
    os.system(f"adb push {cnn_dir}/build/examples/Pipeline/{cnn[graph]} /data/local/ARM-CO-UP/test_graph/")
    os.system('adb shell "echo 0 > /sys/class/gpio/gpio157/value"')
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
        Profile_Transfer_Layers(["min"],Num_frames,_order,graph,timefile,caching=True)
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
if Test==4:
    Run_Profile_Transfer_Time()


### Parse the log results to extract timing paramters for layer times
def Parse(timefile,graph,order,frqss):
    time_df=pd.DataFrame(columns=["Graph", "Component", "Freq", "Freq_Host", "Layer", "Metric", "Time"])
    with open(timefile,errors='ignore') as ff:
        lines=ff.readlines()
    freq_indx=0
    freqs=frqss[0]
    t={}
    ins={}
    outs={}
    trans={}
    parts={} 
    starting_layer={}
    ending_layer={}
    for l in lines:    
        pattern = r'Adding Graph(\d+)\s+target \d+ PE: [A-Z]\s+Host PE: [A-Z]\s+Layers: (\d+)-(\d+)'
        matches = re.findall(pattern, l)
        # Iterate through the matches and extract the start and end layers
        for match in matches:
            graph_number, start_layer, end_layer = match
            starting_layer[graph_number]=start_layer
            ending_layer[graph_number]=end_layer
            
        pattern = r".* Running Graph with .* LW DVFS"
        if re.match(pattern,l):
            freqs=frqss[freq_indx]
            print(f'Next freq:{freqs}')
            freq_indx=freq_indx+1
            
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
                
                #input()
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
            
        
        match = re.search(r"Layer Number: (\d+) \t time: (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            t[k]=value 
            
            
        pattern = r'Graph(\d+)\s+Input:\s+([\d.]+)\s+Task:\s+([\d.]+)\s+send:\s+([\d.]+)\s+Out:\s+([\d.]+)\s+Process:\s+([\d.]+)'
        matches = re.findall(pattern, l)

        if matches:
            graph_number, input_value, task_value, send_value, out_value, process_value = matches[0]    
            #print(graph_number,input_value,starting_layer[graph_number])
            #input()
            in_layer=starting_layer[graph_number]
            out_layer=ending_layer[graph_number]
            
            ins[int(in_layer)]=float(input_value)
            
            outs[int(out_layer)]=float(out_value)
            
            
    
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
    transfer_df_time = pd.DataFrame(columns=['order', 'freq', 'transfer_time', 'RecFreq','SenderFreq'])
    
    for l in lines:     
        if "Profiling these DVFS settings finised" in l:
            print(f'Tasks:{t}')
            print(f'Inputs:{ins}')
            print(f'trans:{trans}')
            print(f'outs:{outs}')
            prof_trans=trans
            transfer_df_time.loc[len(transfer_df_time)]={'order':order, 'freq': tuple(freqs), 'transfer_time':trans[1], 'RecFreq':tuple(freqs[1]),'SenderFreq':tuple(freqs[0])}
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
    Num_runs=Num_frames+Num_Warm_up
    nn=((2*NL*Num_frames)+2)
    nn=((2*NL*Num_runs)+2)
    nnn=nn*len(frqss)
    if len(powers)!=nnn:
        print(f"bad power size:{len(powers)}")
        print(f'Expected size is:NFreqx((2xNLxn)+2) which is {len(frqss)}x((2x{NL}x{Num_frames})+2)=nnn')
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
            if order[layer]=="N":
                Host_freq=freq[layer][0]
                
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
    transfer_df_pwr = pd.DataFrame(columns=['order', 'freq', 'transfer_power','RecFreq','SenderFreq'])
    #for each freq: NL*2(which is input-layer pairs)
    #after each freq we have a excess [0]and[1]interval so:
    nn=((2*NL*Num_frames)+2)
    nnn=nn*len(frqss)
    if len(powers)!=nnn:
        print(f"bad power size: {len(powers)}")
        print(f'Expected size is:NFreqx((2xNLxn)+2) which is {len(frqss)}x((2x{NL}x{Num_frames})+2)=nnn')
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
                transfer_df_pwr.loc[len(transfer_df_pwr)]={'order':order, 'freq': tuple(freq), 'transfer_power':trans_pwrs[layer],'RecFreq':tuple(freq[1]),'SenderFreq':tuple(freq[0])}
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
        Profile(frqss,Num_frames,order,graph,pwrfile,timefile,caching=True,_power_profie_mode="layers")
        #time.sleep(10)
        time_df=Parse(timefile,graph,order,frqss)
        power_df=Parse_Power(pwrfile,graph,order,frqss)
        #time_df['Freq'] = time_df['Freq'].apply(lambda x: tuple(x))
        #power_df['Freq'] = power_df['Freq'].apply(lambda x: tuple(x))
        #print(time_df)
        #input()
        merged_df = pd.merge(power_df, time_df, on=['Graph', 'Component', 'Freq','Freq_Host','Layer','Metric'],how='outer')
        Layers_df=pd.concat([Layers_df,merged_df], ignore_index=True)
        Layers_df.to_csv(Layers_csv,index=False)


### This is for parse power for layers of real graphs 
### So the in ARMCL you need to add a sleep between tasks to be catched with power setup
### As here transfer power is not capture adding this sleep does not affect these data
### but it is neccessary as transfering is less than 1.4 ms (sample interval of power setup)
def Parse_Power_NPU_Yolo(file_name,graph,order,frqss):
    pwr_df=pd.DataFrame(columns=["Graph", "Component", "Freq", "Freq_Host", "Layer", "Metric", "Power"])
    NL=NLayers[graph]
    powers,tts=Read_Power(file_name)
    input_pwrs=[]
    task_pwrs={}
    #for each freq: NL*2(which is input-layer pairs)
    #after each freq we have an excess [0] and [1] interval, so:
    Num_runs=Num_frames+Num_Warm_up
    nn=((2*NL*Num_frames)+2)
    nn=((2*NL*Num_runs)+2)
    nnn=nn*len(frqss)
    
    if graph=="YOLOv3":
        NL2=NL-4
        nn2=((2*NL2*Num_runs)+2)
        nnn2=nn2*len(frqss)
        
    if len(powers)!=nnn:
        print(f"bad power size:{len(powers)}")
        print(f'Expected size is:NFreqx((2xNLxn)+2) which is {len(frqss)}x((2x{NL}x{Num_runs})+2)={nnn}')
        if graph=="YOLOv3":
            if len(powers) != nnn2:
                print(f"Even it is not equal to {nnn2} for yolov3")
                input("what")
                return pwr_df
            else:
                print("Yolov3 power size is correct")
                NL=NL2
                nn=nn2
                nnn=nnn2
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
            if order[layer]=="N":
                Host_freq=freq[layer][0]
                
            if layer==0:
                pwr_df.loc[len(pwr_df)]={"Graph":graph, "Component":order[layer],"Freq":freq[layer][0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"in","Power":input_pwrs}
                print(f'setting power for {graph}-{order}-{layer}-{freq[layer][0]}-in-power-->{input_pwrs}')
            
            task_pwrs[layer]=pwrs[j::2*NL]
            print(f'len layer power {len(task_pwrs[layer])}')
            task_pwrs[layer]=sum(task_pwrs[layer])/len(task_pwrs[layer])
            pwr_df.loc[len(pwr_df)]={"Graph":graph, "Component":order[layer],"Freq":freq[layer][0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"task","Power":task_pwrs[layer]}
            print(f'setting power for {graph}-{order}-{layer}-{freq[layer][0]}-task-power->{task_pwrs[layer]}')
    if graph=="YOLOv3":
        #print(pwr_df)
        pwr_df.loc[(pwr_df['Layer'] >= 57), 'Layer'] += 2
        pwr_df.loc[(pwr_df['Layer'] >= 65), 'Layer'] += 2
        #pwr_df.loc[(pwr_df['Layer'] >= 65) , 'Layer'] += 2
        #pwr_df.loc[(pwr_df['Layer'] >= 65) & (pwr_df['Layer'] <= 71), 'Layer'] += 4
        
        #print(pwr_df)
        #input()
    
    return pwr_df


### Parse the log results to extract timing paramters for layer times
def Parse_NPU(timefile,graph,order,frqss):
    time_df=pd.DataFrame(columns=["Graph", "Component", "Freq", "Freq_Host", "Layer", "Metric", "Time"])
    with open(timefile,errors='ignore') as ff:
        lines=ff.readlines()
    freq_indx=0
    freqs=frqss[0]
     
    T_Layer_Sub_Graph={}
    T_Layer={}
    NPU_input_time={}
    NPU_run_time={}
    NPU_run_time_profile={}
    NPU_output_time={}
    Tasks={}
    Sends={}
    Parts={}
    Ins={}
    Outs={}
        
    starting_layer={}
    ending_layer={}
    for l in lines:    
        pattern = r'Adding Graph(\d+)\s+target \d+ PE: [A-Z]\s+Host PE: [A-Z]\s+Layers: (\d+)-(\d+)'
        matches = re.findall(pattern, l)
        # Iterate through the matches and extract the start and end layers
        for match in matches:
            graph_number, start_layer, end_layer = match
            starting_layer[graph_number]=start_layer
            ending_layer[graph_number]=end_layer
            
        pattern = r".* Running Graph with .* LW DVFS"
        if re.match(pattern,l):
            freqs=frqss[freq_indx]
            print(f'Next freq:{freqs}')
            freq_indx=freq_indx+1
            
        if "Profiling these DVFS settings finised" in l:
            print(f'Tasks:{T_Layer}')
            for layer in NPU_run_time:
                cmp=order[layer]
                freq=freqs[layer]
                Host_freq=-1
                if order[layer]=="G":
                    Host_freq=freq[1]
                if order[layer]=="N":
                    Host_freq=freq[0]
                
                #input()
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"in","Time":Ins[layer]}
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"task","Time":T_Layer[layer]}
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"out","Time":Outs[layer]}
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"NPU_in","Time":NPU_input_time[layer]}
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"NPU_run","Time":NPU_run_time[layer]}
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"NPU_run_profile","Time":NPU_run_time_profile[layer]}
                time_df.loc[len(time_df)]={"Graph":graph, "Component":cmp,"Freq":freq[0],"Freq_Host":Host_freq, 
                                           "Layer":layer,"Metric":"NPU_out","Time":NPU_output_time[layer]}
                    
                
            T_Layer_Sub_Graph={}
            T_Layer={}
            NPU_input_time={}
            NPU_run_time={}
            NPU_run_time_profile={}
            NPU_output_time={}
            Tasks={}
            Sends={}
            Parts={}
            Ins={}
            Outs={}
            
            
            
        
        match = re.search(r"Layer Number: (\d+) \t time: (\d+\.\d+)", l)
        if match:
            k = int(match.group(1))
            value = float(match.group(2))
            T_Layer_Sub_Graph[k]=value 
            
            
        pattern = r'Graph(\d+)\s+Input:\s+([\d.]+)\s+Task:\s+([\d.]+)\s+send:\s+([\d.]+)\s+Out:\s+([\d.]+)\s+Process:\s+([\d.]+)'
        matches = re.findall(pattern, l)

        if matches:
            graph_number, input_value, task_value, send_value, out_value, process_value = matches[0]    
            #print(graph_number,input_value,starting_layer[graph_number])
            #input()
            in_layer=int(starting_layer[graph_number])
            out_layer=int(ending_layer[graph_number])
            
            Ins[int(in_layer)]=float(input_value)
            Tasks[int(graph_number)]=float(task_value)
            Sends[int(out_layer)]=float(send_value)
            Outs[int(out_layer)]=float(out_value)
            Parts[int(graph_number)]=float(process_value)
            print(f'Graph: {graph_number} \tstart_layer:{in_layer} \t end_layer:{out_layer} \
            in:{input_value} \ttask:{task_value} \tout:{out_value} \tprocess:{process_value}')
            for i,k in enumerate(range(in_layer,out_layer+1)):
                T_Layer[k]=T_Layer_Sub_Graph[i]
            print(f'T layers are:{T_Layer}')
            
        pattern = fr'NPU_{graph}_(\d+)_(\d+)\s+AVG_input_time:(\d+\.\d+)\s+AVG_run_time:(\d+\.\d+)\s+AVG_prof_run_time:(\d+\.\d+)\s+AVG_output_time:(\d+\.\d+)'
        matches = re.findall(pattern, l)
        if matches:
            values = matches[0]
            # Convert strings to appropriate numeric types (int or float)
            numeric_values = [float(value) if '.' in value else int(value) for value in values]
            first_layer=numeric_values[0]
            last_layer=numeric_values[1]
            
            
            NPU_input_time[first_layer]=numeric_values[2]
            NPU_run_time[first_layer]=numeric_values[3]
            NPU_run_time_profile[first_layer]=numeric_values[4]
            NPU_output_time[last_layer]=numeric_values[5]
            print(f'start_layer:{first_layer} \tend_layer:{last_layer} \tin:{NPU_input_time[first_layer]} \trun:{NPU_run_time[first_layer]}\
            \trun_profile:{NPU_run_time_profile[first_layer]} \tout:{NPU_output_time[last_layer]}')
            
    
    return time_df  


def Profile_Task_Time_NPU(graph):
    global Layers_df
    NL=NLayers[graph]
    C=["N","B"]
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
    
    for order in orders:
        NL=len(order)
        print(f'graph:{graph} order:{order} ')
        frqss=[]
        NF_Host=NFreqs["B"]
        # for test just run with max freq
        Ttt=0
        if Ttt==1:
            layer_f=[7]
            layers_f=NL*[layer_f]
            frqss.append(layers_f)
        else:
            for f in range(NF_Host):
                layer_f=[f]
                layers_f=NL*[layer_f]
                frqss.append(layers_f)
        
        
        
        print(frqss)
        
        Layers_logs.mkdir(parents=True, exist_ok=True)
        pwrfile=f'{Layers_logs}/power_{graph}_'+order+'.csv'
        timefile=f'{Layers_logs}/time_{graph}_'+order+'.txt'
        Profile(frqss,Num_frames,order,graph,pwrfile,timefile,caching=True,_power_profie_mode="layers")
        #time.sleep(10)
        time_df=Parse_NPU(timefile,graph,order,frqss)
        if graph=="YOLOv3":
            power_df=Parse_Power_NPU_Yolo(pwrfile,graph,order,frqss)
        else:
            power_df=Parse_Power(pwrfile,graph,order,frqss)
        #time_df['Freq'] = time_df['Freq'].apply(lambda x: tuple(x))
        #power_df['Freq'] = power_df['Freq'].apply(lambda x: tuple(x))
        #print(time_df)
        #input()
        time_df = time_df[time_df['Component'] == 'N']
        
        power_df = power_df[power_df['Component'] == 'N']
        merged_df = pd.merge(power_df, time_df, on=['Graph', 'Component', 'Freq','Freq_Host','Layer','Metric'],how='outer')
        
        Layers_df=pd.concat([Layers_df,merged_df], ignore_index=True)
        Layers_df.to_csv(Layers_csv,index=False)


# +
#when reading:
#test=pd.read_csv("data_df.csv",index_col=0)
#or you can use df.to_csv with index=False argument

def Profiling_Layers():
    for graph in graphs[::1]:
        if Layers_df[Layers_df["Graph"]==graph].shape[0]==0:
            Profile_Task_Time(graph)   
            
if Test==5:
    Profiling_Layers()
    
def Profile_Layers_NPU():
    for graph in graphs[::1]:
        if graph=="MobileV1" or graph=="YOLOv3":
        #if graph=="YOLOv3":
            #if Layers_df[(Layers_df["Graph"]==graph) & (Layers_df["Component"]=="N")].shape[0]==0 :
            Profile_Task_Time_NPU(graph)
                #input("berim?")
            
if Test==5:
    Profile_Layers_NPU()


# +
#exploring dvfs for layers for example
def Analyze(graph_name=graphs,metric=['task','in','out','trans'],comp=['G','B','L'],
            freq_h=[-1],f=range(10),layers=range(40),index=['Layer'],columns=['Freq'],parameter='Time'):

    # Group the filtered DataFrame by the 'Layer' and 'Freq' columns, and aggregate the 'Time' column using the 'mean()' function
    grouped_df = Layers_df[(Layers_df['Graph'].isin(graph_name)) & 
                    (Layers_df['Metric'].isin(metric)) & 
                    (Layers_df['Component'].isin(comp)) & 
                    (Layers_df['Freq_Host'].isin(freq_h))&
                    (Layers_df['Layer'].isin(layers)) ].groupby(index+columns)['Time','Power'].sum().reset_index()
    grouped_df['Energy']=grouped_df['Power']*grouped_df['Time']/1000.0
    grouped_df['Power-Efficiency']=1000.0/(grouped_df['Energy'])
    # Create a pivot table to rearrange the data for plotting
    pivot_table = pd.pivot_table(grouped_df, values=parameter, index=index, columns=columns)
    try:
        display(pivot_table)
    except:
        pprint.pprint(pivot_table)
    #pivot_table.plot(kind='bar', stacked=False, figsize=(30, 14))
    pivot_table.plot(kind='bar', stacked=False, figsize=(10, 5.625))
    
    #pivot_table.plot(kind='bar', stacked=False, figsize=(12, 12))
    
    #This part is wrote for getting graph for presentation if you see error later
    # consider removing it and uncommenting next lines that are similar
    font_size = 18
    plt.title(f'{parameter} vs {columns[0]} for {graph_name[0]} {index[0]}s', fontsize=font_size)
    plt.xlabel(f'{index[0]}', fontsize=font_size-2)
    plt.ylabel(f'{parameter}', fontsize=font_size-2)
    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    plt.legend(fontsize=font_size-2)
    '''
    plt.title(f'{metric} {parameter} vs {columns} for {graph_name} ')
    plt.xlabel(f'{index}')
    plt.ylabel(f'{metric} {parameter}')
    plt.show()'''
    return pivot_table

if Test==1:
    g='alex'
    Analyze(graph_name=[g],metric=['task'],comp=['L'],index=['Layer'],columns=['Freq'],parameter='Power-Efficiency')
    Analyze(graph_name=[g],metric=['task'],comp=['B'],index=['Layer'],columns=['Freq'],parameter='Power-Efficiency')
    Analyze(graph_name=[g],metric=['task'],comp=['G'],freq_h=[0],index=['Layer'],columns=['Freq'],parameter='Power-Efficiency')
# -

if Test==5:
    graph_name=['YOLOv3']
    metric=['task']
    comp=['B','N']
    freq_h=[-1,7]
    freq=[7]
    layers=range(80)
    index=['Layer']
    columns=[]
    parameter='Time'
    grouped_df = Layers_df[(Layers_df['Graph'].isin(graph_name)) & 
                        (Layers_df['Metric'].isin(metric)) & 
                        (Layers_df['Component'].isin(comp)) & 
                        (Layers_df['Freq_Host'].isin(freq_h))&
                        (Layers_df['Freq'].isin(freq)) &
                        (Layers_df['Layer'].isin(layers)) ]
    #.groupby(index+columns)['Time','Power'].sum().reset_index()
    grouped_df.to_csv('t.csv')


def Analyze2(graph_name = 'alex'):
    graph_df = Layers_df[Layers_df['Graph'] == graph_name]
    # Group the filtered DataFrame by the 'Layer' and 'Freq' columns, and aggregate the 'Time' column using the 'mean()' function
    #grouped_df = graph_df[graph_df['Metric'] == 'task'].groupby(['Graph', 'Component', 'Freq', 'Layer'])['Time'].sum()
    grouped_df = graph_df[graph_df['Metric'] == 'task'].groupby(['Graph', 'Component', 'Layer', 'Freq'])['Time'].sum().reset_index()
    #print(grouped_df)
    # Create a pivot table to rearrange the data for plotting
    pivot_table = pd.pivot_table(grouped_df,index=['Graph', 'Component', 'Layer'], columns='Freq', values='Time')
    # Generate a line plot to visualize the effect of frequency on task timing for different layers
    pivot_table.plot(kind='bar', stacked=False, figsize=(10, 6))
    plt.title(f'Task Timing vs Frequency for {graph_name}')
    plt.xlabel('Layer')
    plt.ylabel('Task Timing (ms)')
    plt.show()
    return pivot_table
if Test==2:
    Analyze2()


def Value(graph,comp,freq,layer,metric,attr):
    global Layers_df_indexed
    if Layers_df_indexed.shape[0]==0:
        Layers_df_indexed = Layers_df.set_index(['Graph', 'Component', 'Freq', 'Layer', 'Metric', 'Freq_Host'])
    if len(freq)==1 or comp!='G':
        return Layers_df_indexed.loc[(graph, comp, freq[0], layer, metric, -1), attr]
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
        print(f'Energy with dvfs delay: {ee/1000.0}')
        print(f'Energy without dvfs delay: {ee_nodvfs/1000.0}')
    return tt,ee/1000.0


def Comp_Cost_variable_dvfs_delay(g='alex',fn=[[0],[1],[2],[3],[4],[5],[6],[7]],cmps=8*'B', debug=False):
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
    _dvfs_delay=Freq_Transition_Dealy_df[(Freq_Transition_Dealy_df["PE"]==cmps[0]) &\
                                         (Freq_Transition_Dealy_df['Freq']==fc[0][0]) &\
                                         (Freq_Transition_Dealy_df['NextFreq']==fn[0][0])]['AVG'].mean()/1000000.0
    if debug:
        print(f'dvfs delay for inpu: {_dvfs_delay}')
        
    
    t_output=Value(g,cmps[-1],fn[-1],len(fn)-1,'out','Time')
    
    #in arm-co-up apply freq is at the end of the task (not after output); so the part of the input
    # that is run with current freq is _dvfs_delay-t_output so we change _dvfs_delay(of this layer) for simplicity
    if _dvfs_delay > t_output:
        _dvfs_delay=_dvfs_delay-t_output
        
        
    if tfc > _dvfs_delay:
        t=tfn - (_dvfs_delay/tfc)*tfn + _dvfs_delay 
              
    if debug:
        print(f'in:{0}, next_freq:{fn[0]} time(next_freq):{tfn} cur_freq:{fc[0]} time(cur_freq):{tfc} time:{t}')      
    tt+=t
    tt_nodvfs+=tfn
    
    # we consider the input power for output power (because output power is not measureble with the current setup)
    p_output=Value(g,cmps[-1],fn[-1],0,'in','Power')
    e_output=t_output*p_output
    if debug:
        print(f't_output: {t_output}   p_output:{p_output}   e_output:{e_output}')
        
    ee+=e_output
    ee_nodvfs+=e_output
    
    #comp power
    pfn=Value(g,cmps[0],fn[0],0,'in','Power')
    pfc=Value(g,cmps[0],fc[0],0,'in','Power') 
    e=t*pfc
    if t > _dvfs_delay:
        e=_dvfs_delay*pfc + (t-_dvfs_delay)*pfn
    e_nodvfs= tfn*pfn
    ee+=e
    ee_nodvfs+=e_nodvfs
    if debug:
        print(f'in:{0}, next_freq:{fn[0]} power(next_freq):{pfn} cur_freq:{fc[0]} power(cur_freq):{pfc} energy:{e}')
        
    for i in range(0,len(fn)-1):
        tfn=Value(g,cmps[i+1],fn[i+1],i,'task','Time')
        tfc=Value(g,cmps[i+1],fc[i+1],i,'task','Time')
        t=tfc
        _dvfs_delay=Freq_Transition_Dealy_df[(Freq_Transition_Dealy_df["PE"]==cmps[i+1]) &\
                                             (Freq_Transition_Dealy_df['Freq']==fc[i+1][0]) &\
                                             (Freq_Transition_Dealy_df['NextFreq']==fn[i+1][0])]['AVG'].mean()/1000000.0
        if debug:
            print(f'dvfs delay for layer{i}: {_dvfs_delay}')
        if tfc > _dvfs_delay:
            t=tfn - (_dvfs_delay/tfc)*tfn + _dvfs_delay
        if debug:
            print(f'layer:{i}, next_freq:{fn[i+1]} time(next_freq):{tfn} cur_freq:{fc[i+1]} time(cur_freq):{tfc} time:{t}')
        tt+=t
        tt_nodvfs+=tfn
        
        pfn=Value(g,cmps[i+1],fn[i+1],i,'task','Power')
        pfc=Value(g,cmps[i+1],fc[i+1],i,'task','Power') 
        e=t*pfc
        if t > _dvfs_delay:
            e=_dvfs_delay*pfc + (t-_dvfs_delay)*pfn
        e_nodvfs= tfn*pfn
        if debug:
            print(f'layer:{i}, next_freq:{fn[i+1]} power(next_freq):{pfn} cur_freq:{fc[i+1]} power(cur_freq):{pfc} energy:{e}')
        ee+=e
        ee_nodvfs+=e_nodvfs
        
    if debug:
        print(f'time with dvfs delay: {tt}')
        print(f'time without dvfs delay: {tt_nodvfs}')
        print(f'Energy with dvfs delay: {ee/1000.0}')
        print(f'Energy without dvfs delay: {ee_nodvfs/1000.0}')
    return tt,ee/1000.0


def Transfer_Info(p1='B',p2='G',f1=[4],f2=[3,4],_debug=False):
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
    row=Transfer_Freq_df[ (Transfer_Freq_df['freq']==str(freqs)) & (Transfer_Freq_df['order']==order)]
    if _debug:
        print(freqs)
        print(row)
    power=row['transfer_power'].iloc[0]
    coef_t=row['time_ratio'].iloc[0]  
    return power,coef_t
if Test==2:
    a,b=Transfer_Info('G','B',[2.0, 7.0],[7.0])
    Transfer_Freq_df


def Comm_Cost(g='alex',fn=[[0],[1],[2],[3],[4],[5],[6],[7]],cmps=8*'B', debug=False):
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
    # Layers are indexed from 1 (because first index in cmps, fn, and fc is for input)
    # We start from layer=2 because comparing with previous layer
    if debug:
        print(f'cmps: {cmps}')
    for i in range(2,len(fn)):
        if cmps[i]!=cmps[i-1]: 
            if debug:
                print(f'transfer happen between {cmps[i-1]} and {cmps[i]}')
            #transfer_time=transfer_times[g][i][cmps[i]][cmps[i-1]]
            transfer_time=Transfers_df[(Transfers_df["Graph"]==g) &
                                       (Transfers_df["Layer"]==i-1) &
                                       (Transfers_df["Dest"]==cmps[i]) &
                                       (Transfers_df["Src"]==cmps[i-1])]["Time"].iloc[0]
            if debug:
                print(f'{fc[i-1]}--{fc[i]}')
            transfer_power,time_ratio=Transfer_Info(p1=cmps[i-1],p2=cmps[i],f1=fc[i-1],f2=fc[i],_debug=debug)
        
            scaled_time=transfer_time * time_ratio
            transfer_energy=scaled_time * transfer_power
            
            transfer_t+=scaled_time
            transfer_e+=transfer_energy
            if debug:
                print(f"Transfer between layer {i-1} and {i} (inexed start with 1)")
                print(f'transfer_time: {transfer_time}, time_ratio:{time_ratio}, scaled_time:{scaled_time}')
                print(f'transfer_power:{transfer_power}, transfer_energy:{transfer_energy}')
                print(f'total time:{transfer_t}')
                print(f'total energy:{transfer_e}')
    return transfer_t, transfer_e/1000.0


def Inference_Cost(_graph='alex',_freq=[[0],[1],[2],[3],[4],[5],[6],[7]],_order=8*'B',_dvfs_delay=3.5, _debug=False):
    total_time=0
    total_energy=0
    if _dvfs_delay=="variable":
        t_cmp,e_cmp=Comp_Cost_variable_dvfs_delay(g=_graph,fn=_freq,cmps=_order, debug=_debug)
    else:
        t_cmp,e_cmp=Comp_Cost(g=_graph,fn=_freq,cmps=_order,dvfs_delay=_dvfs_delay, debug=_debug)
    t_cmu,e_cmu=Comm_Cost(g=_graph,fn=_freq,cmps=_order, debug=_debug)
    total_time=t_cmp + t_cmu
    total_energy=e_cmp + e_cmu
    return total_time,total_energy
if Test==2:
    print(Inference_Cost(_dvfs_delay=0))
    print(Inference_Cost(_dvfs_delay=3.5))
    print(Inference_Cost(_dvfs_delay='variable'))


def Parse_Power_total(file_name,graph,order,frqss):
    global tts,powers
    powers,tts=Read_Power(file_name)
    power_df = pd.DataFrame(columns=['graph', 'order', 'freq', 'input_power','task_power'])
    NL=1
    Num_runs=Num_frames+Num_Warm_up
    nn=((2*NL*Num_frames)+2)
    nn=((2*NL*Num_runs)+2)
    nnn=nn*len(frqss)
    if len(powers)!=nnn:
        print(f"bad power size: {len(powers)}")
        print(f'Expected size is:NFreqx((2xNLxn)+2) which is {len(frqss)}x((2x{NL}x{Num_runs})+2)={nnn}')
        #input("what")
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

# +


def Parse_total(timefile,graph,order,frqss):
    with open(timefile,errors='ignore') as ff:
        lines=ff.readlines()
    freq_indx=0
    freqs=frqss[0]
    input_time=-1
    output_time=-1
    parts=[]
    df_time = pd.DataFrame(columns=['graph', 'order', 'freq', 'input_time', 'task_time','output_time', 'total_time'])
    for l in lines:        
        if "Profiling these DVFS settings finised" in l:
            print(f'Input_time:{input_time}')
            s=sum(parts)
            print(f'parts:{parts}, sum:{s}')            
            
            df_time.loc[len(df_time)]={'graph':graph, 'order':order, 'freq': tuple(freqs), 'input_time':input_time, 'task_time':s-input_time,'output_time':output_time, 'total_time':s}
            input_time=-1
            output_time=-1
            parts=[]
            
        pattern = r".* Running Graph with .* LW DVFS"
        if re.match(pattern,l):
            freqs=frqss[freq_indx]
            print(f'Next freq:{freqs}')
            freq_indx=freq_indx+1
            
            
        pattern = r'Graph(\d+)\s+Input:\s+([\d.]+)\s+Task:\s+([\d.]+)\s+send:\s+([\d.]+)\s+Out:\s+([\d.]+)\s+Process:\s+([\d.]+)'
        matches = re.findall(pattern, l)

        if matches:
            graph_number, input_value, task_value, send_value, out_value, process_value = matches[0]    
            print(graph_number,input_value,task_value, send_value, out_value, process_value)
            #input()
            parts.append(float(process_value))
            if int(graph_number)==0:
                input_time=float(input_value)
            output_time=float(out_value)
            
            
    if df_time.shape[0] != len(frqss):
        print(f'Parse performance error: number of runs {df_time.shape[0]} is not equals to number of freqs {len(frqss)}')
        input()
    return df_time


# +
def Real_Evaluation(g="alex",_ord='GBBBBBBB',_fs=[ [ [0,0],[0],[0],[0],[0],[0],[0],[0] ] ],suffix=''):
    pf="pwr_whole.csv"
    tf="temp_whole.txt"
    
    if len(_ord)==1:
        _ord=NLayers[g]*_ord
    global Evaluations_df
    if suffix=='':
        suffix=g
    EvalFile=Evaluations_csv.with_name(Evaluations_csv.name.replace(".csv", "_" + suffix + ".csv"))
    #EvalFile=Evaluations_csv.split(".")[0]+'_'+g+Evaluations_csv.split(".")[0]
    if EvalFile.exists():
        Evaluations_df=pd.read_csv(EvalFile)
    else:
        Evaluations_df=pd.DataFrame(columns=['graph','order','freq','input_time','task_time','output_time','total_time', 'input_power','task_power'])
    
    
    cols_to_add = ['input_e','task_e','output_e','total_e', 'power_efficiency']
    # Loop through the columns to add
    
    for col in cols_to_add:
        # Check if the column is not already in the DataFrame
        if col not in Evaluations_df.columns:
            # Add the column with default values (in this case, NaN)
            Evaluations_df[col] = pd.Series([float('nan')] * len(Evaluations_df))
            

    
    
    new_fs=[]
    repeat=False
    #print(evaluations)
    for ff in _fs:
        tpl_f=ff
        if type(ff)==list:
            tpl_f=(tuple(tuple(i) for i in ff))
        
        row=Evaluations_df[(Evaluations_df['order']==_ord) & (Evaluations_df['freq']==str(tpl_f)) & (Evaluations_df['graph']==g)]
        
        if repeat==False and row.shape[0]==0:
            new_fs.append(ff)
        else:
            print(f'{_ord}, Freq:{ff} already evaluated:')
            try:
                display(row)
            except:
                pprint.pprint(row)
            if pd.isna(row.reset_index().loc[0,'task_time']):
                new_fs.append(ff)

    if len(new_fs)==0:
        return Evaluations_df
    global n
    Profile(_ff=new_fs, _Num_frames=Num_frames, order=_ord, graph=g, pwr=pf, tme=tf,caching=False,kernel_c=96*50)
    time_df=Parse_total(timefile=tf, graph=g, order=_ord, frqss=new_fs)
    power_df=Parse_Power_total(file_name=pf,graph=g,order=_ord,frqss=new_fs)
    if type(_fs[0])==list:
        power_df['freq'] = power_df['freq'].apply(lambda x: str(tuple(tuple(i) for i in x)) )
        time_df['freq'] = time_df['freq'].apply(lambda x: str(tuple(tuple(i) for i in x)) )
    merged_df = pd.merge(power_df, time_df, on=['graph', 'order', 'freq'])


    '''input_time=time_df['input_time'].iloc[0]
    task_time=time_df['task_time'].iloc[0]
    input_power=power_df['input_power'].iloc[0]
    task_power=power_df['task_power'].iloc[0]
    input_e=input_power*input_time
    task_e=task_power*task_time
    total_e=input_e+task_e
    merged_df['input_e']=input_e/1000.0
    merged_df['task_e']=task_e/1000.0
    merged_df['total_e']=total_e/1000.0'''
    
    merged_df['input_e']=merged_df['input_power']*merged_df['input_time']/1000.0
    merged_df['task_e']=merged_df['task_power']*merged_df['task_time']/1000.0
    merged_df['output_e']=merged_df['input_power']*merged_df['output_time']/1000.0
    merged_df['total_e']=merged_df['input_e']+merged_df['task_e']+merged_df['output_e']
    merged_df['power_efficiency']=1000.0/merged_df['total_e']
    try:
        display(merged_df)
    except:
        pprint.pprint(merged_df)
    #merged_df=merged_df.reset_index(drop=True,inplace=True)
    
    for i,k in merged_df.iterrows(): 
        r=Evaluations_df[(Evaluations_df['graph']==k['graph']) & (Evaluations_df['order']==k['order']) & (Evaluations_df['freq']==str(k['freq']))].index
        if(len(r)):
            r=r[0]
            for j,col in enumerate(Evaluations_df):
                Evaluations_df.iloc[r,j]=k[col]
        else:
            Evaluations_df=pd.concat([Evaluations_df,merged_df], ignore_index=True)
        

    if pd.isna(Evaluations_df['total_e']).any():
        Evaluations_df['input_e']=Evaluations_df['input_power']*Evaluations_df['input_time']/1000.0
        Evaluations_df['task_e']=Evaluations_df['task_power']*Evaluations_df['task_time']/1000.0
        Evaluations_df['output_e']=Evaluations_df['input_power']*Evaluations_df['output_time']/1000.0
        Evaluations_df['total_e']=Evaluations_df['input_e']+Evaluations_df['task_e']+merged_df['output_e']
        merged_df['power_efficiency']=1000.0/merged_df['total_e']
    
        
    Evaluations_df.to_csv(EvalFile,index=False)
    return Evaluations_df

    

if Test==3:
    Real_Evaluation(g="alex",_ord='GBBBBBBB',_fs=[ [ [4,6],[6],[6],[6],[6],[6],[6],[6] ] ])

#fig 2 of dac paper
def eval_single_pe():
    for _g in graphs:
        for pe in ['L','B','G','N']:
            Real_Evaluation(g=_g,_ord=pe*NLayers[_g],_fs=[[['max']]])
            Real_Evaluation(g=_g,_ord=pe*NLayers[_g],_fs=[[['min']]])
if Test==4:
    eval_single_pe()

def AOA():
    for _g in graphs:
        Real_Evaluation(g=_g,_ord='G',_fs=[[["min"]]],suffix="AOA")
        
if Test==2:
    AOA()
# -



# +
#Fixed freq
Motivation_Fig2=False
#def Motivation_Fig2():
if Motivation_Fig2:
    _g='mobile'
    N=NLayers[_g]
    Real_Evaluation(g=_g,_ord='L',_fs=[[[5]]*N,[[4]]*N,[[3]]*N,[[2]]*N,[[1]]*N,[[0]]*N],suffix="Motivation_Figure")
    Real_Evaluation(g=_g,_ord='B',_fs=[[[0]]*N,[[1]]*N,[[2]]*N,[[3]]*N,[[4]]*N,[[5]]*N,[[6]]*N,[[7]]*N],suffix="Motivation_Figure")
    Real_Evaluation(g=_g,_ord='G',_fs=[[[0,0]]*N,[[1,1]]*N,[[2,2]]*N,[[3,3]]*N,[[4,4]]*N],suffix="Motivation_Figure")
    
#Real_Evaluation(g="google",_ord='LLLLLLLLLLL',_fs=[[[5]]*11],suffix="ttt")

# +
#This version of Real_Evalutaion is for evaluating GA results, so it get the df instead of using global one
def Real_Evaluation_For_GA(g="alex",_ord='GBBBBBBB',_fs=[ [ [0,0],[0],[0],[0],[0],[0],[0],[0] ] ],Evals_df=None,FileName=""):
    pf="pwr_whole.csv"
    tf="temp_whole.txt"
    
    if len(_ord)==1:
        _ord=NLayers[g]*_ord
    
    EvalFile = FileName
    if EvalFile.exists():
        Evals_df=pd.read_csv(EvalFile)
    #else:
    #    Evals_df=pd.DataFrame(columns=['graph','order','freq','socre','input_time','task_time','total_time', 'input_power','task_power'])
    cols_to_add = ['input_time','task_time','total_time', 'input_power','task_power','input_e','task_e','total_e']
    # Loop through the columns to add
    
    for col in cols_to_add:
        # Check if the column is not already in the DataFrame
        if col not in Evals_df.columns:
            # Add the column with default values (in this case, NaN)
            Evals_df[col] = pd.Series([float('nan')] * len(Evals_df))
    
    
    
    new_fs=[]
    repeat=False
    #print(evaluations)
    for ff in _fs:
        tpl_f=ff
        if type(ff)==list:
            tpl_f=(tuple(tuple(i) for i in ff))
        
        row=Evals_df[(Evals_df['order']==_ord) & (Evals_df['freq']==str(tpl_f)) & (Evals_df['graph']==g)]
        
        if repeat==False and row.shape[0]==0:
            print(f'new freq adding freq {ff}')
            new_fs.append(ff)
        else:
            print(f'{_ord}, Freq:{ff} already evaluated:')
            try:
                display(row)
            except:
                pprint.pprint(row)
            
            if pd.isna(row.reset_index()['task_time']).all():
                print(f'task_time is none adding freq {ff}')
                new_fs.append(ff)

    if len(new_fs)==0:
        return Evals_df
    
    Profile(_ff=new_fs, _Num_frames=Num_frames, order=_ord, graph=g, pwr=pf, tme=tf,caching=False,kernel_c=96*50)
    time_df=Parse_total(timefile=tf, graph=g, order=_ord, frqss=new_fs)
    power_df=Parse_Power_total(file_name=pf,graph=g,order=_ord,frqss=new_fs)
    if type(_fs[0])==list:
        power_df['freq'] = power_df['freq'].apply(lambda x: str(tuple(tuple(i) for i in x)) )
        time_df['freq'] = time_df['freq'].apply(lambda x: str(tuple(tuple(i) for i in x)) )
    merged_df = pd.merge(power_df, time_df, on=['graph', 'order', 'freq'])


    '''input_time=time_df['input_time'].iloc[0]
    task_time=time_df['task_time'].iloc[0]
    input_power=power_df['input_power'].iloc[0]
    task_power=power_df['task_power'].iloc[0]
    input_e=input_power*input_time
    task_e=task_power*task_time
    total_e=input_e+task_e
    merged_df['input_e']=input_e/1000.0
    merged_df['task_e']=task_e/1000.0
    merged_df['total_e']=total_e/1000.0'''
    
    merged_df['input_e']=merged_df['input_power']*merged_df['input_time']/1000.0
    merged_df['task_e']=merged_df['task_power']*merged_df['task_time']/1000.0
    merged_df['total_e']=merged_df['input_e']+merged_df['task_e']
    merged_df['score']=Evals_df['score']
    try:
        display(merged_df)
    except:
        pprint.pprint(merged_df)
    #merged_df=merged_df.reset_index(drop=True,inplace=True)
    
    for i,k in merged_df.iterrows(): 
        r=Evals_df[(Evals_df['graph']==k['graph']) & (Evals_df['order']==k['order']) & (Evals_df['freq']==str(k['freq']))].index
        if(len(r)):
            r=r[0]
            for j,col in enumerate(Evals_df):
                if col!='score':
                    Evals_df.iloc[r,j]=k[col]
        else:
            Evals_df=pd.concat([Evals_df,merged_df], ignore_index=True)
        

    Evals_df.to_csv(EvalFile,index=False)
    return Evals_df



# -

#This is for reading GA result file and run the the real evalation for GA
def Run_Eval_For_GA(_FileName):
    
    
    if _FileName.exists():
        Evals_df=pd.read_csv(_FileName).drop_duplicates()
    else:
        print("Ga result file is not existed")
        return
    
    cases=Evals_df.shape[0]
    print(f'There are {cases}')
    

    for g in  graphs:
        
        grouped = Evals_df[Evals_df['graph']==g].groupby('order')
        unique_values_order = Evals_df[Evals_df['graph']==g]['order'].unique()

        # Loop through the unique values in column 'order'
        for value in unique_values_order:
            # Get the group corresponding to the current value in column 'order'
            group = grouped.get_group(value)
            # Get the values in column 'freq' for the current group
            column_freq_values = group['freq'].values
            # Print the value in column 'A' and the corresponding values in column 'freq'
            print(f"Value in column 'order': {value}")
            print(f"Values in column 'freq': {column_freq_values}")
            print("----")
            list_fs=format_to_list(column_freq_values)
            Real_Evaluation_For_GA(g,_ord=value,_fs=list_fs,Evals_df=Evals_df,FileName=_FileName)
if Test==3:
    Run_Eval_For_GA(_FileName=GA_Results_PELSI)
    Run_Eval_For_GA(_FileName=GA_Results_LW)


# +
def Fill_prediction(_FileName, dvfs_delay):
    if _FileName.exists():
        Evals_df=pd.read_csv(_FileName).drop_duplicates()
    else:
        print("Ga result file is not existed")
        return
    
    cases=Evals_df.shape[0]
    print(f'There are {cases}')
    
    Regenerate_Predeiction=False
    Regenerate_Errors=False
    
    def prediction(row):
        #print(row)
        graph=row['graph']
        freq=format_to_list([row['freq']])[0]
        order=row['order']
        #print(graph,freq,order,dvfs_delay)
        return Inference_Cost(_graph=graph,_freq=freq,_order=order,_dvfs_delay=dvfs_delay, _debug=False)
    if 'Predicted_Time' not in Evals_df:
        Evals_df[['Predicted_Time','Predicted_Energy']]=Evals_df.apply(prediction,axis=1, result_type='expand')
    if 'Predicted_Time' in Evals_df:
        if pd.isna(Evals_df['Predicted_Time']).any() or Regenerate_Predeiction:
            Evals_df[['Predicted_Time','Predicted_Energy']]=Evals_df.apply(prediction,axis=1, result_type='expand')
    
    #display(Evals_df)
    def calc_EE(row):
        Measured=1000.0/row['total_e']
        Pred=1000.0/row['Predicted_Energy']
        Err=(Pred-Measured)/Measured
        return 100.0*Err
    
    def calc_Power(row):
        measured=row['total_e']/row['total_time']
        pred=row['Predicted_Energy']/row['Predicted_Time']
        Err=100*(pred-measured)/measured
        return Err
    
    def calc_FPS(row):
        measured=1000/row['total_time']
        pred=1000/row['Predicted_Time']
        Err=100.0*(pred-measured)/measured
        return Err
    
    if 'Error_Time' not in Evals_df or Regenerate_Errors:
        Evals_df['Error_Time']=Evals_df.apply(lambda x:100*(x['Predicted_Time']-x['total_time'])/x['total_time'],axis=1)
    if 'Error_Energy' not in Evals_df or Regenerate_Errors:
        Evals_df['Error_Energy']=Evals_df.apply(lambda x:100*(x['Predicted_Energy']-x['total_e'])/x['total_e'],axis=1)
    if 'Error_EE' not in Evals_df or Regenerate_Errors:
        Evals_df['Error_EE']=Evals_df.apply(calc_EE,axis=1)
    #Evals_df['Error_Power']=Evals_df.apply(lambda x:100*abs( (x['Predicted_Energy']/x['Predicted_Time']) - (x['total_e']/x['total_time']) /(x['total_e']/x['total_time']) ),axis=1)
    if 'Error_Power' not in Evals_df or Regenerate_Errors:
        Evals_df['Error_Power']=Evals_df.apply(calc_Power,axis=1)
    if 'Error_FPS' not in Evals_df or Regenerate_Errors:
        Evals_df['Error_FPS']=Evals_df.apply(calc_FPS,axis=1)
    new_file=_FileName.with_name(_FileName.name.replace(".csv", "_prediction.csv"))
    Evals_df.to_csv(new_file)

if Test==2:
    for g in graphs:
        fname=Path('Evaluations_'+g+'.csv')
        Fill_prediction(fname, 'variable')
# -

#def Anlze_Error():
#if True:
if Test==1:
    for g in graphs:
        print(f'Graph: {g}')
        if not Path('Evaluations_'+g+'_prediction.csv').exists():
            continue
        Evals_df=pd.read_csv('Evaluations_'+g+'_prediction.csv')
        #error_time = abs(100.0*(Evals_df['Predicted_Time'] - Evals_df['total_time'])/Evals_df['total_time'])
        #print(abs(error_time).describe())
        #error_energy = abs(100.0*((1000.0/Evals_df['Predicted_Energy']) - (1000.0/Evals_df['total_e']))/(1000.0/Evals_df['total_e']))
        #error_energy=Evals_df['Error_Time']
        error_energy=abs(Evals_df['Error_Energy'])
        #error_energy=Evals_df['Error_EE']
        #plt.hist(error_energy, bins=50, density=True)
        print(error_energy.describe())
        # Add normal curve
        mu, std = norm.fit(error_energy)
        x = np.linspace(-30, 30, 200)

        y = norm.pdf(x, mu, std)
        plt.plot(x, y, label=g)
        plt.xlabel('Error%')
        plt.ylabel('Pdf')
        
    plt.legend()
    plt.show()


# +
def prediction(File,row_num,dvfs_delay):
        _FileName=Path(File)
        if _FileName.exists():
            Evals_df=pd.read_csv(_FileName).drop_duplicates()
        else:
            print("Ga result file is not existed")
            return

        cases=Evals_df.shape[0]
        print(f'There are {cases}')
        #print(row)
        #Evals_df=Evals_df.sort_values('Error_Time',ascending=False)
        row=Evals_df.iloc[row_num]
        graph=row['graph']
        freq=format_to_list([row['freq']])
        order=row['order']
        #print(graph,freq,order,dvfs_delay)
        t,e=Inference_Cost(_graph=graph,_freq=freq[0],_order=order,_dvfs_delay=dvfs_delay, _debug=True)
        print(f'total_time:{t}, total_e:{e}')
        run=False
        if run:
            Real_Evaluation(g=graph,_ord=order,_fs=freq)
            
        return t,e
    

if Test==2:
    g='google'
    prediction('Evaluations_'+g+'_prediction.csv',0,'variable')


# +
#Layers_df[(Layers_df['Graph']=='google') & (Layers_df['Layer']==4) & (Layers_df['Component']=='B') &(Layers_df['Freq']==0)]

# +
#Value('google','L',[0],[3],'task','Time')
# -

def _Test():
    _fs=[ [ [0],[1],[2],[3],[4],[5],[6],[7] ],
         [ [7],[6],[5],[4],[3],[2],[1],[0] ] ]
    _order='BBBBBBBB'
    _g="alex"
    for fs in _fs:
        Real_Evaluation(g="alex",_ord=_order,_fs=[fs])
        ''' Profile(_ff=[fs], _Num_frames=Num_frames, order=_order, graph=_g, pwr="pwr.csv", tme="temp.txt",caching=False)
        time=Parse(timefile="temp.txt", graph=_g, order=_order, frqss=[fs])
        power=Parse_Power(pwrfile="pwr.csv", graph=_g, order=_order, frqss=[fs])
        print(time)
        print(power)'''
        _dvfs_delay=3.5
        #np.reshape(fs,-1)
        cmp=Comp_Cost(g=_g,fn=fs,cmps=_order,dvfs_delay=_dvfs_delay, debug=False)
        cmm=Comm_Cost(g=_g,fn=fs,cmps=_order,dvfs_delay=_dvfs_delay, debug=False)
        print(cmp)
        print(cmm)


def Transfer_Cost(_order,fs,_kernel_c=96*100):   
    g="test_transfer"
    trans=[]
    trans_pwr=[]
    
    Synthetic_Tranfer_logs.mkdir(parents=True, exist_ok=True)
    pwrfile=f'{Synthetic_Tranfer_logs}/power_{g}_{_order}_{str(_kernel_c)}_{str(fs)}.csv'
    timefile=f'{Synthetic_Tranfer_logs}/time_{g}_{_order}_{str(_kernel_c)}_{str(fs)}.txt'
    
    Profile(_ff=fs, _Num_frames=Num_frames, order=_order, graph=g, pwr=pwrfile, tme=timefile,caching=False,kernel_c=_kernel_c)
    
    trans,transfer_df_time=Parse_transfer_graph(timefile=timefile, graph=g, order=_order, frqss=fs)

    trans_pwr,trans_pwr_df=Parse_Power_Transfer_graph(file_name=pwrfile,graph=g,order=_order,frqss=fs)
    
    return trans,trans_pwr,trans_pwr_df,transfer_df_time


def Explore_Freq_on_Transfering():

    _fs={'BL':[[[0],[i]] for i in range(NFreqs['L'])],
            'LB':[[[0],[i]] for i in range(NFreqs['B'])],
            'GB':[[[0,i],[i]] for i in range(NFreqs['B'])],
            'LG':[[[0],[0,i]] for i in range(NFreqs['B'])],
            'BG':[[[i],[0,i]] for i in range(NFreqs['B'])],
        }
    global Transfer_Freq_df
    if Transfer_Freq_csv.exists():
        Transfer_Freq_df=pd.read_csv(Transfer_Freq_csv)
    else:
        Transfer_Freq_df = pd.DataFrame(columns=['kernels', 'order', 'SenderFreq','RecFreq', 'transfer_time', 'transfer_power'])
    kernel_cs=[150]
    kernel_cs=[96*i for i in kernel_cs]
    
    #global trans,trans_pwr,trans_pwr_df,transfer_df_time
    for kernel_c in kernel_cs:
        for order in _fs:
            print(f'order:{order}, kernels:{kernel_c}, shape: {Transfer_Freq_df[(Transfer_Freq_df["order"]==order) & (Transfer_Freq_df["kernels"]==kernel_c)].shape[0]}')
            if Transfer_Freq_df[(Transfer_Freq_df['order']==order) & (Transfer_Freq_df['kernels']==kernel_c)].shape[0]==0:
                try:
                #if True:
                    trans,trans_pwr,trans_pwr_df,transfer_df_time=Transfer_Cost(_order=order,fs=_fs[order],_kernel_c=kernel_c)
                    #transfer_df_freq.loc[len(transfer_df_freq)] = {"kernels":kernel_c, "c":orders[_c], "transfer_time":transfer_df_time, "transfer_power":trans_pwr_df}
                    trans_pwr_df['freq'] = trans_pwr_df['freq'].apply(lambda x: tuple(tuple(i) for i in x))
                    transfer_df_time['freq'] = transfer_df_time['freq'].apply(lambda x: tuple(tuple(i) for i in x))
                    merged_df = pd.merge(trans_pwr_df, transfer_df_time, on=['order', 'freq','SenderFreq','RecFreq'])
                    merged_df['kernels']=kernel_c
                    Transfer_Freq_df=pd.concat([Transfer_Freq_df,merged_df], ignore_index=True)
                    print(f'merged is:\n{merged_df}')
                    print(f'accumulated result is:\n{Transfer_Freq_df}')
                    Transfer_Freq_df.to_csv(Transfer_Freq_csv,index=False)
                    time.sleep(5)
                    #input()
                except Exception as e:
                    print("Error occurred:", e)
                    print("Traceback:")
                    traceback.print_exc()
                    ab()
    first_transfer_time = Transfer_Freq_df.groupby('order')['transfer_time'].first()
    first_transfer_power = Transfer_Freq_df.groupby('order')['transfer_power'].first()
    Transfer_Freq_df['time_ratio'] = Transfer_Freq_df['transfer_time'] / Transfer_Freq_df['order'].map(first_transfer_time)
    Transfer_Freq_df['power_ratio'] = Transfer_Freq_df['transfer_power'] / Transfer_Freq_df['order'].map(first_transfer_power)   
    Transfer_Freq_df.to_csv(Transfer_Freq_csv,index=False)
    plt.plot(Transfer_Freq_df['freq'],Transfer_Freq_df['time_ratio'])
if Test==2:
    Explore_Freq_on_Transfering()


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
if Test==2:
    Plot_Transfer_VS_Data_size("BG","min")


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
                if Transfer_Data_Size_Max_Freq_df[(Transfer_Data_Size_Max_Freq_df['order']==order) & 
                                                  (Transfer_Data_Size_Max_Freq_df['kernels']==kernel_c)].shape[0]==0:
                    try:
                        if order[1]=='G' and kernel_c > 150*96:
                            continue
                                                    
                        ff=_fs["max"][order]
                        trans,trans_pwr,trans_pwr_df,transfer_df_time=Transfer_Cost(_order=order,fs=ff,_kernel_c=kernel_c)
                        Transfer_Data_Size_Max_Freq_df.loc[len(Transfer_Data_Size_Max_Freq_df)] = {"kernels":kernel_c, "order":order, "transfer_time":trans[1], "transfer_power":trans_pwr[1]}
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
                if Transfer_Data_Size_Min_Freq_df[(Transfer_Data_Size_Min_Freq_df['order']==order) & 
                                                  (Transfer_Data_Size_Min_Freq_df['kernels']==kernel_c)].shape[0]==0:
                    try:
                        if order[1]=='G' and kernel_c > 150*96:
                            continue
                        ff=_fs["min"][order]
                        trans,trans_pwr,trans_pwr_df,transfer_df_time=Transfer_Cost(_order=order,fs=ff,_kernel_c=kernel_c)
                        Transfer_Data_Size_Min_Freq_df.loc[len(Transfer_Data_Size_Min_Freq_df)] = {"kernels":kernel_c, "order":order, "transfer_time":trans[1], "transfer_power":trans_pwr[1]}
                        Transfer_Data_Size_Min_Freq_df.to_csv(Transfer_Data_Size_Min_Freq_csv,index=False)
                        time.sleep(10)
                    except:
                        ab()               

        return Transfer_Data_Size_Min_Freq_df


# +
def Run_Explore_Data_Size_on_Transfering(_freq_mode="max"):
    orders={0:'BL', 1:'LB', 2:'GB', 3:'LG', 4:'BG'}
    Explore_Data_Size_on_Transfering(freq_mode=_freq_mode)
    for i in orders:
        Plot_Transfer_VS_Data_size(order=orders[i],freq_mode=_freq_mode)
        
if Test==2:
    Run_Explore_Data_Size_on_Transfering(_freq_mode="max")
    Run_Explore_Data_Size_on_Transfering(_freq_mode="min")


# -

def Compute_Layer_Percentage():
#if True:
    sum_time_per_graph_component = Layers_df.groupby(['Graph', 'Component', 'Freq', 'Freq_Host'])['Time'].sum().reset_index()
    pd.set_option('display.max_rows', 1000)
    Layers_With_Percentage_df=Layers_df.merge(sum_time_per_graph_component, on=['Graph', 'Component', 'Freq', 'Freq_Host'], suffixes=('', '_sum'))
    Layers_With_Percentage_df['Time_Percentage'] = Layers_With_Percentage_df['Time'] / Layers_With_Percentage_df['Time_sum'] * 100
    #print(Layers_With_Percentage_df[(Layers_With_Percentage_df["Graph"]=="alex") & (Layers_With_Percentage_df["Freq"]==0) & (Layers_With_Percentage_df["Component"]=="G") & (Layers_With_Percentage_df["Freq_Host"]==0)]["Time_Percentage"].sum())
    Layers_With_Percentage_df.to_csv(Layers_With_Percentage_csv, index=False)
    Layers_Percentage_df=Layers_With_Percentage_df.groupby(['Graph', 'Component','Layer','Metric'])['Time_Percentage'].mean().reset_index()
    #print(Layers_Percentage_df)
    #Layers_Percentage_df.to_csv(Layers_Percentage_csv, index=False)
    pivot_df = Layers_Percentage_df.pivot_table(index=['Graph', 'Layer', 'Metric'], columns='Component', values='Time_Percentage')
    pivot_df.columns = ['Time_Percentage_{}'.format(col) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    
    pivot_df['Time_Percentage_Average'] = pivot_df[['Time_Percentage_B', 'Time_Percentage_G', 'Time_Percentage_L']].mean(axis=1)
    
    pivot_df = pivot_df.groupby(['Graph', 'Layer']).sum().reset_index()
    pivot_df.to_csv(Layers_Percentage_csv, index=False)
    try:
        display(pivot_df)
    except:
        pprint.pprint(pivot_df)


# +
## plot energy of layers running with different components with freq min
def _Analyze_Components(g=['alex']):
    Layers_df['Energy']=Layers_df['Time']*Layers_df['Power']/1000.0
    grouped_df = Layers_df[(Layers_df['Graph'].isin(g)) & 
                        (Layers_df['Metric'].isin(['in','task'])) & 
                        (Layers_df['Freq'].isin(range(10))) & 
                        (Layers_df['Freq_Host'].isin([0,-1]))&
                        (Layers_df['Layer'].isin(range(10))) ].groupby(['Component','Layer','Metric'])\
                        ['Time','Power','Energy'].mean().reset_index()

    
    
    #display(grouped_df)

    '''grouped_df['Layer'] = grouped_df['Layer'].where(grouped_df['Metric'] != 'in', 'input')
    grouped_df = grouped_df.drop('Metric', axis=1)
    print(grouped_df)'''

    aggregations = {
        'Time': 'sum',
        'Power': 'mean',
        'Energy': 'sum'
    }
    grouped_df = grouped_df.groupby(['Component', 'Layer']).agg(aggregations).reset_index()
    #display(grouped_df)
    grouped_df['Power-Efficiency']=1000.0/grouped_df['Energy']

    pivot_df = grouped_df.pivot_table(index=['Layer'], columns='Component', values=['Time', 'Energy', 'Power-Efficiency'])
    pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    try:
        display(pivot_df)
    except:
        pprint.pprint(pivot_df)
        
    font_size = 18
    PE_cols = ['Power-Efficiency_G', 'Power-Efficiency_B', 'Power-Efficiency_L']
    #pivot_table.plot(kind='bar', stacked=False, figsize=(10, 5.625))
    energy_plot = pivot_df.plot(x='Layer', y=PE_cols, kind='bar', title='Power-Efficiency of {}net layers for Average Freqs'.format(g[0]),figsize=(10, 5.625))
    energy_plot.set_xlabel('Layer', fontsize=font_size-2)
    energy_plot.set_ylabel('Power-Efficiency FPS/Watt', fontsize=font_size-2)
    
    energy_plot.title.set_fontsize(font_size)
    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    plt.legend(fontsize=font_size-2)
    
    '''energy_plot = pivot_df.plot(x='Layer', y=PE_cols, kind='bar', title='Energy-Efficiency of {}net layers for Average Freqs'.format(g[0]))
    energy_plot.set_xlabel('Layer')
    energy_plot.set_ylabel('Energy-Efficiency FPS/Watt')
    legend = energy_plot.legend()
    for label in legend.get_texts():
        label.set_text(label.get_text().replace('Power-Efficiency', 'Energy-Efficiency'))'''
    
    plt.show()
        
    energy_cols = ['Energy_G', 'Energy_B', 'Energy_L']
    energy_plot = pivot_df.plot(x='Layer', y=energy_cols, kind='bar', title='{} Energy for Average Freqs'.format(g))
    energy_plot.set_xlabel('Layer')
    energy_plot.set_ylabel('Energy')
    plt.show()

    # Plot Time columns
    time_cols = ['Time_G', 'Time_B', 'Time_L']
    time_plot = pivot_df.plot(x='Layer', y=time_cols, kind='bar', title='{} Time for Average Freqs'.format(g))
    time_plot.set_xlabel('Layer')
    time_plot.set_ylabel('Time')
    plt.show()

if Test==1:
    _Analyze_Components(g=['alex'])


# -

## plot (and extract and save result to csv files) energy of layers running with different components with freq min
def Analyze_Components(g=['alex']):

    grouped_df = Layers_df[(Layers_df['Graph'].isin(g)) & 
                        (Layers_df['Metric'].isin(['in','task'])) &  
                        (Layers_df['Freq_Host'].isin([0,-1]))&
                        (Layers_df['Layer'].isin(range(10))) ].groupby(['Freq','Component','Layer','Metric'])\
                        ['Time','Power'].sum().reset_index()

    grouped_df['Energy']=grouped_df['Time']*grouped_df['Power']/1000.0
    
    #print(grouped_df)

    '''grouped_df['Layer'] = grouped_df['Layer'].where(grouped_df['Metric'] != 'in', 'input')
    grouped_df = grouped_df.drop('Metric', axis=1)
    print(grouped_df)'''

    aggregations = {
        'Time': 'sum',
        'Power': 'mean',
        'Energy': 'sum'
    }
    grouped_df = grouped_df.groupby(['Freq','Component', 'Layer']).agg(aggregations).reset_index()
    #print(grouped_df)

    grouped_df['Power-Efficiency']=1000.0/grouped_df['Energy']
    
    pivot_df = grouped_df.pivot_table(index=['Layer','Freq'], columns='Component', values=['Time', 'Energy','Power-Efficiency'])
    pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
    #pivot_df = grouped_df.pivot_table(index=['Layer','Freq'], columns='Component', values='Energy')
    #pivot_df.columns = ['Energy_{}'.format(col) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    #print(pivot_df)
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
        'Time_L_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_L.values)]['Time_L'].values,
        'Power-Efficiency_L_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_L.values)]['Power-Efficiency_L'].values,
        'Power-Efficiency_B_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_B.values)]['Power-Efficiency_B'].values,
        'Power-Efficiency_G_MaxFreq': pivot_df[pivot_df['Freq'].isin(max_freq_G.values)]['Power-Efficiency_G'].values,
    })
    try:
        display(freq_df)
    except:
        pprint.pprint(freq_df)
    energy_cols = ['Energy_G_MaxFreq', 'Energy_B_MaxFreq', 'Energy_L_MaxFreq']
    energy_plot = freq_df.plot(x='Layer', y=energy_cols, kind='bar', title='Energy for Freq Max')
    energy_plot.set_xlabel('Layer')
    energy_plot.set_ylabel('Energy')
    plt.show()
    
    energy_cols = ['Power-Efficiency_G_MaxFreq', 'Power-Efficiency_B_MaxFreq', 'Power-Efficiency_L_MaxFreq']
    energy_plot = freq_df.plot(x='Layer', y=energy_cols, kind='bar', title='Power-Efficiency for Freq Max')
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
        try:
            display(freq_df)
        except:
            pprint.pprint(freq_df)
        
        # Plot Energy-Efficiency columns
        energy_efficiency_cols = ['Power-Efficiency_G', 'Power-Efficiency_B', 'Power-Efficiency_L']
        energy_plot = freq_df.plot(x='Layer', y=energy_efficiency_cols, kind='bar', title='Power-Efficiency for Freq {}'.format(freq))
        energy_plot.set_xlabel('Layer')
        energy_plot.set_ylabel('Power-Efficiency FPS/Watt')
        plt.show()

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
if Test==2:
    Analyze_Components(g=['alex'])


def generate_random_strings(_n, num_strings):
    chars = ['L', 'B', 'G']
    random_strings = []
    for _ in range(num_strings):
        random_string = ''.join(random.choice(chars) for _ in range(_n))
        random_strings.append(random_string)
    return random_strings
#random_strings = generate_random_strings(8, 100)


def Run_Eval(g='alex',num_evals=1000,num_freqs=10):
    EvalFile=Evaluations_csv.with_name(Evaluations_csv.name.replace(".csv", "_" + g + ".csv"))
    if EvalFile.exists():
        Evaluations_df=pd.read_csv(EvalFile)
    else:
        Evaluations_df=pd.DataFrame(columns=['graph','order','freq','input_time','task_time','total_time', 'input_power','task_power'])    
    cases=Evaluations_df[Evaluations_df['graph']==g].shape[0]
    print(f'There are {cases} existed for graph {g}')
    num_evals=max(0,num_evals-cases)
    num_orders=math.ceil(num_evals/num_freqs)
    
    _n=NLayers[g]
    orders=generate_random_strings(_n,num_orders)
               
    fs={}
    for order in orders:
        fs[order]=[]
        for k in range(num_freqs):
            f=[]
            for i,comp in enumerate(order):
                v=[]
                v.append(random.randint(0, NFreqs[comp]-1))
                if comp=='G':
                    v.append(random.randint(0, NFreqs['B']-1))
                f.append(tuple(v))
                
            fs[order].append(str(tuple(f)))
            
            
    for order in fs:
        for f in fs[order]:
            row=Evaluations_df[(Evaluations_df['order']==order) & (Evaluations_df['freq']==str(f)) & (Evaluations_df['graph']==g)]
            if row.shape[0]==0:
                Evaluations_df.loc[len(Evaluations_df)]={"graph":g,"order":order,"freq":f}
            
    Evaluations_df.to_csv(EvalFile,index=False)
    
    grouped = Evaluations_df.groupby('order')
    unique_values_order = Evaluations_df['order'].unique()

    # Loop through the unique values in column 'order'
    for value in unique_values_order:
        # Get the group corresponding to the current value in column 'order'
        group = grouped.get_group(value)
        # Get the values in column 'freq' for the current group
        column_freq_values = group['freq'].values
        # Print the value in column 'A' and the corresponding values in column 'freq'
        print(f"Value in column 'order': {value}")
        print(f"Values in column 'freq': {column_freq_values}")
        print("----")
        list_fs=format_to_list(column_freq_values)
        Real_Evaluation(g,_ord=value,_fs=list_fs)
if Test==3:
    Run_Eval(g='alex')


def Gather_real_profile(_g,_num_evals):
    Finished=False
    while not Finished:
        try:
            Run_Eval(g=_g,num_evals=_num_evals)
            Finished=True
        except Exception as e:
            print("Error occurred:", e)
            print("Traceback:")
            traceback.print_exc()
            # #!sudo apt install sox
            os.system('play -nq -t alsa synth {} sine {}'.format(5, 440))
            #input("Continue?")
            ab()
            time.sleep(5)
#3
if Test==2:
    for g in graphs:
            Gather_real_profile(g,1000)


def main():

    Load_Data()
    
    '''print('\n\n\n\n***************Run_Profile_Transfer_Time\n')
    input('Make sure to set profile mode to PROFILE_MODE_TRANSFER_TIMES in ExecutionHelpers.cpp')
    Run_Profile_Transfer_Time()
    
    print('\n\n\n\n***************Profiling_Layers\n')
    input('Make sure to set profile mode to PROFILE_MODE_LAYERS in ExecutionHelpers.cpp')
    Profiling_Layers()
    
    
    print('\n\n\n\n***************Explore_Freq_on_Transfering\n')
    #input('Make sure to set profile mode to PROFILE_MODE_SYNTHETIC_TRANSFERS in ExecutionHelpers.cpp')
    Explore_Freq_on_Transfering()
    
    # For first kernel size (10*96) it needs several runs because time is small
    # and may power sampling does not happen
    print('\n\n\n\n***************Run_Explore_Data_Size_on_Transfering(Max)\n')
    Run_Explore_Data_Size_on_Transfering(_freq_mode="max")
    
    print('\n\n\n\n***************Run_Explore_Data_Size_on_Transfering(min)\n')
    Run_Explore_Data_Size_on_Transfering(_freq_mode="min")'''
    
    
    print('\n\n\n\n***************Real_Evaluation\n')
    #input('Make sure to set profile mode to PROFILE_MODE_WHOLE_NETWORK in ExecutionHelpers.cpp')
    Real_Evaluation(g="alex",_ord='GBBBBBBB',_fs=[ [ [4,6],[6],[6],[6],[6],[6],[6],[6] ] ])
    Real_Evaluation(g="alex",_ord='BBBBBBBB',_fs=[ [ [0],[1],[2],[3],[4],[5],[6],[7] ] ])
    
    
    print('\n\n\n\n***************Compute_Layer_Percentage\n')
    Compute_Layer_Percentage()
    
    print('\n\n\n\n***************Value function for indexing\n')
    Value('alex','B',[7],7,'task','Time')
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
    _Test()
    
    
    
    print('\n\n\n\n***************Transfer_Info\n')
    a,b=Transfer_Info('G','B',[2.0, 7.0],[7.0])
    print(a,b)
    
    print('\n\n\n\n***************Comm_Cost\n')
    _fn=[[0],[1],[2],[3],[4],[5],[6],[7]]
    Comm_Cost(cmps="LLLBBBBB",debug=True)
    
    print('\n\n\n\n***************Comp_Cost\n')
    print(Comp_Cost(g="alex",cmps='BBBBBBBB',fn=_fn[::-1]))
    
    print('\n\n\n\n***************_Analyze_Components\n')
    _g='google'
    _Analyze_Components(g=[_g])
    
    print('\n\n\n\n***************Analyze_Components\n')
    Analyze_Components(g=[_g])
    
    print('\n\n\n\n***************Random strings:\n')
    _n = 8  # replace with desired length of the random strings
    num_strings = 1000  # replace with desired number of random strings
    random_strings = generate_random_strings(_n, num_strings)
if Test==4:
    main()


def irad():
#if True:
    a=np.array(tts)
    b=np.array([a[j*202:j*202+203] for j in range(10)])
    ind=np.where(a>1000)
    a[ind]




