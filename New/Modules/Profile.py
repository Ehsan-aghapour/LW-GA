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

cnn_dir="/home/ehsan/UvA/ARMCL/Rock-Pi/ComputeLibrary_64_Yolov3/"

cnn={
    "alex":"graph_alexnet_pipeline",
    "google":"graph_googlenet_pipeline",
    "mobile":"graph_mobilenet_pipeline",
    "res50":"graph_resnet50_pipeline",
    "squeeze":"graph_squeezenet_pipeline",
    "test_transfer":"graph_test_transfer_pipeline"
}


graphs=["alex", "google", "mobile", "res50", "squeeze"]
NLayers={"alex":8, "google":11, "mobile":14, "res50":18, "squeeze":10, "test_transfer":2}
NFreqs={"L":6, "B":8, "G":5}
Metrics=["in","task","out","trans"]
Num_frames=10




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

### This is common function to run a case
## Remember to modify ARMcL code based on your desire
def Profile(_ff=[[[0],[1],[2],[3,6],[4],[5],[6],[7]]],_Num_frames=Num_frames,order='BBBGBBBB',graph="alex",pwr="pwr.csv",tme="temp.txt", caching=True, kernel_c=96):
    caching=False
    if os.path.isfile(pwr) and os.path.isfile(tme) and caching:
        print("loading existed files")
        return 
    
    ff=format_freqs(_ff)
    print(f'\n\nformatted freqs:\n {ff}')
    os.system(f"adb push {cnn_dir}/build/examples/Pipeline/{cnn[graph]} /data/local/ARM-CO-UP/test_graph/")
    os.system('adb shell "echo 0 > /sys/class/gpio/gpio157/value"')
    time.sleep(3)
    Power_monitoring = threading.Thread(target=Arduino_read.run,args=(pwr,))
    Power_monitoring.start()
    rr=f"{cnn_dir}/Run_CO-UP model=Alex --n={_Num_frames} --order={order}  push=1 compile=1 --kernel_c={kernel_c}"
    print(f'run command is {rr}')
    oo=open(tme,'w+')
    
    # if you want to set freqs with cin:
    Run_Graph(ff,rr,oo,True)
    
    # if you want to set with run command
    #run_command=rr + f'--freqs=ff[0]'
    #p = subprocess.Popen(run_command.split(),stdout=oo,stderr=oo, stdin=subprocess.PIPE, text=True)
    #time.sleep(5)
    #p.wait()
    
    time.sleep(2)
    Power_monitoring.do_run = False
    oo.close()


#Profile(caching=False,_Num_frames=10)