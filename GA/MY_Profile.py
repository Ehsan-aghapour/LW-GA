import itertools
import sys
from unittest import result
import numpy as np
import pickle
import subprocess
import threading
import Arduino_read
import time
import os


debug=0
UseCachedResult=True
No_Board=False


# 2 run approximately 146 ms --> One run approximately takes 140 ms
# 21 run approximately 271 ms --> One exess run approximately 6 ms
############### Frequency Tables #################################
#LittleFrequencyTable=[500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000]
LittleFrequencyTable = [408000, 600000, 816000, 1008000, 1200000, 1416000]
#BigFrequencyTable=[500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000, 1908000, 2016000, 2100000, 2208000]
BigFrequencyTable = [408000, 600000, 816000, 1008000, 1200000, 1416000, 1608000, 1800000]
GPUFrquencyTable = [200000000, 300000000, 400000000, 600000000, 800000000]
Freq_Table={'L':LittleFrequencyTable, 'B':BigFrequencyTable, 'G':GPUFrquencyTable}
""

###################### Graphs #########################3
_Graphs=['Alex','Google','Mobile','Res50', 'Squeeze']
_Graphs_ARMCL={'Alex':"graph_alexnet_n_pipe_npu",
                'Google':"graph_googlenet_n_pipe_npu",
                'Mobile':"graph_mobilenet_n_pipe_npu",
                'Res50':"graph_resnet50_n_pipe_npu",
                'Squeeze':"graph_squeezenet_n_pipe_npu",}
_Ns={'Alex':8,
    'Google':13,
    'Mobile':28,
    'Squeeze':19,
    'Res50':18,}
_dts={'Alex':1,
    'Google':2,
    'Mobile':3,
    'Squeeze':5,
    'Res50':4,}
# 1->227 2->224
_imgs={'Alex':1,
    'Google':2,
    'Mobile':2,
    'Squeeze':1,
    'Res50':2,}
_lbls={'Alex':1,
    'Google':1,
    'Mobile':1,
    'Squeeze':1,
    'Res50':1,}



################################## Resgression model of transfer timings ##################################
def mapping_time(data_size):
    A=[7.72e-4, 4.62e-11, 105.71]
    t = ( A[0] * data_size + A[1] * (data_size**2) + A[2] )*(10**-3)
    if debug:
        print(f'map time:{t}')
    return t
	
def unmapping_time(data_size):
    A=[2.56e-5, 1.54e-12, 10.96]
    t = ( A[0] * data_size + A[1] * (data_size**2) + A[2] )*(10**-3)
    if debug:
        print(f'unmap time:{t}')
    return t
	
def GPU_copy_time(data_size):
    A=[1.87e-3, 1.12e-10, 68.74]
    t = ( A[0] * data_size + A[1] * (data_size**2) + A[2] )*(10**-3)
    if debug:
        print(f'GPU copy time:{t}')
    return t

def NPU_copy_time(data_size):
    A=[1.87e-3, 1.12e-10, 68.74]
    t = ( A[0] * data_size + A[1] * (data_size**2) + A[2] )*(10**-3)
    if debug:
        print(f'NPU copy time:{t}')
    return t
""


####################################### Overhead calculation based on transfers ##############################	
def GC_overhead(data_size):
	if debug:
		print(f'datasize:{data_size}')
	return (mapping_time(data_size)+GPU_copy_time(data_size))
	
def CG_overhead(data_size):
	if debug:
		print(f'datasize:{data_size}')
	return (mapping_time(data_size)+GPU_copy_time(data_size)+unmapping_time(data_size))
	
def CC_overhead(data_size):
	if debug:
		print(f'datasize:{data_size}')
	return (GPU_copy_time(data_size))

def CN_overhead(data_size):
	if debug:
		print(f'datasize:{data_size}')
	return (NPU_copy_time(data_size))

def NC_overhead(data_size):
	if debug:
		print(f'datasize:{data_size}')
	return (NPU_copy_time(data_size))

def GN_overhead(data_size):
	if debug:
		print(f'datasize:{data_size}')
	return (mapping_time(data_size)+GPU_copy_time(data_size)+NPU_copy_time(data_size))

def NG_overhead(data_size):
	if debug:
		print(f'datasize:{data_size}')
	return (NPU_copy_time(data_size)+mapping_time(data_size)+GPU_copy_time(data_size)+unmapping_time(data_size))
##########################################################################################################
# ##################### General Overhead function to call approperiate overhead function #############################
# ovh={'GC':GC_overhead, 'CG':CG_overhead, 'CC':CC_overhead, 'CN':CN_overhead, 'NC': NC_overhead, 'NG': NG_overhead, 'GN':GN_overhead}
# overhead=locals()["myfunction"]()
# overhead=eval("myfunction")()
def overhead(order,data_size):
    fun_name=order+'_overhead'
    return eval(fun_name)(data_size)
""

################### Read Power Monitor to clear previous serial data ###################
def Read_Arduino():
    Power_monitoring = threading.Thread(target=Arduino_read.run,args=('t.txt',))
    Power_monitoring.start()
    time.sleep(5)
    Power_monitoring.do_run = False


######################### set Global parameters #################################################
def set_parameters(_Dir="/home/ehsan/UvA/ARMCL/Rock-Pi/ComputeLibrary_64/build/examples/NPU",
_Graph="Alex",
#_N=8,
#_Graph_ARMCL="/graph_alexnet_n_pipe_npu",
#_img=1,
#_data=1,
#_label=1,
_n=10,
_save=0,
_annotate=0,
_layer_timing=0,
_B_threads=2,
_L_threads=4):
    
    global Dir
    global Graph
    global N
    global Graph_ARMCL
    global img
    global data
    global label
    global n
    global save
    global annotate
    global layer_timing
    global B_threads
    global L_threads
    global Host_Configs
    global Max_chromosome
    global Ref_PE
    global ProfResult
    Dir=_Dir
    Graph=_Graph

    N=_Ns[Graph]
    Graph_ARMCL=_Graphs_ARMCL[Graph]
    img=_imgs[Graph]
    data=_dts[Graph]
    label=_lbls[Graph]
    rr='ab'
    print(f'Command is: {rr}')
    p = subprocess.Popen(rr.split())
    p.communicate()
    while(p.returncode):
        print('ab not successful next try after 10s ...')
        time.sleep(10)
        p = subprocess.Popen(rr.split())
        p.communicate()

    rr=f'PiPush {Dir}/{Graph_ARMCL} test_graph'
    print(f'Command is:{rr}')
    p = subprocess.Popen(rr.split())

    n=_n
    save=_save
    annotate=_annotate
    layer_timing=_layer_timing
    B_threads=_B_threads
    L_threads=_L_threads
    Host_Configs=['LL','LB','BL','BB']
    Max_chromosome=[len(LittleFrequencyTable),len(BigFrequencyTable),len(GPUFrquencyTable),
                    len(Host_Configs),
                    4,4,4,4,
                    N+1,N+1,N+1,N+1]
    Ref_PE='LBGN'
    if UseCachedResult:
        try:
            with open('Profile/ProfileResult.pkl','rb') as f:
                ProfResult=pickle.load(f)
        except:
            print('Could not open Profile/ProfileResult.pkl')
            ProfResult={}
            input('For exit press ctrl+c otherwise let it continue with empty dictionary')
    else:
        ProfResult={}
    #Read_Arduino()
    

""
def Check_Chromosome(chromosome):
    valid=True
    #check gene range
    for i,gene in enumerate(chromosome):
        if gene not in range(0,Max_chromosome[i]):
            valid=False
            print("Error invalid gene value (not in range)")
            print(f'i: {i}, gene:{gene}, range: {range(0,Max_chromosome[i])}')
            return
    
    
    Freqs=chromosome[:3]
    Host_Config=Host_Configs[chromosome[3]]
    _PEs=chromosome[4:8]
    _Parts=chromosome[8:]
    #Check Order:
    if len(_PEs) != len(list(set(_PEs))):
        valid=False
        print("_PEs part is not valid (Repeated Component)\n")
    if sum(_Parts) != N:
        valid=False
        print("Parts are not consistent (sum is not N)\n")
        print(f'N: {N}, _Parts: {_Parts}\n')
    return valid
############################### Decode Chromosome #######################################################
def Decode(_chromosome):
    chromosome=_chromosome.copy()
    if not Check_Chromosome(chromosome):
        print("Check chromosome failed\n")
        return(-1)
    Freqs=chromosome[:3]
    Freqs[0]=Freq_Table['L'][Freqs[0]]
    Freqs[1]=Freq_Table['B'][Freqs[1]]
    Freqs[2]=Freq_Table['G'][Freqs[2]]
    Host_Config=Host_Configs[chromosome[3]]
    _PEs=chromosome[4:8]
    _Parts=chromosome[8:]

    #decode, reference order is LBGN
    Ref_Order='LBGN'
    Order=''
    for i in range(len(_PEs)):
        #Order=Order + Ref_Order[_PEs[i]]*_Parts[i]
        Order=Order + Ref_Order[_PEs[i]]*_Parts[_PEs[i]]
        #Order=Order + Ref_Order[i]*_Parts[i]
    
    return (Order,Host_Config,Freqs)
""
############################# Encode to Chromosome ####################################################
def Encode(Order,Host_config,Freqs):
    #Check:
    if len(Order) != N:
        print("Order is not valid \n")
    
    
    #encode, reference order is LBGN
    Ref_Order='LBGN'
    _PEs=[]
    _Parts=[]
    c=1
    for i in range(len(Order)-1):
        if Order[i]==Order[i+1]:
            c=c+1
        else:
            if Ref_Order.index(Order[i]) in _PEs:
                print("invalid Order\n")
            _PEs.append(Ref_Order.index(Order[i]))
            _Parts.append(c)
            c=1
    _PEs.append(Ref_Order.index(Order[i+1]))
    _Parts.append(c)

    chromosome=[]
    chromosome.append(Freqs)
    chromosome.append(Host_Configs.index(Host_config))
    chromosome.append(_PEs)
    chromosome.append(_Parts)
    return chromosome
""



n_eval=0
def Eval(chromosome):
    print('**********************\n\n*****************')
    #input('Contine?')
    global n_eval
    n_eval=n_eval+1
    print(f'{n_eval} Evaluating: {chromosome}')
    if n_eval % 5 == 0 and No_Board == 0:
        print('Saving Profile Result')
        save_ProfResult()
    Order, Host_Config, Freqs=Decode(chromosome)
    print(f'{Order}-{Freqs}-{Host_Config}')

    '''#Resnet 6-8 and 7-8 NPU has problem 
    if Order[5]==Order[6]==Order[7]=='N' and Order[4]!='N' and Order[8]!='N':
        return ({'Latency':10000,'FPS':10000,'Power':10000})
    if Order[6]==Order[7]=='N' and Order[5]!='N' and Order[8]!='N':
        return ({'Latency':10000,'FPS':0.1,'Power':100000})
    if Order[10]==Order[11]=='N' and Order[9]!='N' and Order[12]!='N':
        return ({'Latency':10000,'FPS':0.1,'Power':100000})
    if Order[2]==Order[3]=='N' and Order[1]!='N' and Order[4]!='N':
        return ({'Latency':10000,'FPS':0.1,'Power':100000})
    if Order[11]==Order[12]=='N' and Order[10]!='N' and Order[13]!='N':
        return ({'Latency':10000,'FPS':0.1,'Power':100000})'''
    ##################################

    return Eval_Experimental(Order,Host_Config, Freqs)



######################### Analytical Evaluation ##############################################
def Eval_Analytical(Order,End_Points,N):
    if debug:
        print(f'Order:{Order}, End Points:{End_Points}, N:{N}')
    Stages=len(Order)
    t=[0]*len(Stages)
    overhead=[0]*len(Stages)
    pwr=[[]]*len(Stages)
    Energy=0
    stage_dict={0:'G',1:'B',2:'L'}
    stage_dict2={0:'L',1:'B',2:'G', 3:'N'}
	
	# read data:
    with open('Data.pkl','rb') as f: data=pickle.load(f)
    if debug:
        print(data)
	

    # iterate over components 
    start_point=1
    for i,c in enumerate(Order):
        if debug:
            print(i,c)
        
        if i==0:
            Start_point=1
            Prev_comp='C'
            s=224
            input_size.append(s*s*3)
        else:
            Prev_comp=Order[i-1]
            Start_point=End_Points[i-1]
            input_size=data[c][i-1][500000]['OutputSize']

        End_point=End_Points[i]

        overhead[i]=overhead(Prev_comp+c,input_size)
        #Add overhead energy
        # Energy = Energy + overhead[i](energy) 

        # set Frequency
        Freq=Freq_Table[c][0]

        # iterate over layers in this component
        for l in range(Start_point, End_point+1):
            layer_time=data[c][l][Freq]['timing']
            layer_power=data[c][l][Freq]['power']
            pwr[i].append(layer_power)
            layer_energy=layer_power*(layer_time)
            Energy+=layer_energy
            t[i]+=layer_time
            if debug:
                print(f'Component:{c}, layer:{i},stage:{i}, power:{layer_power}, time{layer_time}')
		
            
    if debug:	
        print(f'Graph process time:{t}, switch overheads:{overhead}')
    #max_stage_latency=max(t)+max(overhead)
    max_stage_latency=0
    for i in enumerate(Order):
        if max(t)+overhead[i] > max_stage_latency:
            max_stage_latency = max(t)+overhead[i]
        
    FPS=1000.0/max_stage_latency
    Latency=sum(t)+sum(overhead)

    #avg_pwr=sum([(sum(p)/len(p)) for p in pwr])
    #max_pwr=sum([max(p) for p in pwr])

    if debug:
        print(f'Performance is:{FPS}FPS, and {Latency}ms. energy:{Energy}')
	
    return (Energy, FPS, Latency)
""




############################### Parse output file of graph run #################
def Parse_output(k):
    ProfResult.setdefault(k,{})
    with open('./temp.txt') as ff:
        lines=ff.readlines()
    #ProfResultult={}
    t0=0
    t1=0
    t2=0
    tn=0
    f=0
    #L=0
    for l in lines:      
        if "Running Graph with Frequency:" in l:
            f=l.split(':')[-1].strip()
        #if "FPS:" in l:
            #ProfResultult[f]['FPS']=l.split(':')[-1]
        if "Latency:" in l:
            if f !=0 :
                ProfResult[k].setdefault(f,{})
                L=float(l.split(':')[-1].strip())
                ProfResult[k][f]['Latency']=L
                FPS=1000/max(t0,t1,t2,tn)
                ProfResult[k][f]['FPS']=FPS
            t0=t1=t2=tn=0
        if "total0_time:" in l:
            t0=float(l.split(':')[-1].strip())
        if "total1_time:" in l:
            t1=float(l.split(':')[-1].strip())
        if "total2_time:" in l:
            t2=float(l.split(':')[-1].strip())
        if "NPU subgraph: 0 --> Cost:" in l:
            tn=float(l.split(':')[-1].strip())
        
        

    return (L,FPS)


""

############################# Parse power file #################################
def Parse_Power(file_name,key,_f):
    f=open(file_name)
    lines=f.readlines()
    f.close()
    #print(len(lines))
    powers=[]
    pin_last=0
    c=0
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
                print(f'pin value changed to {v}')
                if len(powers):
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
    powers[-1]=sum(powers[-1])/len(powers[-1])
    print(f'powers:{powers}')
    powers=powers[0:-1:2]
    i=0
    print(len(powers))
    print(f'powers(just runnings):{powers}')
    ProfResult[key][_f]['Power']=powers[-1]
    '''
    for freq in ProfResultult[key]:
        print(f'F:{freq}')
        ProfResultult[key][freq]["Power"]=powers[i]
        i=i+1
    '''
    return powers[-1]
""


########################## Run a Config on board ############################
def Run_Graph(ALL_Freqs, run_command, myoutput, blocking=True):
    print(run_command)
    p = subprocess.Popen(run_command.split(),
        #stdout=subprocess.PIPE,
        stdout=myoutput,
        stderr=myoutput, stdin=subprocess.PIPE, text=True)
    time.sleep(8)
    for Freqs in ALL_Freqs:
        #print(f'Freqs: {Freqs}')
        for F in Freqs:
            #print(f'F: {F}')
            p.stdin.write(f'{F}\n')
        p.stdin.flush()
    for v in [0,0,0]:
        p.stdin.write(f'{v}\n')
    p.stdin.flush()
    if blocking:
        p.wait()
""
debug=0
pwr=[]

################################################ Experimental Evaluation ##################################
def Eval_Experimental(Order, Host_Config,Freqs):
    #if debug:
        #print(Order, Host_Config,Freqs)
    f=f'{Freqs[0]},{Freqs[1]},{Freqs[2]}'
    k=Order+'_'+Host_Config
    if k in ProfResult and f in ProfResult[k]:        
        print(f'Already evaluated')
        if 'FPS' in ProfResult[k][f] and 'Power' in ProfResult[k][f] and 'Latency' in ProfResult[k][f]:
            return ProfResult[k][f]
        else:
            print('Previous Evaluation is not complete')
    else:
        if No_Board:
            return -1
    Active_PEs=set([c for c in Order if c != 'N'])
    Idle_PEs=set(c for c in 'LBG').difference(Active_PEs)
    ##print(f'Active PEs: {Active_PEs}, Idle PEs: {Idle_PEs}')
    #Add freq table of active PEs
    '''
    Freq_Table_Order=[Freq_Table[c] for c in Active_PEs]
    ##print(f'Freq table: {Freq_Table_Order}')
    #Add min freq of Idle PEs
    if Idle_PEs:
        Freq_Table_Order.append([Freq_Table[c][0] for c in Idle_PEs])
    '''

    Freq_Table_Order=[]
    for c in 'LBG':
        if c in Active_PEs or c in Host_Config:
            Freq_Table_Order.append(Freq_Table[c])
        else:
            Freq_Table_Order.append([Freq_Table[c][0]])
    ##print(f'Freq table: {Freq_Table_Order}')
    #Generate all Freq combinations
    '''
    ALL_Freqs=list(itertools.product(*Freq_Table_Order))
    ##print(f'All Freqs: {ALL_Freqs}')
    if debug:
        print(len(ALL_Freqs))
        ##print(ALL_Freqs)'''
    #input()
    myoutput = open('./temp.txt','w+')
    
    
    #PiPushtest graph_alexnet_n_pipe_npu test_graph/ CL 0 0 0 1 0 0 100 100 BBNNLLGG 0 2 4 Alex
    run_command=f'PiTest {Graph_ARMCL} test_graph/ CL {img} {data} {label} {n} {save} {annotate} {100} \
{100} {Order} {layer_timing} {B_threads} {L_threads} {Graph} {Host_Config[0]} {Host_Config[1]}'
    #print(f'{run_command}')
    os.makedirs('Power/', exist_ok=True)
    file_name='Power/'+Order+'_'+Host_Config+'.csv'
    Power_monitoring = threading.Thread(target=Arduino_read.run,args=(file_name,))
    #time.sleep(2)
    Power_monitoring.start()
    time.sleep(4)
    #print(f'ALL_Freqs: {ALL_Freqs}')
    #for Freq in ALL_Freqs:
    a=time.time_ns()
    Run_Graph([Freqs], run_command, myoutput, True)
    b=time.time_ns()
    print(f'Evaluation time: {(b-a)/10**9}')
    

    #https://stackoverflow.com/questions/18018033/how-to-stop-a-looping-thread-in-python
    Power_monitoring.do_run = False
    myoutput.close()
    
    ProfResult.setdefault(k,{})
    #print('salam1')
    try:
        _L,_FPS=Parse_output(k)
    except Exception as e:
        print(f'Error in parsing output: {e}')
        rr='ab'
        print(f'Command is: {rr}')
        time.sleep(10)
        p = subprocess.Popen(rr.split())
        p.communicate()
        while(p.returncode):
            print('ab not successful next try after 10s ...')
            time.sleep(10)
            p = subprocess.Popen(rr.split())
            p.communicate()
        time.sleep(10)
        R={'Latency':20000,'FPS':0.002,'Power':200000}
        ProfResult[k].setdefault(f,R)
        return -1

    #print(f'salam2:{_L}-{_FPS}')
    time.sleep(2)
    _P=Parse_Power(file_name,k,_f=f)
    #print(f'slam3:{_P}')
    #input("next one...")
    return (ProfResult[k][f])
    #return(_P,_FPS,_L)

def save_ProfResult():
    #global ProfResult
    print (f'Saving Profile Result with len: {len(ProfResult)}')
    os.makedirs('Profile',exist_ok=True)
    with open('Profile/ProfileResult.pkl','wb') as pp:
        pickle.dump(ProfResult,pp)

if __name__=='__main__':
    set_parameters()
    #BGLN2123 F=342 Host=3=BB
    R=Eval([3,4,2,3,1,2,0,3,2,1,2,3])
    print(R)
    #print(f'Power: {R[0]}, FPS: {R[1]}, Latency:{R[2]}')


""

