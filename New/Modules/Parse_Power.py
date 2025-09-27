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
    nn=((2*NL*Num_frames)+2)
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
                
            if layer==0:
                pwr_df.loc[len(pwr_df)]={"Graph":graph, "Component":order[layer],"Freq":freq[layer][0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"in","Power":input_pwrs}
                print(f'setting power for {graph}-{order}-{layer}-{freq[layer][0]}-in-power-->{input_pwrs}')
            
            task_pwrs[layer]=pwrs[j::2*NL]
            print(f'len layer power {len(task_pwrs[layer])}')
            task_pwrs[layer]=sum(task_pwrs[layer])/len(task_pwrs[layer])
            pwr_df.loc[len(pwr_df)]={"Graph":graph, "Component":order[layer],"Freq":freq[layer][0],"Freq_Host":Host_freq, "Layer":layer,"Metric":"task","Power":task_pwrs[layer]}
            print(f'setting power for {graph}-{order}-{layer}-{freq[layer][0]}-task-power->{task_pwrs[layer]}')
    return pwr_df





def Parse_Power_total(file_name,graph,order,frqss):
    global tts,powers
    powers,tts=Read_Power(file_name)
    power_df = pd.DataFrame(columns=['graph', 'order', 'freq', 'input_power','task_power'])
    NL=1
    nn=((2*NL*Num_frames)+2)
    nnn=nn*len(frqss)
    if len(powers)!=nnn:
        print(f"bad power size: {len(powers)}")
        print(f'Expected size is:NFreqx((2xNLxn)+2) which is {len(frqss)}x((2x{NL}x{Num_frames})+2)={nnn}')
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