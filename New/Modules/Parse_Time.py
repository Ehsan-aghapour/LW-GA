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
    if df_time.shape[0] != len(frqss):
        print(f'Parse performance error: number of runs {df_time.shape[0]} is not equals to number of freqs {len(frqss)}')
        input()
    return df_time