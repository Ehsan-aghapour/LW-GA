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
        Profile(frqss,Num_frames,order,graph,pwrfile,timefile,caching=True)
        #time.sleep(10)
        time_df=Parse(timefile,graph,order,frqss)
        power_df=Parse_Power(pwrfile,graph,order,frqss)
        #time_df['Freq'] = time_df['Freq'].apply(lambda x: tuple(x))
        #power_df['Freq'] = power_df['Freq'].apply(lambda x: tuple(x))
        merged_df = pd.merge(power_df, time_df, on=['Graph', 'Component', 'Freq','Freq_Host','Layer','Metric'])
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