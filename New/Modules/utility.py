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
    if tfc > _dvfs_delay:
        t=tfn - (_dvfs_delay/tfc)*tfn + _dvfs_delay  
    if debug:
        print(f'in:{0}, next_freq:{fn[0]} time(next_freq):{tfn} cur_freq:{fc[0]} time(cur_freq):{tfc} time:{t}')      
    tt+=t
    tt_nodvfs+=tfn
    
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