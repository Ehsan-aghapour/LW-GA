import MY_Profile as m
import sys

LittleFrequencyTable = [408000, 600000, 816000, 1008000, 1200000, 1416000]
#BigFrequencyTable=[500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000, 1908000, 2016000, 2100000, 2208000]
BigFrequencyTable = [408000, 600000, 816000, 1008000, 1200000, 1416000, 1608000, 1800000]
GPUFrequencyTable = [200000000, 300000000, 400000000, 600000000, 800000000]
Freq_Table={'L':LittleFrequencyTable, 'B':BigFrequencyTable, 'G':GPUFrequencyTable}
##################################################################


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

def Compare(Graph=_Graphs[0],max=True):
    m.set_parameters(_Graph=Graph)
    N=_Ns[Graph]
    #chromosome=[0] * 12
    chromosome=[0,0,0,0,0,1,2,3,N,0,0,0]
    if max:
        chromosome[0]=(len(LittleFrequencyTable)-1)
        chromosome[1]=(len(BigFrequencyTable)-1)
        chromosome[2]=(len(GPUFrequencyTable)-1)
    Parts=[0,0,0,0]
    f=open(Graph+'_Comparison.csv','a')
    for i in range(len(Parts)):
        _Parts=Parts.copy()
        _Parts[i]=N
        chromosome[8:12]=_Parts
        _Hosts=[0]
        if _Parts[2]:
            _Hosts.append(2)
        if _Parts[3]:
            _Hosts.append(1)
            #_Hosts.append(3)
        for Host in _Hosts:
            chromosome[3]=Host
            print(f'Evaluating chromosome: {chromosome}')
            R=m.Eval(chromosome)
            print(f'Result is: {R}')
            f.write('['+''.join(str(x) for x in chromosome)+']')
            Order,Host_Config,Freqs=m.Decode(chromosome)
            f.write(','+Order)
            f.write(','+Host_Config)
            f.write(',[ ' + str(Freqs[0]/1000) + '-' + str(Freqs[1]/1000) + '-' + str(Freqs[2]/1000000) + ' ]')
            f.write(','+str(R['Latency']))
            f.write(','+str(R['FPS']))
            f.write(','+str(R['Power']))
            f.write('\n')
    f.close()


if __name__=='__main__':
    g=0
    if len(sys.argv) > 1 :
        g=int(sys.argv[1])
    Compare(_Graphs[g])
    Compare(_Graphs[g],max=False)

    
    


