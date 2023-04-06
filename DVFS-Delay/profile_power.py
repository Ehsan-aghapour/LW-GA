import os
import time
import threading
import subprocess
import Arduino_read

def profile(pwr="pwr.csv",tme="temp.txt"):
    os.system('adb shell "echo 0 > /sys/class/gpio/gpio157/value"')
    time.sleep(2)
    Power_monitoring = threading.Thread(target=Arduino_read.run,args=(pwr,))
    Power_monitoring.start()
    run_command=f"adb shell /system/a.out"
    oo=open(tme,'w+')
    p = subprocess.Popen(run_command.split(),stdout=oo,stderr=oo, stdin=subprocess.PIPE, text=True)
    p.wait()
    time.sleep(2)
    Power_monitoring.do_run = False
    oo.close()


############################# Parse power file #################################
def Read_Power(file_name="pwr.csv"):#(graph,file_name,frqss):
    f=open(file_name)
    lines=f.readlines()
    f.close()
    powers=[]
    pin_last=0
    c=0
    tts=[]
    si=0
    sj=0
    for kk,l in enumerate(lines):
        c=c+1
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
                if len(powers)==2:
                    si=kk-1
                if len(powers):
                    tts.append(len(powers[-1]))
                    powers[-1]=sum(powers[-1])/len(powers[-1])
                powers.append([float(values[2].strip())])
                pin_last=v
               
            else: 
                if len(powers):
                    powers[-1].append(float(values[2].strip()))
        except:
            print(f'Error in parse power line {c}')
            
    
   
    tts.append(len(powers[-1]))
    powers[-1]=sum(powers[-1])/len(powers[-1])
   
    print(f'number of intervals:{len(tts)}')
    print(f'number of samples in each interval:{tts}')
    
    
    threshold=abs(powers[2]-powers[1])/2
    
    
    for kk,l in enumerate(lines[si:]):
        values=l.split(',')
        p=float(values[2].strip())
        print(l)
        if abs(p-powers[1]) > threshold:
            print(l)
            sj=si+kk-1
            break
    
    mindelay=float(lines[sj].split(',')[1].strip()) - float(lines[si+1].split(',')[1].strip())
    maxdelay=float(lines[sj+1].split(',')[1].strip()) - float(lines[si].split(',')[1].strip())
            
    print(f'si:{si}, sample:{lines[si]}\n'
          f'sj:{sj}, sample:{lines[sj]}\n'
          f'Power0:{powers[1]}, Power1:{powers[2]}, threshold:{threshold}\n'
          f'Min Delay:{mindelay}, Max Delay:{maxdelay}\n')
        
    return mindelay,maxdelay

mins=[]
maxs=[]
for j in range(100):
    profile()
    time.sleep(1)
    mn,mx=Read_Power()
    mins.append(mn)
    maxs.append(mx)

profile()


