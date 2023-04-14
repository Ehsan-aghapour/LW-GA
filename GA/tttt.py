_Order=[]
for C in Order:
    if Parts[C] != 0:
        _Order.append(C)


for C in range(0,4):
    if C not in _Order:
        _Order.append(C)


Devices={0:{0:[2,3],1:[2],2:{3},3:[]},
         1:{0:[],1:[3],2:[2],3:[2,3]}}

HostConfig=X[3]
host=False
for D in Devices[0][HostConfig]:
    if Parts[D] != 0:
        host=True
if not host:
    X[0]=0

host=False
for D in Devices[1][HostConfig]:
    if Parts[D] != 0:
        host=True

if not host:
    X[1]=0