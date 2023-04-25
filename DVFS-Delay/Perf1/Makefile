#!/bin/bash

## Set the parameters and kernel source code dir
#target=armv7a-linux-androideabi$1-clang++
compiler=aarch64-linux-android23-clang++
wdir="/system/"
app="DVFS_Delay"
data="FreqMeasurements.csv"


default:
	${compiler} DVFS_Delay_perf.cpp ../OpenCL.cpp -static-libstdc++ -lstdc++ -I.. -L.. -pie -Wl,-rpath,/system/usr/lib64/ -rpath /vendor/lib64/ -lz -o ${app}

power:
	${compiler} DVFS_Delay_power.cpp OpenCL.cpp -static-libstdc++ -lstdc++ -I. -L. -pie -Wl,-rpath,/system/usr/lib64/ -rpath /vendor/lib64/ -lz -o ${app}
	

run:
	adb push ${app} ${wdir}/
	adb shell chmod +x ${wdir}/${app}
	adb shell /system/${app}; adb pull /system/${data} Data/

all: default run

clean:
	rm -f *.o ${app}
