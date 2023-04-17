#!/bin/bash

## Set the parameters and kernel source code dir
#target=armv7a-linux-androideabi$1-clang++
compiler=aarch64-linux-android23-clang++
wdir="/system/"
app="GPUUtil"

default:
	${compiler} main.cpp OpenCL.cpp -I.  -pie -Wl,-rpath,/system/usr/lib64/ -lz  -o ${app}
	

run:
	adb push ${app} ${wdir}/
	adb shell chmod +x ${wdir}/${app}
	adb shell /system/${app}

all: default run

clean:
	rm -f *.o ${app}
