#!/bin/bash

## Set the parameters and kernel source code dir
#target=armv7a-linux-androideabi$1-clang++
compiler=aarch64-linux-android23-clang++
wdir="/system/"
app="CPUUtil"

default:
	${compiler} main.cpp -static-libstdc++ -lstdc++ -I. -L. -pie -Wl,-rpath,/system/usr/lib64/ -rpath /vendor/lib64/ -lz -o ${app}
	

run:
	adb push ${app} ${wdir}/
	adb shell chmod +x ${wdir}/${app}
	adb shell /system/${app}

all: default run

clean:
	rm -f *.o ${app}
