# Set the parameters and kernel source code dir
compiler := aarch64-linux-android23-clang++
wdir := /system/
app := DVFS_Delay
data := FreqMeasurements2.csv

# Compiler flags
CFLAGS := -I.. -L.. -static-libstdc++ -lstdc++ -pie -Wl,-rpath,/system/usr/lib64/ -rpath /vendor/lib64/ -lz

# Targets
.PHONY: all clean run

all: $(app)

$(app): DVFS_Delay_perf.cpp ../OpenCL.cpp
	$(compiler) $^ $(CFLAGS) -o $@

run: $(app)
	adb push $< $(wdir)/
	adb shell chmod +x $(wdir)/$(app)
	adb shell /system/$(app); adb pull /system/$(data) Data/

clean:
	rm -f $(app)
