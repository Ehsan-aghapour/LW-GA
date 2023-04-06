#include <iostream>
#include <chrono>
#include <cmath>

#include "Power.h"
#include "DVFS.h"

class DVFS dvfs;
/*
int init_rockpi(){
    std::string CPU_path="/sys/devices/system/cpu/cpufreq/";
    std::string GPU_path="/sys/class/devfreq/ff9a0000.gpu/";
    std::string IOCTL_path="/dev/pandoon_device";
    std::string Command="chmod 666 " + IOCTL_path;
    Command="echo pandoon > " + CPU_path + "policy4/scaling_governor";
    system(Command.c_str());
    Command="echo pandoon > " + CPU_path + "policy0/scaling_governor";
    system(Command.c_str());
    Command="echo pandoon > " + GPU_path + "/governor";
    system(Command.c_str());

    Command="cat " + CPU_path + "policy4/scaling_governor";
    system(Command.c_str());
    Command="cat " + CPU_path + "policy4/scaling_governor";
    system(Command.c_str());
    Command="cat " + GPU_path + "/governor";
    system(Command.c_str());
    return 0;
}

int open_pandoon(){
    //int fddd=-1;
    fd = open("/dev/pandoon_device", O_RDWR);
    std::cerr<<"Pandoon opened at "<<fd<<std::endl;
    if (fd<0){
        printf("DRPM_Ehsan: Pandoon Not Opened. ERROR CODE:%d, ERROR MEANING:%s\n",errno,strerror(errno));
        close(fd);
        return -1;
    }
    return fd;
}*/

int init_GPIO(){
    if (-1 == GPIOExport(POUT))
        return(1);
    if (-1 == GPIODirection(POUT, OUT))
        return(2);
    if (-1 == GPIOWrite(POUT, 0))
        std::cerr<<"Could not write 0 to GPIO\n";
        return (3);
    return 0;
}

int init(){
    int ret=0;
    //ret=init_rockpi();
    //ret=open_pandoon();
    dvfs.init();
    ret=init_GPIO();
    return ret;
}

int calc_num_iterations(const double target_duration_ms=1000){
    int num_iterations = 10000000;    // initial number of iterations
    double result = 0;

    // Measure the execution time of the loop
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        result += std::sqrt(i);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Adjust the number of iterations based on the measured duration
    while (duration_ms < target_duration_ms) {
        num_iterations *= 2;
        result = 0;
        start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            result += std::sqrt(i);
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    }

    //std::cout << "Result: " << result << std::endl;
    std::cout << "Number of iterations: " << num_iterations << std::endl;
    std::cout << "Duration: " << duration_ms << " ms" << std::endl;
    return num_iterations;
}

int main(){
    init();
    dvfs.commit_freq(0, 0, 0);
    const double target_ms=1000;
    const int num_iterations = calc_num_iterations(target_ms);
    double result = 0;
    auto start = std::chrono::high_resolution_clock::now();

    // target_ms with minimum freq (for triggering the profile)
    if (-1 == GPIOWrite(POUT, 1)){
        std::cerr<<"Could not write to GPIO\n";
    }
    for (int i = 0; i < num_iterations; ++i) {
        result += std::sqrt(i);
    }

    // target_ms with minimum freq 
    if (-1 == GPIOWrite(POUT, 0)){
        std::cerr<<"Could not write to GPIO\n";
    }
    for (int i = 0; i < num_iterations; ++i) {
        result += std::sqrt(i);
    }

    auto start2 = std::chrono::high_resolution_clock::now();
    // target_ms with min freq
    for(int i=0;i<100;i++){
        result += std::sqrt(i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    dvfs.commit_freq(0, 7, 0);
    start2 = std::chrono::high_resolution_clock::now();
    for(int i=0;i<100;i++){
        result += std::sqrt(i);
    }
    end2 = std::chrono::high_resolution_clock::now();



    
    if (-1 == GPIOWrite(POUT, 1)){
        std::cerr<<"Could not write to GPIO\n";
    }
    for (int i = 0; i < num_iterations; ++i) {
        result += std::sqrt(i);
    }

    // target_ms with minimum freq
    dvfs.commit_freq(0, 0, 0);
    if (-1 == GPIOWrite(POUT, 0)){
        std::cerr<<"Could not write to GPIO\n";
    }
    for (int i = 0; i < num_iterations; ++i) {
        result += std::sqrt(i);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Result: " << result << std::endl;
    std::cerr << "Time taken: " << duration.count() << " microseconds" << std::endl;
    
    return 0;
}
