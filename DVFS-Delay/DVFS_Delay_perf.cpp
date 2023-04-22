#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/cl2.hpp>
#include <dlfcn.h>
#include "OpenCL.h"
#include <chrono>
#include <thread>
#include <ctime>

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <thread>


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

std::chrono::microseconds measure_time(int num_iterations, std::vector<int> cpus, int LittleFreq, int BigFreq, int GPUFreq){
    double result=0;
    cpu_set_t set;
    CPU_ZERO(&set);
    for(int i=0;i<cpus.size();i++){
        CPU_SET(cpus[i],&set);
    }
    sched_setaffinity(0, sizeof(set), &set);
    dvfs.commit_freq(LittleFreq, BigFreq, GPUFreq);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<num_iterations;i++){
        result += std::sqrt(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cerr << "Time taken: " << duration.count() << " microseconds"<< std::endl;
    return duration;
}


void initialise_context_properties(const cl::Platform &platform, const cl::Device &device, std::array<cl_context_properties, 7> &prop)
{
#if defined(ARM_COMPUTE_ASSERTS_ENABLED)
    // Query devices in the context for cl_arm_printf support
    if(arm_compute::device_supports_extension(device, "cl_arm_printf"))
    {
        // Create a cl_context with a printf_callback and user specified buffer size.
        std::array<cl_context_properties, 7> properties_printf =
        {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()),
            // Enable a printf callback function for this context.
            CL_PRINTF_CALLBACK_ARM, reinterpret_cast<cl_context_properties>(printf_callback),
            // Request a minimum printf buffer size of 4MB for devices in the
            // context that support this extension.
            CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
            0
        };
        prop = properties_printf;
    }
    else
#endif // defined(ARM_COMPUTE_ASSERTS_ENABLED)
    {
        std::array<cl_context_properties, 3> properties =
        {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()),
            0
        };
        std::copy(properties.begin(), properties.end(), prop.begin());
    };
}



std::tuple<cl::Context, cl::Device, cl_int>
create_opencl_context_and_device()
{
    if(!arm_compute::opencl_is_available()){
        std::cerr<<"OpenCL is not available\n";
    }
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.size() == 0)
        std::cerr<<"Couldn't find any OpenCL platform\n";
    cl::Platform            p = platforms[0];
    cl::Device              device;
    std::vector<cl::Device> platform_devices;
    p.getDevices(CL_DEVICE_TYPE_DEFAULT, &platform_devices);
    if(platform_devices.size() == 0)
        std::cerr<<"Couldn't find any OpenCL device\n";
    device     = platform_devices[0];
    cl_int err = CL_SUCCESS;
    std::array<cl_context_properties, 7> properties = { 0, 0, 0, 0, 0, 0, 0 };
    initialise_context_properties(p, device, properties);
    cl::Context cl_context = cl::Context(device, properties.data(), nullptr, nullptr, &err);
    if(err != CL_SUCCESS)
        std::cerr<<"Failed to create OpenCL context\n";
    return std::make_tuple(cl_context, device, err);
}
// Function to generate random float values between 0 and 1
float random_float() {
    return static_cast<float>(rand()) / RAND_MAX;
}
   

    
    
    
std::chrono::microseconds measure_time_GPU(int vectorSize, std::vector<int> cpus, int LittleFreq, int BigFreq, int GPUFreq, cl::CommandQueue queue, cl::Context context ){

    cpu_set_t set;
    CPU_ZERO(&set);
    for(int i=0;i<cpus.size();i++){
        CPU_SET(cpus[i],&set);
    }
    sched_setaffinity(0, sizeof(set), &set);
    dvfs.commit_freq(LittleFreq, BigFreq, GPUFreq);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::vector<float> a(vectorSize);
    std::vector<float> b(vectorSize);
    std::vector<float> c(vectorSize);
    
    // Initialize input vectors with random values
    for (int i = 0; i < vectorSize; ++i) {
        a[i] = random_float();
        b[i] = random_float();
    }

    // Create OpenCL buffers
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeof(float) * vectorSize);
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY, sizeof(float) * vectorSize);
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * vectorSize);

    // Copy input vectors to device
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * vectorSize, a.data());
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * vectorSize, b.data());

    // Load and compile the OpenCL kernel
    std::string kernelSource = R"(
        __kernel void vector_addition(__global float* a, __global float* b, __global float* c) {
            const int gid = get_global_id(0);
            c[gid] = a[gid] + b[gid];
        }
    )";
    cl::Program program(context, kernelSource);
    program.build();

    // Create OpenCL kernel
    cl::Kernel kernel(program, "vector_addition");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);


    
    
    // Run the OpenCL kernel
    cl::Event event;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto start=std::chrono::high_resolution_clock::now();
    //cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSize), cl::NullRange, nullptr, &event);
    // Wait for the kernel to complete
    event.wait();
    auto end=std::chrono::high_resolution_clock::now();
    //cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    
    // Copy result back to host
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * vectorSize, c.data());    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //double elapsed = static_cast<double>(endTime - startTime) * 1e-9;
    std::cerr << "Time taken: " << duration.count() << " microseconds"<< std::endl;

    return duration;
}

std::pair<double, double> measure_time_combination(int num_iterations, std::vector<int> cpus, int LittleFreq, int BigFreq, int GPUFreq, int LittleFreqNew, int BigFreqNew, int GPUFreqNew){
    double result=0;
    cpu_set_t set;
    CPU_ZERO(&set);
    for(int i=0;i<cpus.size();i++){
        CPU_SET(cpus[i],&set);
    }
    sched_setaffinity(0, sizeof(set), &set);
    dvfs.commit_freq(LittleFreq, BigFreq, GPUFreq);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<num_iterations;i++){
        result += std::sqrt(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    dvfs.commit_freq(LittleFreqNew, BigFreqNew, GPUFreqNew);
    for(int i=0;i<num_iterations;i++){
        result += std::sqrt(i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start);
    std::cerr << "Time taken: " << duration2.count() << " microseconds\n"<<"first: "<<duration.count() <<"   second: "<< duration2.count()-duration.count()<< std::endl;
    return std::make_pair(duration.count(),duration2.count());
}

std::pair<double, double> measure_time_combination_GPU(int vectorSize, std::vector<int> cpus, int LittleFreq, int BigFreq, int GPUFreq, int LittleFreqNew, int BigFreqNew, int GPUFreqNew, cl::CommandQueue queue, cl::Context context){
    double result=0;
    cpu_set_t set;
    CPU_ZERO(&set);
    for(int i=0;i<cpus.size();i++){
        CPU_SET(cpus[i],&set);
    }
    sched_setaffinity(0, sizeof(set), &set);
    dvfs.commit_freq(LittleFreq, BigFreq, GPUFreq);
    

    std::vector<float> a(vectorSize);
    std::vector<float> b(vectorSize);
    std::vector<float> c(vectorSize);
    
    // Initialize input vectors with random values
    for (int i = 0; i < vectorSize; ++i) {
        a[i] = random_float();
        b[i] = random_float();
    }

    // Create OpenCL buffers
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeof(float) * vectorSize);
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY, sizeof(float) * vectorSize);
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * vectorSize);

    // Copy input vectors to device
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * vectorSize, a.data());
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * vectorSize, b.data());

    // Load and compile the OpenCL kernel
    std::string kernelSource = R"(
        __kernel void vector_addition(__global float* a, __global float* b, __global float* c) {
            const int gid = get_global_id(0);
            c[gid] = a[gid] + b[gid];
        }
    )";
    cl::Program program(context, kernelSource);
    program.build();

    // Create OpenCL kernel
    cl::Kernel kernel(program, "vector_addition");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);


    
    
    // Run the OpenCL kernel
    cl::Event event;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    auto start=std::chrono::high_resolution_clock::now();
    //cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSize), cl::NullRange, nullptr, &event);
    // Wait for the kernel to complete
    event.wait();
    auto end=std::chrono::high_resolution_clock::now();

    //Second part
    dvfs.commit_freq(LittleFreqNew, BigFreqNew, GPUFreqNew);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSize), cl::NullRange, nullptr, &event);
    // Wait for the kernel to complete
    event.wait();
    auto end2=std::chrono::high_resolution_clock::now();
    
    // Copy result back to host
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * vectorSize, c.data());    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start);
    //double elapsed = static_cast<double>(endTime - startTime) * 1e-9;
    std::cerr << "Time taken: " << duration2.count() << " microseconds\n"<<"first: "<<duration.count() <<"   second: "<< duration2.count()-duration.count()<<std::endl;

    return std::make_pair(duration.count(),duration2.count());
}


struct Key {
    std::string PE;
    int Freq;
    int NextFreq;
};

bool operator==(const Key& lhs, const Key& rhs) {
    return lhs.PE == rhs.PE && lhs.Freq == rhs.Freq && lhs.NextFreq == rhs.NextFreq;
}

// Hash function for Key to be used in unordered_map
struct KeyHash {
    std::size_t operator()(const Key& key) const {
        std::size_t h1 = std::hash<std::string>{}(key.PE);
        std::size_t h2 = std::hash<int>{}(key.Freq);
        std::size_t h3 = std::hash<int>{}(key.NextFreq);
        return h1 ^ h2 ^ h3 ;
    }
};
std::unordered_map<Key, std::pair<double,double>, KeyHash> FreqMeasurements;
std::string FreqMeasurementFile="/system/FreqMeasurements.csv";
const double target_ms=1000;
int NumLittleFreqs=6;
int NumBigFreqs=8;
int NumGPUFreqs=5;
std::vector<int> littlecpus={0,1,2,3};
std::vector<int> bigcpus={4,5};
std::vector<int> gpucpus={5};
int relaxtime=4000; //ms

int Load_FreqMeasurements(){
    std::ifstream file(FreqMeasurementFile); // Replace "data.csv" with your actual CSV file name
    if (!file.is_open()) {
        std::cout << "Failed to open file" << std::endl;
        return 1;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        int num_iterations;
        std::string PE;
        int Freq;
        double timeFreq1;
        double timeTotal;
        if (iss >> num_iterations >> PE >> Freq >> timeFreq1 >> timeTotal) {
            // Create a key based on graph, layer, and component
            Key key{PE, Freq};
            FreqMeasurements[key] = std::make_pair(timeFreq1,timeTotal);
        } else {
            std::cout << "Failed to parse line: " << line << std::endl;
        }
    }
    return 0;
}

void write_to_FreqMeasurements(){
    // Open file for writing
    std::ofstream file(FreqMeasurementFile);
    // Write header row
    file << "Num_iterations,PE,Freq,NextFreq,TimeFreq1,TimeTotal\n";
    // Write data rows
    for (const auto& kv : FreqMeasurements) {
        file << num_iterations <<","<< kv.first.PE << "," << kv.first.Freq << "," <<kv.first.NextFreq<<","<< kv.second.first<<","<<kv.second.second << "\n";
    }
    // Close file
    file.close();
}

void Measure_Freq_times(int num_iterations, cl::CommandQueue queue, cl::Context cntx){
    //Measure Little CPU freqs
    Key key{"Little", 0, 0};    
    std::cerr<<"Measure Little Freqs:\n";
    for(int i=0;i<NumLittleFreqs;i++){
        key.NextFreq=i;
        if (FreqMeasurements.count(key) > 0){
            std::cout<<"Already evaluated:"<<FreqMeasurements[key].first<<","<<FreqMeasurements[key].second<<std::endl;
            continue;
        }
        auto duraton=measure_time(num_iterations, littlecpus, i, 0, 0);
        FreqMeasurements[key] = std::make_pair(duraton.count(),0);
        std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
    }

    //Measure Big CPU freqs
    key.PE="Big";  
    std::cerr<<"Measure Big Freqs:\n";
    for(int i=0;i<NumBigFreqs;i++){
        key.NextFreq=i;
        if (FreqMeasurements.count(key) > 0){
            std::cout<<"Already evaluated:"<<FreqMeasurements[key].first<<","<<FreqMeasurements[key].second<<std::endl;
            continue;
        }
        auto duraton=measure_time(num_iterations, bigcpus, 0, i, 0);
        FreqMeasurements[key] = std::make_pair(duraton.count(),0);
        std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
    }

    //Measure GPU freqs
    key.PE="GPU";
    std::cerr<<"Measure GPU Freqs:\n";
    for(int i=0;i<NumGPUFreqs;i++){
        key.NextFreq=i;
        if (FreqMeasurements.count(key) > 0){
            std::cout<<"Already evaluated:"<<FreqMeasurements[key].first<<","<<FreqMeasurements[key].second<<std::endl;
            continue;
        }
        auto duraton=measure_time_GPU(num_iterations, gpucpus, 0, 0, i, queue, cntx);
        FreqMeasurements[key] = std::make_pair(duraton.count(),0);
        std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
    }
    write_to_FreqMeasurements();
}


void Measure_Two_Freqs_times(int num_iterations, cl::CommandQueue queue, cl::Context cntx){
    //Measure Little CPU freqs
    Key key{"Little", 0, 0};    
    std::cerr<<"Measure Little Freqs:\n";
    for(int j=0;j<NumLittleFreqs;j++){
        key.Freq=j;
        std::cerr<<"Cur Freq:"<<j<<std::endl;
        for(int i=0;i<NumLittleFreqs;i++){
            key.NextFreq=i;
            if (FreqMeasurements.count(key) > 0){
                std::cout<<"Already evaluated:"<<FreqMeasurements[key].first<<","<<FreqMeasurements[key].second<<std::endl;
                continue;
            }
            std::pair<double,double> durations=measure_time_combination(num_iterations, littlecpus, j, 0, 0, i, 0, 0);
            FreqMeasurements[key] = durations;
            std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
        }
    }
    write_to_FreqMeasurements();
    //Measure Big CPU freqs
    key.PE="Big";  
    std::cerr<<"Measure Big Freqs:\n";
    for(int j=0;j<NumBigFreqs;j++){
        key.Freq=j;
        std::cerr<<"Cur Freq:"<<j<<std::endl;
        for(int i=0;i<NumBigFreqs;i++){
            key.NextFreq=i;
            if (FreqMeasurements.count(key) > 0){
                std::cout<<"Already evaluated:"<<FreqMeasurements[key].first<<","<<FreqMeasurements[key].second<<std::endl;
                continue;
            }
            std::pair<double,double> durations=measure_time_combination(num_iterations, littlecpus, j, 0, 0, i, 0, 0);
            FreqMeasurements[key] = durations;
            std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
        }
    }   
    write_to_FreqMeasurements();
    //Measure GPU freqs
    key.PE="GPU";
    std::cerr<<"Measure GPU Freqs:\n";
    for(int j=0;j<NumGPUFreqs;j++){
        key.Freq=j;
        std::cerr<<"Cur Freq:"<<j<<std::endl;
        for(int i=0;i<NumGPUFreqs;i++){
            key.NextFreq=i;
            if (FreqMeasurements.count(key) > 0){
                std::cout<<"Already evaluated:"<<FreqMeasurements[key].first<<","<<FreqMeasurements[key].second<<std::endl;
                continue;
            }
            std::pair<double,double> durations=measure_time_combination(num_iterations, littlecpus, j, 0, 0, i, 0, 0);
            FreqMeasurements[key] = durations;
            std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
        }
    }
    write_to_FreqMeasurements();

 
}

int main(){

    init();
    Load_FreqMeasurements();
    dvfs.commit_freq(0, 0, 0);
    
    int num_iterations = calc_num_iterations(target_ms);
    
    cl::Context context;
    cl::Device  gpuDevice;
    cl_int      err;
    std::tie(context, gpuDevice, err) = create_opencl_context_and_device();
    if(err != CL_SUCCESS)
        std::cerr<<"Failed to create OpenCL context\n";
    cl::CommandQueue queue = cl::CommandQueue(context, gpuDevice,CL_QUEUE_PROFILING_ENABLE);

    //Measure_Freq_times(num_iterations, queue, context);
    Measure_Two_Freqs_times(num_iterations, queue, context);
    


    return 0;
}
