// +
#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/cl2.hpp>
#include <dlfcn.h>
#include "../OpenCL.h"
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
#include <numeric>
#include <random>


// -

#include "../Power.h"
#include "../DVFS.h"

#define DL0 0
#define DL1 1
#define DL2 2
#define Debug DL1

class DVFS dvfs;
std::string FreqMeasurementFile="/system/FreqMeasurements2.csv";
const double target_ns_CPU=20000;
const double target_ns_GPU=20000;
//20000000/target_ns_CPU To run about 20ms(==20000000ns)
const int N_loops_CPU=2000;
const int N_loops_GPU=500;
int vectorSizeGPU=64;
int num_iterations_CPU=0;
int num_iterations_GPU=0;
int NumLittleFreqs=6;
int NumBigFreqs=8;
int NumGPUFreqs=5;
std::vector<int> littlecpus={3};
std::vector<int> bigcpus={5};
std::vector<int> gpucpus={5};
int relaxtime=8000; //ms
int relaxtime_0=1000; //ms
int N_runs=10;

struct Key {
    int Num_iterations;
    std::string PE;
    int Freq;
    int NextFreq;
};

bool operator==(const Key& lhs, const Key& rhs) {
    return lhs.PE == rhs.PE && lhs.Freq == rhs.Freq && lhs.NextFreq == rhs.NextFreq && lhs.Num_iterations == rhs.Num_iterations;
}

// Hash function for Key to be used in unordered_map
struct KeyHash {
    std::size_t operator()(const Key& key) const {
        std::size_t h1 = std::hash<std::string>{}(key.PE);
        std::size_t h2 = std::hash<int>{}(key.Freq);
        std::size_t h3 = std::hash<int>{}(key.NextFreq);
        std::size_t h4 = std::hash<int>{}(key.Num_iterations);
        return h1 ^ h2 ^ h3 ^ h4 ;
    }
};
std::unordered_map<Key, std::vector<double>, KeyHash> FreqMeasurements;

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
        int NextFreq;
        double t;
        if (iss >> num_iterations >> PE >> Freq>> NextFreq) {
            // Create a key based on graph, layer, and component
            Key key{num_iterations, PE, Freq, NextFreq};
            while(iss >> t){
                FreqMeasurements[key].push_back(t);
            }
            if(FreqMeasurements[key].size()!=N_runs+1){
                std::cout<<"Number of read times is not equal to "<<N_runs<<" for line: "<<line;
            }
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
    file << "Num_iterations,PE,Freq,NextFreq";
    for(int j=0;j<N_runs;j++){
        file<<","<<"T"<<j+1;
    }
    file<<",AVG\n";
    // Write data rows
    for (const auto& kv : FreqMeasurements) {
        file << kv.first.Num_iterations <<","<< kv.first.PE << "," << kv.first.Freq << "," <<kv.first.NextFreq;
        if (kv.second.size()!=N_runs+1){
            std::cout<<"Error, number of elements "<<kv.second.size()-1<<" is not euqal to "<<N_runs<<std::endl;
        }
        for(int j=0;j<kv.second.size();j++){
            file<<","<< kv.second[j];
        }
        file<<"\n";
    }
    // Close file
    file.close();
}

//Functions for initializing GPU
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


//Function for init GPIO
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

//Call this init at start of the program
//to init dvfs and GPIO
int init(){
    int ret=0;
    //ret=init_rockpi();
    //ret=open_pandoon();
    dvfs.init();
    ret=init_GPIO();
    return ret;
}

//Calculate number of iterations for target ns 
int calc_num_iterations(const double target_duration_ns=10000){
    int num_iterations = 1;    // initial number of iterations
    double result = 0;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(3,&set);
    sched_setaffinity(0, sizeof(set), &set);
    dvfs.commit_freq(0, 0, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    // Measure the execution time of the loop
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        result += std::sqrt(i);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    // Adjust the number of iterations based on the measured duration
    while (duration_ns < target_duration_ns) {
        num_iterations *= 2;
        result = 0;
        start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            result += std::sqrt(i);
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    }

    //std::cout << "Result: " << result << std::endl;
    std::cout << "Number of iterations: " << num_iterations << std::endl;
    std::cout << "Duration: " << duration_ns << " ns" << std::endl;
    return num_iterations;
}

//Calculate number of iterations for target ns 
int calc_num_iterations_GPU(const double target_duration_ns, cl::CommandQueue queue, cl::Context context){
    //dvfs.commit_freq(LittleFreq, BigFreq, GPUFreq);
    int num_iterations = 1;    // initial number of iterations
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(5,&set);
    sched_setaffinity(0, sizeof(set), &set);
    dvfs.commit_freq(0, 0, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::vector<float> a(vectorSizeGPU);
    std::vector<float> b(vectorSizeGPU);
    std::vector<float> c(vectorSizeGPU);
    
    // Initialize input vectors with random values
    for (int i = 0; i < vectorSizeGPU; ++i) {
        a[i] = random_float();
        b[i] = random_float();
    }

    // Create OpenCL buffers
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeof(float) * vectorSizeGPU);
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY, sizeof(float) * vectorSizeGPU);
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * vectorSizeGPU);

    // Copy input vectors to device
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * vectorSizeGPU, a.data());
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * vectorSizeGPU, b.data());

    
    int nn=16;
    std::vector<cl::Kernel> kernels;
    std::vector<cl::Event> events(nn);
    
    for(int i = 0; i < nn; i++){
        //cl::Event event;
        //events[i]=std::move(event);
        events[i]=cl::Event();
    }
    
    auto first_event = clCreateUserEvent(context.get(), NULL);
    if(first_event==nullptr){
        std::cerr<<"user event null\n";
    }
    std::vector<cl::Event> first_events;
    cl::Event first_event_obj(first_event);
    first_events.push_back(first_event_obj);
    num_iterations=1;
    for(int i=0;i<nn;i++){
        // Load and compile the OpenCL kernel
        std::string kernelSource = R"(
            __kernel void vector_addition(__global float* a, __global float* b, __global float* c, int k) {
                    const int gid = get_global_id(0);
                    for (int i = 0; i < k; ++i) {
                        c[gid] += a[gid] * b[gid] * (i % 7);
                    }
                }
            )";
            
        cl::Program program(context, kernelSource);
        program.build();
        // Create OpenCL kernel
        cl::Kernel kernel(program, ("vector_addition"));
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, sizeof(int), &num_iterations);
        kernels.push_back(std::move(kernel));
        //std::this_thread::sleep_for(std::chrono::milliseconds(100));
        num_iterations=num_iterations*2;
    }
    std::vector<double> durations(nn);
    std::vector<double> durations2(nn);
    queue.enqueueNDRangeKernel(kernels[0], cl::NullRange, cl::NDRange(vectorSizeGPU), cl::NullRange, &first_events, &events[0]);
    for(int i=1;i<nn;i++){
        queue.enqueueNDRangeKernel(kernels[i], cl::NullRange, cl::NDRange(vectorSizeGPU), cl::NullRange, nullptr, &events[i]);                                      
    }
    
    //queue.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    clSetUserEventStatus(first_event, CL_COMPLETE);
    for(int i=0;i<nn;i++){
        auto start=std::chrono::high_resolution_clock::now();
        // Wait for the kernel to complete
        events[i].wait();
        auto end=std::chrono::high_resolution_clock::now();
        durations[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        cl_ulong start_time = events[i].getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end_time = events[i].getProfilingInfo<CL_PROFILING_COMMAND_END>();
        durations2[i] = static_cast<double>(end_time - start_time);
        
    }
#if Debug==DL1
    for(int i=0;i<nn;i++){
        std::cout<<"num_iterations: "<<num_iterations<<"  Total time: "<<durations[i]<<", Exec time on GPU"<<durations2[i]<<std::endl;
    }
#endif
    num_iterations=1;
    double duration_ns=durations2[0];
    for(int i=0;i<nn;i++){
        duration_ns = durations2[i];
        if (duration_ns < target_duration_ns) {
            std::cerr<<duration_ns<<", "<<num_iterations<<std::endl;
            num_iterations *= 2;   
        }
        else{
            //std::cerr<<"Num iterations GPU: "<<num_iterations<<", Time: "<<duration_ns<<" nanoseconds\n";
            break;
        }
    }

        
    
    //double elapsed = static_cast<double>(endTime - startTime) * 1e-9;
    std::cout << "Number of iterations for GPU: " << num_iterations << std::endl;
    // Copy result back to host
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * vectorSizeGPU, c.data());
    std::cout << "Duration: " << duration_ns << " ns" << std::endl;
    return num_iterations;
}



// Function to calculate the mean of a subarray
double subarray_mean(const std::vector<double>& values, int start, int end)
{
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        sum += values[i];
    }
    return sum / (end - start);
}

// Function to calculate the variance of a subarray
//I comment dividing to number of samples for variance (see return value) because this way it works
double subarray_variance(const std::vector<double>& values, int start, int end)
{
    double mean = subarray_mean(values, start, end);
    double sum_sq_diff = 0.0;
    for (int i = start; i < end; i++) {
        double diff = (values[i] - mean);
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff;
    //return sum_sq_diff / (end - start);
}


// Function to detect the change point in a series of values
//It calculate A=variance_left_part + variance_right_part for all partition_point i and find the i with min A
//just be aware that instead of variance modified variance (which does not devide error by n) works 
int detect_change_point_1(const std::vector<double>& values)
{
    int n = values.size();

    // Calculate the variance of the entire series
    double total_variance = subarray_variance(values, 0, n);

    // Find the subarray with the minimum variance
    int min_variance_index = -1;
    double min_variance = std::numeric_limits<double>::max();
    for (int i = 1; i < n - 1; i++) {
        double variance_left = subarray_variance(values, 0, i);
        double variance_right = subarray_variance(values, i, n);
        double sum_variance = variance_left + variance_right;
        //std::cerr<<"i: "<<i<<", sum_var: "<<sum_variance<<std::endl;
        if (sum_variance < min_variance) {
            min_variance = sum_variance;
            min_variance_index = i;
        }
    }
    std::cerr<<"Min variance:"<<min_variance<<", index: "<<min_variance_index<<", total var: "<<total_variance<<std::endl;
    // Calculate the ratio of the variance of the subarray with minimum variance to the total variance
    double ratio = min_variance / total_variance;

    // If the ratio is below a certain threshold, assume that there is a change point
    double threshold = 0.999;
    if (ratio < threshold) {
        return min_variance_index;
    } else {
        return -1;  // No change point detected
    }
}



int fit_counts(const std::vector<double>& values, int start, int change_point, int end, double left_mean, double right_mean){
    int left_count = 0;
    int right_count=0;
    for (int i = start; i < change_point; i++) {
        double left_dist=std::abs(values[i]-left_mean);
        double right_dist=std::abs(values[i]-right_mean);
        left_count+=(left_dist<right_dist)?1:0;
    }
    for (int i = change_point; i < end; i++) {
        double left_dist=std::abs(values[i]-left_mean);
        double right_dist=std::abs(values[i]-right_mean);
        right_count+=(right_dist<left_dist)?1:0;
    }
#if Debug == DL2
    std::cerr<<"i: "<<change_point<<", left_mean: "<<left_mean
        <<" right_mean"<<right_mean<<"\nleft_count: "<<left_count<<" right_count: "<<right_count<<std::endl;
#endif
    return left_count+right_count;
}

//similar to detect_change_point_3 but instead of profiled values of v1 and v2 it uses mean of left and right parts
int detect_change_point_2(const std::vector<double>& values)
{
    int n = values.size();
    int max_fits = 0;
    int max_fits_index = -1;
    for (int i = 1; i < n - 1; i++) {
        double mean_left = subarray_mean(values, 0, i);
        double mean_right = subarray_mean(values, i, n);
        int fits= fit_counts(values,0,i,n,mean_left,mean_right);
        if (fits > max_fits) {
            max_fits = fits;
            max_fits_index = i;
        }
    }
    //std::cerr<<"max fits:"<<max_fits<<", index: "<<max_fits_index<<std::endl;
    return max_fits_index;
}


/*v1 and v2 values for freq and freq next are set based on initial profile 
(maybe it is possible to obtain with clustering or another method but consider that
values that have max acccurances are not correct as v1,v2 because it is possible that for example
7000 and 7001 be the most accur values while v1 is 1000)
Then we can optionally adjust all values to v1 and v2 (to which one that is closer to the value)
Then we could have two arrays CntV1 and CntV2 with size of data
CntV1[i]:the count(number) of V1 values from 0 to i
CntV2[i]:the count(number) of V2 vlaues from i to n
Then find the index(i) that CntV1[i]+CntV2[i] is maximum
This is the switch index

Simply counts numbers close to v1 and numbers close to v2 and 
*/
int detect_change_point_3(const std::vector<double>& values, Key key)
{
    Key key1=key;
    key1.NextFreq=key.Freq;
    
    int indx1=FreqMeasurements[key1].size()-1;
    double v1=FreqMeasurements[key1][indx1];

    Key key2=key;
    key2.Freq=key.NextFreq;
    int indx2=FreqMeasurements[key2].size()-1;
    double v2=FreqMeasurements[key2][indx2];
    
    int n = values.size();
    std::vector<int> CntVBig(n);
    std::vector<int> CntVSmall(n);
    int edge=(v1+v2)/2;
#if Debug==DL1
    std::cerr<<key1.PE<<", Freq: "<<key1.Freq<<", t: "<<v1<<std::endl;
    std::cerr<<key2.PE<<", Freq: "<<key2.Freq<<", t: "<<v2<<std::endl;
    std::cerr<<"edge: "<<edge<<std::endl;
#endif

    CntVSmall[0]=0;
    CntVBig[0]=0;
    //std::cerr<<"value: "<<values[0]<<std::endl;
    if(values[0]<edge){
        CntVSmall[0]=1;
    }
    else{
        CntVBig[0]=1;
    }
    //std::cerr<<"cntsmall: "<<CntVSmall[0]<<",  cntBig: "<<CntVBig[0]<<std::endl;
    for(int i=1; i<n;i++){
        //std::cerr<<"i: "<<i<<", value: "<<values[i];
        if(values[i]<edge){
            CntVSmall[i]=CntVSmall[i-1]+1;
            CntVBig[i]=CntVBig[i-1];
        }
        else{
            CntVBig[i]=CntVBig[i-1]+1;
            CntVSmall[i]=CntVSmall[i-1];
        }
        //std::cerr<<", cntsmall: "<<CntVSmall[i]<<",  cntBig: "<<CntVBig[i]<<std::endl;
    }
    
    std::vector<int>& cv1 = CntVSmall;
    std::vector<int>& cv2 = CntVBig;
    if(v1>v2){
        std::swap(cv1, cv2);
    }
    //std::cerr<<cv1[n-1]<<", "<<cv2[n-1]<<std::endl;
    int max_fits = 0;
    int max_fits_index = -1;
    int total_cnt_v2=cv2[n-1];
    //std::cerr<<total_cnt_v2<<std::endl;
    for (int i = 1; i < n - 1; i++) {
        int left_fit=cv1[i];
        int right_fit=total_cnt_v2-cv2[i];
        int fits=left_fit+right_fit;
#if Debug==DL2
        std::cerr<<"i: "<<i<<", left_fit: "<<left_fit<<", right_fit: "<<right_fit<<std::endl;
#endif
        if (fits > max_fits) {
            max_fits = fits;
            max_fits_index = i;
        }
    }
    //std::cerr<<"max fits:"<<max_fits<<", index: "<<max_fits_index<<std::endl;
    return max_fits_index;
}



/*For a point: 
s1=sum of a window at right
s2=sum of a window at left
index at which s1-s2 is maximum
*/
int detect_change_point_4(std::vector<double> durations, int window=10){
    double max_diff = 0.0;
    int max_diff_index=0;
    for(int j=window+1;j<durations.size()-window;j++){
        double s1 = std::accumulate(durations.begin()+j-window, durations.begin()+j, 0.0) ;
        double s2 = std::accumulate(durations.begin()+j, durations.begin()+j+window, 0.0) ;
        double diff = std::abs(s2 - s1);
        //double diff = std::abs(durations[j] - durations[j-1]);
#if Debug == DL2
        std::cerr<<"j: "<<j<<", s1: "<<s1<<", s2: "<<s2<<", diff: "<<diff<<std::endl;
#endif
        if(diff > max_diff){
            max_diff = diff;
            max_diff_index=j;
        }
    }
    //std::cerr << "Max diff: " << max_diff << " nano at index: "<<max_diff_index<<"\n";
    return max_diff_index;
}

//Profile PEs with swithch from a freqs settings to new freqs settings without wait to apply
double Measure_Time_2(Key key, std::vector<int> cpus, int LittleFreq, int BigFreq, int GPUFreq, int LittleFreqNew, int BigFreqNew, int GPUFreqNew){
    int num_iterations=key.Num_iterations;
    double result=0;
    //20000000/target_ns_CPU To run about 20ms(==20000000ns)
    
    std::vector<double> durations(N_loops_CPU);
    std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> time_stamps(N_loops_CPU);
    cpu_set_t set;
    CPU_ZERO(&set);
    for(int i=0;i<cpus.size();i++){
        CPU_SET(cpus[i],&set);
    }
    sched_setaffinity(0, sizeof(set), &set);
    dvfs.commit_freq(LittleFreq, BigFreq, GPUFreq);
    //warmup
    for(int i=0;i<num_iterations;i++){
        result += std::sqrt(i);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));


    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<num_iterations;i++){
        result += std::sqrt(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    time_stamps[0]=end;
    durations[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    dvfs.commit_freq(LittleFreqNew, BigFreqNew, GPUFreqNew);
    auto start0 = std::chrono::high_resolution_clock::now();
    for(int j=1;j<N_loops_CPU;j++){
        start = std::chrono::high_resolution_clock::now();
        for(int i=0;i<num_iterations;i++){
            result += std::sqrt(i);
        }
        end = std::chrono::high_resolution_clock::now();
        durations[j] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        time_stamps[j]=end;
    }
    auto end0 = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end0 - start0).count();
#if Debug == DL2
    double total_duration = std::accumulate(durations.begin()+1, durations.end(), 0.0) ;
    std::cerr << "accumulated execution time: " << total_duration << " nano\n";
    std::cerr << "elapsed execution time: " << elapsed << " nano\n";
    double coef=elapsed/total_duration;
#endif


    //int change_point_1 = detect_change_point_1(durations);
    //std::cerr<<"detector_1; changepoint_1:"<<change_point_1<<std::endl;
    //int change_point_2 = detect_change_point_2(durations);
    //std::cerr<<"detector_2; changepoint_2:"<<change_point_2<<std::endl;
    int change_point_3 = detect_change_point_3(durations,key);
    std::cerr<<"detector_3; changepoint_3:"<<change_point_3<<std::endl;
    //int change_point_4 = detect_change_point_4(durations,10);
    //std::cerr<<"detector_4; changepoint_4:"<<change_point_4<<std::endl;
    int change_point=change_point_3;
#if Debug == DL1
    int N_print=change_point*2;
    //N_print=N_loops_CPU/10;
    for(int j=0;j<N_print;j++){
        std::cerr <<j<< ": " << durations[j] << "ns\t";
        if ((j+1)%8==0)
            std::cerr<<"\n";
    }
#endif


    auto measured_delay = std::chrono::duration_cast<std::chrono::nanoseconds>(time_stamps[change_point] - start0).count();

#if Debug == DL2
    double delay = std::accumulate(durations.begin(), durations.begin()+change_point, 0.0) ;
    std::cerr<<"Delay is: "<<delay<<std::endl;
    std::cerr<<"coef x delay: "<<coef*delay<<std::endl;
#endif
    std::cerr<<"Measure delay: "<<measured_delay<<std::endl;
    std::cerr<<"**********************************************************\n\n\n";
    return measured_delay;
}






//Profile time (when switch from freqs to newfreqs) for GPU
double Measure_Time_GPU_2(Key key, std::vector<int> cpus, int LittleFreq, int BigFreq, int GPUFreq, int LittleFreqNew, int BigFreqNew, int GPUFreqNew, cl::CommandQueue queue, cl::Context context){
    std::vector<double> durations(N_loops_GPU);
    std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> time_stamps(N_loops_GPU);
    cpu_set_t set;
    CPU_ZERO(&set);
    for(int i=0;i<cpus.size();i++){
        CPU_SET(cpus[i],&set);
    }
    sched_setaffinity(0, sizeof(set), &set);
    dvfs.commit_freq(LittleFreq, BigFreq, GPUFreq);
    

    std::vector<float> a(vectorSizeGPU);
    std::vector<float> b(vectorSizeGPU);
    std::vector<float> c(vectorSizeGPU);
    
    // Initialize input vectors with random values
    for (int i = 0; i < vectorSizeGPU; ++i) {
        a[i] = random_float();
        b[i] = random_float();
    }

    // Create OpenCL buffers
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeof(float) * vectorSizeGPU);
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY, sizeof(float) * vectorSizeGPU);
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * vectorSizeGPU);

    // Copy input vectors to device
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * vectorSizeGPU, a.data());
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * vectorSizeGPU, b.data());

    
    // Load and compile the OpenCL kernel
    std::string kernelSource = R"(
        __kernel void vector_addition(__global float* a, __global float* b, __global float* c, int k) {
                const int gid = get_global_id(0);
                for (int i = 0; i < k; ++i) {
                    c[gid] += a[gid] * b[gid] * (i % 7);
                }
            }
        )";
        
    cl::Program program(context, kernelSource);
    program.build();
    // Create OpenCL kernel
    cl::Kernel kernel(program, ("vector_addition"));
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, sizeof(int), &num_iterations_GPU);
    


    //std::vector<cl::Kernel> kernels;
    std::vector<cl::Event> events(N_loops_GPU);
    
    for(int i = 0; i < N_loops_GPU; i++){
        //cl::Event event;
        //events[i]=std::move(event);
        events[i]=cl::Event();
    }
    
    auto first_event = clCreateUserEvent(context.get(), NULL);
    if(first_event==nullptr){
        std::cerr<<"user event null\n";
    }
    std::vector<cl::Event> first_events;
    cl::Event first_event_obj(first_event);
    first_events.push_back(first_event_obj);

    // Run the OpenCL kernel
    cl::Event prof_event;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    //warmup
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSizeGPU), cl::NullRange, nullptr, &prof_event);
    // Wait for the kernel to complete
    prof_event.wait();

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSizeGPU), cl::NullRange, &first_events, &events[0]);
    for(int i=1;i<N_loops_GPU;i++){
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSizeGPU), cl::NullRange, nullptr, &events[i]);                                      
    }
    
    //queue.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    clSetUserEventStatus(first_event, CL_COMPLETE);
    //dvfs.commit_freq(LittleFreqNew, BigFreqNew, GPUFreqNew);
    auto start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N_loops_GPU;i++){
        // Wait for the kernel to complete
        if(i==10){
            dvfs.commit_freq(LittleFreqNew, BigFreqNew, GPUFreqNew);
            start=std::chrono::high_resolution_clock::now();
        }
        events[i].wait();
        auto end=std::chrono::high_resolution_clock::now();
        time_stamps[i]=end; 
    }
    for(int i=0;i<N_loops_GPU;i++){
        cl_ulong start_time = events[i].getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end_time = events[i].getProfilingInfo<CL_PROFILING_COMMAND_END>();
        durations[i] = static_cast<double>(end_time - start_time);
    }

    int change_point_3 = detect_change_point_3(durations,key);
    std::cerr<<"detector_3; changepoint_3:"<<change_point_3<<std::endl;
    //int change_point_4 = detect_change_point_4(durations,10);
    //std::cerr<<"detector_4; changepoint_4:"<<change_point_4<<std::endl;
    int change_point=change_point_3;
#if Debug == DL1
    int N_print=change_point*2;
    //N_print=N_loops_CPU/10;
    for(int j=0;j<N_print;j++){
        std::cerr <<j<< ": " << durations[j] << "ns\t";
        if ((j+1)%8==0)
            std::cerr<<"\n";
    }
#endif

    auto measured_delay = std::chrono::duration_cast<std::chrono::nanoseconds>(time_stamps[change_point] - start).count();
    std::cerr<<"Measure delay: "<<measured_delay<<std::endl;
    std::cerr<<"**********************************************************\n\n\n";
    return measured_delay;
}


//Profile time (when switch from freqs to newfreqs) for GPU
//One big loop and then analetical 
double Measure_Time_GPU_3(Key key, std::vector<int> cpus, int LittleFreq, int BigFreq, int GPUFreq, int LittleFreqNew, int BigFreqNew, int GPUFreqNew, cl::CommandQueue queue, cl::Context context){
    int num_iterations=N_loops_GPU*num_iterations_GPU;
    //std::vector<double> durations(N_loops_GPU);
    //std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> time_stamps(N_loops_GPU);
    cpu_set_t set;
    CPU_ZERO(&set);
    for(int i=0;i<cpus.size();i++){
        CPU_SET(cpus[i],&set);
    }
    sched_setaffinity(0, sizeof(set), &set);
    dvfs.commit_freq(LittleFreq, BigFreq, GPUFreq);
    

    std::vector<float> a(vectorSizeGPU);
    std::vector<float> b(vectorSizeGPU);
    std::vector<float> c(vectorSizeGPU);
    
    // Initialize input vectors with random values
    for (int i = 0; i < vectorSizeGPU; ++i) {
        a[i] = random_float();
        b[i] = random_float();
    }

    // Create OpenCL buffers
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeof(float) * vectorSizeGPU);
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY, sizeof(float) * vectorSizeGPU);
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * vectorSizeGPU);

    // Copy input vectors to device
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * vectorSizeGPU, a.data());
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * vectorSizeGPU, b.data());

    
    // Load and compile the OpenCL kernel
    std::string kernelSource = R"(
        __kernel void vector_addition(__global float* a, __global float* b, __global float* c, int k) {
                const int gid = get_global_id(0);
                for (int i = 0; i < k; ++i) {
                    c[gid] += a[gid] * b[gid] * (i % 7);
                }
            }
        )";
        
    cl::Program program(context, kernelSource);
    program.build();
    // Create OpenCL kernel
    cl::Kernel kernel(program, ("vector_addition"));
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, sizeof(int), &num_iterations);

    
    auto first_event = clCreateUserEvent(context.get(), NULL);
    if(first_event==nullptr){
        std::cerr<<"user event null\n";
    }
    std::vector<cl::Event> first_events;
    cl::Event first_event_obj(first_event);
    first_events.push_back(first_event_obj);

    // Run the OpenCL kernel
    cl::Event prof_event_F0;
    cl::Event prof_event_F1;
    cl::Event prof_event_F0_1;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    //warmup
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSizeGPU), cl::NullRange, nullptr, nullptr);
    // Wait for the kernel to complete
    queue.finish();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));


    //Run with first Freq (F0)
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSizeGPU), cl::NullRange, nullptr, &prof_event_F0);
    // Wait for the kernel to complete
    prof_event_F0.wait();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    //Run when switch from F0 to F1 (F_0_1)
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSizeGPU), cl::NullRange, nullptr, &prof_event_F0_1);
    dvfs.commit_freq(LittleFreqNew, BigFreqNew, GPUFreqNew);
    // Wait for the kernel to complete
    prof_event_F0_1.wait();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    //Run with first Freq (F1)
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSizeGPU), cl::NullRange, nullptr, &prof_event_F1);
    // Wait for the kernel to complete
    prof_event_F1.wait();

    queue.finish();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    cl_ulong start_time = prof_event_F0.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end_time = prof_event_F0.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    auto duration_F_0 = static_cast<double>(end_time - start_time);

    start_time = prof_event_F0_1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    end_time = prof_event_F0_1.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    auto duration_F_0_1 = static_cast<double>(end_time - start_time);

    start_time = prof_event_F1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    end_time = prof_event_F1.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    auto duration_F_1 = static_cast<double>(end_time - start_time);


    double Delay=duration_F_0*(duration_F_0_1-duration_F_1)/(duration_F_0-duration_F_1);
#if Debug==DL1
    std::cerr<<"  Duration_F_0: "<<duration_F_0<<"\n  Duration_F_1: "<<duration_F_1<<"\nDuration_F_0_1: "<<duration_F_0_1<<std::endl;
    std::cerr<<"delay: "<<Delay<<std::endl;
    std::cerr<<"**********************************************************\n\n\n";
#endif
    
    return Delay;
}


void Measure_Two_Freqs_times(cl::CommandQueue queue, cl::Context cntx){
    Key key{0,"?", 0, 0};  
    //Measure GPU freqs
    key.Num_iterations=num_iterations_GPU;
    key.PE="GPU";
    std::cerr<<"Measure GPU Freqs:\n";
    for(int j=0;j<NumGPUFreqs;j++){
        key.Freq=j;
        std::cerr<<"Cur Freq:"<<j<<std::endl;
        for(int i=0;i<NumGPUFreqs;i++){
            if(i==j){
                continue;
            }

            key.NextFreq=i;
            std::cerr<<"NextFreq: "<<i<<std::endl;
            if (FreqMeasurements.count(key) > 0){
                int s=FreqMeasurements[key].size();
                std::cout<<"Already evaluated size:"<<s-1<<", avg:"<<FreqMeasurements[key][s-1]<<std::endl;
                continue;
            }
            double sum=0;
            //Warmup
            //Measure_Time_GPU_2(gpucpus, 0, 0, 0, 0, 0, 0, queue, cntx);
            //std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
            for(int k=0; k<N_runs; k++){            
                auto duration=Measure_Time_GPU_2(key, gpucpus, 0, 0, j, 0, 0, i, queue, cntx);
                FreqMeasurements[key].push_back(duration);
                sum+=duration;
                std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
            }
            FreqMeasurements[key].push_back(sum/N_runs);
        }
    }
    write_to_FreqMeasurements();


    //Measure GPU freqs
    key.Num_iterations=num_iterations_GPU*N_loops_GPU;
    key.PE="GPU";
    std::cerr<<"Measure GPU Freqs:\n";
    for(int j=0;j<NumGPUFreqs;j++){
        key.Freq=j;
        std::cerr<<"Cur Freq:"<<j<<std::endl;
        for(int i=0;i<NumGPUFreqs;i++){
            if(i==j){
                continue;
            }

            key.NextFreq=i;
            std::cerr<<"NextFreq: "<<i<<std::endl;
            if (FreqMeasurements.count(key) > 0){
                int s=FreqMeasurements[key].size();
                std::cout<<"Already evaluated size:"<<s-1<<", avg:"<<FreqMeasurements[key][s-1]<<std::endl;
                continue;
            }
            double sum=0;
            //Warmup
            //Measure_Time_GPU_2(gpucpus, 0, 0, 0, 0, 0, 0, queue, cntx);
            //std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
            for(int k=0; k<N_runs; k++){            
                auto duration=Measure_Time_GPU_3(key, gpucpus, 0, 0, j, 0, 0, i, queue, cntx);
                FreqMeasurements[key].push_back(duration);
                //In some profile it is negative (I think it happed when it is small)
                duration=std::max(0.0,duration);
                sum+=duration;
                std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
            }
            FreqMeasurements[key].push_back(sum/N_runs);
        }
    }
    write_to_FreqMeasurements();

    //Measure Little CPU freqs
    key.PE="Little";
    key.Num_iterations=num_iterations_CPU;  
    std::cerr<<"Measure Little Freqs:\n";
    for(int j=0;j<NumLittleFreqs;j++){
        key.Freq=j;
        std::cerr<<"Cur Freq:"<<j<<std::endl;
        for(int i=0;i<NumLittleFreqs;i++){
            if(i==j){
                continue;
            }
            key.NextFreq=i;
            std::cerr<<"NextFreq: "<<i<<std::endl;
            if (FreqMeasurements.count(key) > 0){
                int s=FreqMeasurements[key].size();
                std::cout<<"Already evaluated size:"<<s-1<<", avg:"<<FreqMeasurements[key][s-1]<<std::endl;
                continue;
            }
            double sum=0;
            //Warmup
            //Measure_Time_2(num_iterations, littlecpus, 0, 0, 0, 0, 0, 0);
            //std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
            //std::string sss;
            for(int k=0; k<N_runs; k++){            
                auto duration=Measure_Time_2(key, littlecpus, j, 0, 0, i, 0, 0);
                FreqMeasurements[key].push_back(duration);
                sum+=duration;
                std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
                //std::cin>>sss;
            }
            FreqMeasurements[key].push_back(sum/N_runs);
        }
    }
    write_to_FreqMeasurements();
    //Measure Big CPU freqs
    key.PE="Big";  
    key.Num_iterations=num_iterations_CPU;
    std::cerr<<"Measure Big Freqs:\n";
    for(int j=0;j<NumBigFreqs;j++){
        key.Freq=j;
        std::cerr<<"Cur Freq:"<<j<<std::endl;
        for(int i=0;i<NumBigFreqs;i++){
            if(i==j){
                continue;
            }
            key.NextFreq=i;
            std::cerr<<"NextFreq: "<<i<<std::endl;
            if (FreqMeasurements.count(key) > 0){
                int s=FreqMeasurements[key].size();
                std::cout<<"Already evaluated size:"<<s-1<<", avg:"<<FreqMeasurements[key][s-1]<<std::endl;
                continue;
            }
            double sum=0;
            //Warmup
            //Measure_Time_2(num_iterations, bigcpus, 0, 0, 0, 0, 0, 0);
            //std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
            for(int k=0; k<N_runs; k++){            
                auto duration=Measure_Time_2(key, bigcpus, 0, j, 0, 0, i, 0);
                FreqMeasurements[key].push_back(duration);
                sum+=duration;
                std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime));
            }
            FreqMeasurements[key].push_back(sum/N_runs);
        }
    }
    write_to_FreqMeasurements();
}


//Profile PEs with a freq(wait to applied)
double Measure_Time_initialize( std::vector<int> cpus, int LittleFreq, int BigFreq, int GPUFreq){
    double result=0;
    //std::array<double,N> durations;    
    std::vector<double> durations(N_loops_CPU);
    std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> time_stamps(N_loops_CPU);
    cpu_set_t set;
    CPU_ZERO(&set);
    for(int i=0;i<cpus.size();i++){
        CPU_SET(cpus[i],&set);
    }
    sched_setaffinity(0, sizeof(set), &set);
    dvfs.commit_freq(LittleFreq, BigFreq, GPUFreq);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    //warmup
    for(int i=0;i<num_iterations_CPU;i++){
        result += std::sqrt(i);
    }
    auto start0 = std::chrono::high_resolution_clock::now();
    for(int j=0;j<N_loops_CPU;j++){
        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0;i<num_iterations_CPU;i++){
            result += std::sqrt(i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        durations[j] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        time_stamps[j]=end;
    }
    auto end0 = std::chrono::high_resolution_clock::now();
    double t=std::accumulate(durations.begin(),durations.end(),0.0)/durations.size();
#if Debug==DL1
    std::cerr<<"t: "<<t<<std::endl;
#endif
    return t;
}

//Profile PEs with a freq(wait to applied)
double Measure_Time_initialize_GPU( std::vector<int> cpus, int LittleFreq, int BigFreq, int GPUFreq, cl::CommandQueue queue, cl:: Context context){
    std::vector<double> durations(N_loops_GPU);
    cpu_set_t set;
    CPU_ZERO(&set);
    for(int i=0;i<cpus.size();i++){
        CPU_SET(cpus[i],&set);
    }
    sched_setaffinity(0, sizeof(set), &set);
    dvfs.commit_freq(LittleFreq, BigFreq, GPUFreq);
    

    std::vector<float> a(vectorSizeGPU);
    std::vector<float> b(vectorSizeGPU);
    std::vector<float> c(vectorSizeGPU);
    
    // Initialize input vectors with random values
    for (int i = 0; i < vectorSizeGPU; ++i) {
        a[i] = random_float();
        b[i] = random_float();
    }

    // Create OpenCL buffers
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeof(float) * vectorSizeGPU);
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY, sizeof(float) * vectorSizeGPU);
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * vectorSizeGPU);

    // Copy input vectors to device
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * vectorSizeGPU, a.data());
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * vectorSizeGPU, b.data());

    
    // Load and compile the OpenCL kernel
    std::string kernelSource = R"(
        __kernel void vector_addition(__global float* a, __global float* b, __global float* c, int k) {
                const int gid = get_global_id(0);
                for (int i = 0; i < k; ++i) {
                    c[gid] += a[gid] * b[gid] * (i % 7);
                }
            }
        )";
        
    cl::Program program(context, kernelSource);
    program.build();
    // Create OpenCL kernel
    cl::Kernel kernel(program, ("vector_addition"));
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, sizeof(int), &num_iterations_GPU);

    
    // Run the OpenCL kernel
    cl::Event prof_event;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //warmup
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSizeGPU), cl::NullRange, nullptr, &prof_event);
    // Wait for the kernel to complete
    prof_event.wait();

    double total_time=0;
    for(int i=0; i<N_loops_GPU; i++) {
        //clEnqueueNDRangeKernel(queue, kernel, 1, NULL,&num_items, NULL, 0, NULL, &prof_event);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSizeGPU), cl::NullRange, nullptr, &prof_event);
        /*if(err < 0) {
            perror("Couldn't enqueue the kernel");
            exit(1);
        }*/
        queue.finish();
        /*clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_START,sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_END,sizeof(time_end), &time_end, NULL);*/
        cl_ulong start_time = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end_time = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        auto d = static_cast<double>(end_time - start_time);
        total_time += d;
    }
    // Copy result back to host
    //queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * vectorSizeGPU, c.data());   


    double avg=total_time/N_loops_GPU;
    
     
    
#if Debug==DL1
    printf("Average time = %fu\n", avg);
#endif
    
    return avg;
}


//Profile without changing freqs
void Init_Profile(cl::CommandQueue queue, cl::Context cntx){
    Key key{0,"?", 0, 0};  
    
    //Measure GPU freqs
    key.PE="GPU";
    key.Num_iterations=num_iterations_GPU;
    std::cerr<<"Measure GPU Freqs:\n";
    for(int j=0;j<NumGPUFreqs;j++){
        key.Freq=j;
        std::cerr<<"Cur Freq:"<<j<<std::endl;
        key.NextFreq=j;
        std::cerr<<"NextFreq: "<<j<<std::endl;
        if (FreqMeasurements.count(key) > 0){
            int s=FreqMeasurements[key].size();
            std::cout<<"Already evaluated size:"<<s-1<<", avg:"<<FreqMeasurements[key][s-1]<<std::endl;
            continue;
        }
        double sum=0;
        //Warmup
        //std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime_0));
        for(int k=0; k<N_runs; k++){            
            auto duration=Measure_Time_initialize_GPU(gpucpus, 0, 0, j, queue, cntx);
            FreqMeasurements[key].push_back(duration);
            sum+=duration;
            std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime_0));
        }
        FreqMeasurements[key].push_back(sum/N_runs);
    }
    write_to_FreqMeasurements();
    //Measure Little CPU freqs
    key.PE="Little";  
    key.Num_iterations=num_iterations_CPU;
    std::cerr<<"Measure Little Freqs:\n";
    for(int j=0;j<NumLittleFreqs;j++){
        key.Freq=j;
        std::cerr<<"Cur Freq:"<<j<<std::endl;  
        key.NextFreq=j;
        std::cerr<<"NextFreq: "<<j<<std::endl;
        if (FreqMeasurements.count(key) > 0){
            int s=FreqMeasurements[key].size();
            std::cout<<"Already evaluated size:"<<s-1<<", avg:"<<FreqMeasurements[key][s-1]<<std::endl;
            continue;
        }
        double sum=0;
        //Warmup
        //Measure_Time_2(num_iterations, littlecpus, 0, 0, 0, 0, 0, 0);
        //std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime_0));
        //std::string sss;
        for(int k=0; k<N_runs; k++){            
            auto duration=Measure_Time_initialize(littlecpus, j, 0, 0);
            FreqMeasurements[key].push_back(duration);
            sum+=duration;
            std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime_0));
            //std::cin>>sss;
        }
        FreqMeasurements[key].push_back(sum/N_runs);
    }
    write_to_FreqMeasurements();
    //Measure Big CPU freqs
    key.PE="Big";  
    key.Num_iterations=num_iterations_CPU;
    std::cerr<<"Measure Big Freqs:\n";
    for(int j=0;j<NumBigFreqs;j++){
        key.Freq=j;
        std::cerr<<"Cur Freq:"<<j<<std::endl;
        key.NextFreq=j;
        std::cerr<<"NextFreq: "<<j<<std::endl;
        if (FreqMeasurements.count(key) > 0){
            int s=FreqMeasurements[key].size();
            std::cout<<"Already evaluated size:"<<s-1<<", avg:"<<FreqMeasurements[key][s-1]<<std::endl;
            continue;
        }
        double sum=0;
        //Warmup
        //Measure_Time_2(num_iterations, bigcpus, 0, 0, 0, 0, 0, 0);
        //std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime_0));
        for(int k=0; k<N_runs; k++){            
            auto duration=Measure_Time_initialize(bigcpus, 0, j, 0);
            FreqMeasurements[key].push_back(duration);
            sum+=duration;
            std::this_thread::sleep_for(std::chrono::milliseconds(relaxtime_0));
        }
        FreqMeasurements[key].push_back(sum/N_runs);
    }
    write_to_FreqMeasurements();
}

int main(){

    init();
    Load_FreqMeasurements();
    dvfs.commit_freq(0, 0, 0);
    
    num_iterations_CPU = calc_num_iterations(target_ns_CPU);
    
    
    cl::Context context;
    cl::Device  gpuDevice;
    cl_int      err;
    std::tie(context, gpuDevice, err) = create_opencl_context_and_device();
    if(err != CL_SUCCESS)
        std::cerr<<"Failed to create OpenCL context\n";
    cl::CommandQueue queue = cl::CommandQueue(context, gpuDevice,CL_QUEUE_PROFILING_ENABLE);

    num_iterations_GPU = calc_num_iterations_GPU(target_ns_GPU,queue,context);

    //Measure_Freq_times(queue, context);
    
    Init_Profile(queue, context);
    //Key key{num_iterations_CPU,"Little",0,4};
    //Measure_Time_2(key,littlecpus, 0, 0, 0, 4, 0, 0);
    Measure_Two_Freqs_times(queue, context);
    


    return 0;
}
