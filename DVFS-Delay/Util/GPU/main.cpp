/*
We need to load .so libs (libOpencCL.so and other libs that this lib required) from board and specify them with -llibname (like -lOpenCL)
Another way is to add -shared to compile command which load these libraries at runtime but it generates a .o lib or -so lib which we need 
Another main source code to load and run the function inside it. 
Another way is to load functions with dlopen with lazy flag. "OpenCL.h" and "OpenCL.cpp" which are copied form armcl
exactly load the requred functions with dlopen, and I think the name is exactly same is their names in cl2.hpp
so when compiling it doesnot give error because implementation of those functions in .hpp are loaded with dlopen
(for example if you see the Context in cl2.hpp, that create a context, calls clCreateContext_ptr which is loaded with dlopen
*/

#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_MINIMUM_OPENCL_VERSION 110

#include <iostream>
#include <vector>
#include <CL/cl2.hpp>
#include <dlfcn.h>
#include "OpenCL.h"

#include <chrono>
#include <thread>
#include <ctime>
#include <fstream>

std::string active="/sys/devices/platform/ff9a0000.gpu/power/runtime_active_time";
std::string suspend="/sys/devices/platform/ff9a0000.gpu/power/runtime_suspended_time";

// Function to get CPU usage statistics
void getGpuStats(unsigned long long& active_time, unsigned long long& idle_time) {
    std::ifstream file_active(active.c_str());
    std::ifstream file_suspend(suspend.c_str());
    std::string line;
    if (file_active.is_open()) {
        std::getline(file_active, line);
        sscanf(line.c_str(), "%llu", &active_time);
        //std::cerr<<"active: "<<active_time<<std::endl;
    }
    file_active.close();

    if (file_suspend.is_open()) {
        std::getline(file_suspend, line);
        sscanf(line.c_str(), "%llu", &idle_time);
        //std::cerr<<"idle: "<<idle_time<<std::endl;
    }
    file_suspend.close();
}

// Function to generate random float values between 0 and 1
float random_float() {
    return static_cast<float>(rand()) / RAND_MAX;
}

/*
See armcl:src/runtimeCL/CLHelpers.cpp
and: armcl:src/runtimeCL/CL/CLScheduler.cpp
std::tuple<cl::Context, cl::Device, cl_int>
*/

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

int main() {
    
    
    //cl::Context context(CL_DEVICE_TYPE_GPU);
    //cl::CommandQueue queue(context);
    
    cl::Context context;
    cl::Device  gpuDevice;
    cl_int      err;
    std::tie(context, gpuDevice, err) = create_opencl_context_and_device();
    if(err != CL_SUCCESS)
        std::cerr<<"Failed to create OpenCL context\n";
    cl::CommandQueue queue = cl::CommandQueue(context, gpuDevice,CL_QUEUE_PROFILING_ENABLE);

    
    // Create input and output vectors
    const int vectorSize = 10000000;
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


    unsigned long long prev_active_time, prev_idle_time;
    unsigned long long cur_active_time, cur_idle_time;
    getGpuStats(prev_active_time, prev_idle_time);
    auto tstart=std::chrono::high_resolution_clock::now();

    // Run the OpenCL kernel
    cl::Event event;
    cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vectorSize), cl::NullRange, nullptr, &event);
    //cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    // Wait for the kernel to complete
    event.wait();

    // Copy result back to host
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * vectorSize, c.data());

    // Measure GPU utilization during the task
    cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    getGpuStats(cur_active_time, cur_idle_time);
    auto tfinish=std::chrono::high_resolution_clock::now();
    double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tfinish - tstart).count();
    std::cerr<<"Cost: "<<1000*cost0<<" ms"<<std::endl;

    unsigned long long active_time_diff = cur_active_time - prev_active_time;
    unsigned long long idle_time_diff = cur_idle_time - prev_idle_time;
    std::cerr<<"active time: "<<active_time_diff<<" ms"<<std::endl;
    std::cerr<<"idle time: "<<idle_time_diff<<" ms"<<std::endl;
    // Calculate CPU utilization
    double utilization = 100.0 * ((double)active_time_diff / (active_time_diff + idle_time_diff));
    // Output the calculated CPU utilization
    std::cout << "GPU Utilization1: " << utilization << "%" << std::endl;

    
    double elapsed = static_cast<double>(endTime - startTime) * 1e-9;
    double gpuUtilization = elapsed / (endTime - startTime);

    // Print GPU utilization
    //std::cout << "GPU Utilization: " << gpuUtilization * 100 << "%" << std::endl;

    return 0;
}

