#include <iostream>
#include <fstream>
#include <unistd.h>

// Function to read CPU utilization from sysfs
double getCPUUtilization() {
    std::ifstream statFile("/proc/stat");
    std::string line;
    std::getline(statFile, line); // Read first line
    statFile.close();

    // Extract CPU utilization from the first line
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
    sscanf(line.c_str(), "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
           &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal, &guest, &guest_nice);

    unsigned long long totalCpuTime = user + nice + system + idle + iowait + irq + softirq + steal;
    unsigned long long idleCpuTime = idle + iowait;
    double cpuUtilization = 100.0 * (1.0 - static_cast<double>(idleCpuTime) / totalCpuTime);

    return cpuUtilization;
}

// Function to read GPU utilization using Mali Graphics Debugger (MGD)
double getGPUUtilization() {
    // Execute the "mgd_sysfs.sh" script to get GPU utilization
    std::string command = "mgd_sysfs.sh -d";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        std::cerr << "Failed to execute Mali Graphics Debugger (MGD)" << std::endl;
        return 0.0;
    }

    // Read the output of the script
    char buffer[128];
    std::string output;
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != nullptr)
            output += buffer;
    }
    pclose(pipe);

    // Extract GPU utilization from the output
    double gpuUtilization = 0.0;
    sscanf(output.c_str(), "%lf", &gpuUtilization);

    return gpuUtilization;
}

int main() {
    while (true) {
        double cpuUtilization = getCPUUtilization();
        double gpuUtilization = getGPUUtilization();
        std::cout << "CPU Utilization: " << cpuUtilization << "%" << std::endl;
        std::cout << "GPU Utilization: " << gpuUtilization << "%" << std::endl;

        usleep(100000);  // Sleep for 100 milliseconds (adjust as needed for desired interval)
    }

    return 0;
}

