#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>

// Function to get CPU usage statistics
void getCpuStats(unsigned long long& total_time, unsigned long long& idle_time) {
    std::ifstream file("/proc/stat");
    std::string line;
    //unsigned long long user, nice, system, idle;
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
    total_time = 0;
    idle_time = 0;

    if (file.is_open()) {
        std::getline(file, line);
        /*
        sscanf(line.c_str(), "cpu %llu %llu %llu %llu", &user, &nice, &system, &idle);
        total_time = user + nice + system + idle;
        idle_time = idle;
        */
        sscanf(line.c_str(), "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
           &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal, &guest, &guest_nice);
        total_time = user + nice + system + idle + iowait + irq + softirq + steal;
        idle_time = idle + iowait;
    }
    file.close();
}

/*int main() {
    // Read initial CPU usage statistics
    unsigned long long prev_total_time, prev_idle_time;
    unsigned long long curr_total_time, curr_idle_time;
    getCpuStats(prev_total_time, prev_idle_time);

    // Run a loop to periodically calculate CPU utilization
    while (true) {
	getCpuStats(prev_total_time, prev_idle_time);
        // Sleep for 1 second
        //std::this_thread::sleep_for(std::chrono::seconds(1));
	double j=0.0;
	for(int i=0;i<100000000;i++){
		j+=i+i*4-2/6;
	}
        // Read current CPU usage statistics
        //unsigned long long curr_total_time, curr_idle_time;
        getCpuStats(curr_total_time, curr_idle_time);

        // Calculate time difference between current and previous readings
        unsigned long long total_time_diff = curr_total_time - prev_total_time;
        unsigned long long idle_time_diff = curr_idle_time - prev_idle_time;

        // Calculate CPU utilization
        double utilization = 100.0 * (1.0 - (double)idle_time_diff / total_time_diff);

        // Output the calculated CPU utilization
        std::cout << "CPU Utilization: " << utilization << "%" << std::endl;

        // Update previous CPU usage statistics
        //prev_total_time = curr_total_time;
        //prev_idle_time = curr_idle_time;
    }

    return 0;
}*/

