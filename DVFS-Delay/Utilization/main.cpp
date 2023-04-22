#include "WorkerThread.cpp"
#include "getU2.cpp"
//#include <chrono>
#include <vector>
#include <list>



int main(){
    int num_threads_small=4;
    int num_threads_big=2;
    int num_threads=num_threads_big+num_threads_small;
    std::vector<std::unique_ptr<Thread>> Threads;
    for (int i=0;i<num_threads;i++){
        Threads.emplace_back(std::move(new Thread(i)));
    }

    // Read initial CPU usage statistics
    unsigned long long prev_total_time_small, prev_idle_time_small,prev_total_time_big, prev_idle_time_big;
    unsigned long long curr_total_time_small, curr_idle_time_small,curr_total_time_big, curr_idle_time_big;
    //getCpuStats(prev_total_time, prev_idle_time);

    //std::this_thread::sleep_for(std::chrono::seconds(2));
    

        

    // Run a loop to periodically calculate CPU utilization
    while (true) {
	    
        std::this_thread::sleep_for(std::chrono::seconds(1));
        getCpuStats(prev_total_time_small, prev_idle_time_small);
        for (int i=0;i<num_threads_small;i++){
            Threads[i]->start();
        }
        for (int i=0;i<num_threads_small;i++){
            Threads[i]->wait();
        }
        // Read current CPU usage statistics
        //unsigned long long curr_total_time, curr_idle_time;
        getCpuStats(curr_total_time_small, curr_idle_time_small);
        // Calculate time difference between current and previous readings
        unsigned long long total_time_diff_small = curr_total_time_small - prev_total_time_small;
        unsigned long long idle_time_diff_small = curr_idle_time_small - prev_idle_time_small;

        // Calculate CPU utilization
        double utilization_small = 100.0 * (1.0 - (double)idle_time_diff_small / total_time_diff_small);

        // Output the calculated CPU utilization
        std::cout << "Small CPU Utilization: " << utilization_small << "%" << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(1));
        getCpuStats(prev_total_time_big, prev_idle_time_big);
        for (int i=0;i<num_threads_big;i++){
            Threads[i]->start();
        }
        for (int i=0;i<num_threads_big;i++){
            Threads[i]->wait();
        }
        getCpuStats(curr_total_time_big, curr_idle_time_big);
    
        // Calculate time difference between current and previous readings
        unsigned long long total_time_diff_big = curr_total_time_big - prev_total_time_big;
        unsigned long long idle_time_diff_big = curr_idle_time_big - prev_idle_time_big;

        // Calculate CPU utilization
        double utilization_big = 100.0 * (1.0 - (double)idle_time_diff_big / total_time_diff_big);

        // Output the calculated CPU utilization
        std::cout << "Big CPU Utilization: " << utilization_big << "%" << std::endl;

        // Update previous CPU usage statistics
        //prev_total_time = curr_total_time;
        //prev_idle_time = curr_idle_time;
    }

    for (int i=0;i<num_threads;i++){
        Threads[i]->Done();
    }
    for (int i=0;i<num_threads;i++){
        Threads[i]->start();
    }
    std::cout<<"Finished\n";
    return 0;
}

