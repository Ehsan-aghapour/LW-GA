#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <unistd.h>


void set_thread_affinity(int core_id)
{
    if(core_id < 0)
    {
        return;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core_id, &set);
    sched_setaffinity(0, sizeof(set), &set);
}
class Thread final
{
public:
    /** Start a new thread
     *
     * Thread will be pinned to a given core id if value is non-negative
     *
     * @param[in] core_pin Core id to pin the thread on. If negative no thread pinning will take place
     */
    explicit Thread(int core_pin = -1);

    /*Thread(const Thread &) = delete;
    Thread &operator=(const Thread &) = delete;
    Thread(Thread &&)                 = delete;
    Thread &operator=(Thread &&) = delete;*/
	//void Create();

    

    /** Destructor. Make the thread join. */
    ~Thread();

    /** Request the worker thread to start executing workloads.
     *
     * The thread will start by executing workloads[info.thread_id] and will then call the feeder to
     * get the index of the following workload to run.
     *
     * @note This function will return as soon as the workloads have been sent to the worker thread.
     * wait() needs to be called to ensure the execution is complete.
     */
    //void start(std::vector<IScheduler::Workload> *workloads, ThreadFeeder &feeder, const ThreadInfo &info);
    void start();

    /** Wait for the current kernel execution to complete. */
    void wait();

    /** Function ran by the worker thread. */
    void worker_thread();
    void Done(){
        done=true;
    }

    bool				done{false};
private:
    std::thread                        _thread{};
    //ThreadInfo                         _info{};
    //std::vector<IScheduler::Workload> *_workloads{ nullptr };
    //ThreadFeeder                      *_feeder{ nullptr };
    std::mutex                         _m{};
    std::condition_variable            _cv{};
    bool                               _wait_for_work{ false };
    bool                               _job_complete{ true };
    std::exception_ptr                 _current_exception{ nullptr };
    int                                _core_pin{ -1 };
    std::mutex                         _m_create{};
    bool								created{ false};

};


Thread::Thread(int core_pin)
    : _core_pin(core_pin)
{
    _thread = std::thread(&Thread::worker_thread, this);
    //_thread = std::thread(&Thread::worker_thread);
}


/*void Thread::Create()
//    : _core_pin(core_pin)
{
	{
        std::unique_lock<std::mutex> lock(_m_create);
        if(!created){
        	created=true;
        	lock.unlock();
        	ALOGI("DRPM__Ehsan: Creating Thread, PID:%d.",getpid());
        	_thread = std::thread(&Thread::worker_thread, this);

        }
	}
    //_thread = std::thread(&Thread::worker_thread);
}*/


Thread::~Thread()
{
    //std::cerr<<"tamam\n";
    // Make sure worker thread has ended
    if(_thread.joinable())
    {
        //ThreadFeeder feeder;
        //start(nullptr, feeder, ThreadInfo());
        start();
        _thread.join();
    }
}

//Thread worker_thread;


//void Thread::start(std::vector<IScheduler::Workload> *workloads, ThreadFeeder &feeder, const ThreadInfo &info)
void Thread::start()
{
    //_workloads = workloads;
    //_feeder    = &feeder;
    //_info      = info;
    timespec t_switch;
	//printf("DRPM__Ehsan: start function before mutex, PID:%d.",getpid());
	clock_gettime(CLOCK_MONOTONIC, &t_switch);
    {
        std::lock_guard<std::mutex> lock(_m);
        _wait_for_work = true;
        _job_complete  = false;
        //printf("DRPM__Ehsan: start function after mutex, PID:%d.",getpid());
    }
    _cv.notify_one();
}


void Thread::wait()
{
	//printf("DRPM__Ehsan: in waiting, PID:%d.",getpid());
    {
        std::unique_lock<std::mutex> lock(_m);
        //printf("DRPM__Ehsan: wait lock aquired, PID:%d.",getpid());
        _cv.wait(lock, [&] { return _job_complete; });
    }

    if(_current_exception)
    {
        std::rethrow_exception(_current_exception);
    }
}





float process_workloads(int c){
    int t=100000000;
    float s=0;
    bool cluster=c>3;
    double coef=1;
    if (cluster){
        coef=4.4;
    }
    t=t*coef;
    //std::cout<<"c:"<<c<<" cluster:"<<cluster<<" t:"<<t<<std::endl;
    for (int i=0;i<t;i++){
        s+=(c+1)*2/3.6-4;
    }
    s=s/1000000;
    return s;
}


void Thread::worker_thread()
{
    set_thread_affinity(_core_pin);

    while(true)
    {
        std::unique_lock<std::mutex> lock(_m);
        _cv.wait(lock, [&] { return _wait_for_work; });
        //std::cerr<<"notified\n";
        _wait_for_work = false;

        _current_exception = nullptr;

        // Time to exit
        //if(_workloads == nullptr)
        if(done)
        {
            return;
        }

/*
 * #ifndef EXCEPTIONS_DISABLEDD
        try
        {
//#endif
            process_workloads(1000);

//#ifndef EXCEPTIONS_DISABLEDD
        }
        catch(...)
        {
            _current_exception = std::current_exception();
        }
//#endif
 */
        //process_workloads(1000);
        float ss=process_workloads(_core_pin);
        std::cout<<"Result for thread on core "<<_core_pin<<" is :"<<ss<<std::endl;
        _job_complete = true;
        //printf("DRPM__Ehsan: job complete, PID:%d.",getpid());
        lock.unlock();
        _cv.notify_one();
    }
}
