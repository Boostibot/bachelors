#pragma once

//This file selects and caches info about the optimal device. It also prints the selected device info.
//
//We will expand this to be more involved when when multi-gpu setups are supported.

#include "cuda_util.cuh"

struct Cuda_Info {
    int device_id;
    cudaDeviceProp prop;
};

static Cuda_Info cuda_one_time_setup()
{
    static bool was_setup = false;
    static Cuda_Info info = {0};
    
    if(was_setup == false)
    {
        enum {MAX_DEVICES = 100};
        cudaDeviceProp devices[MAX_DEVICES] = {0};
        double scores[MAX_DEVICES] = {0};
        double peak_memory[MAX_DEVICES] = {0};
        
        int nDevices = 0;
        CUDA_TEST(cudaGetDeviceCount(&nDevices));
        if(nDevices > MAX_DEVICES)
        {
            ASSERT(false, "wow this should probably not happen!");
            nDevices = MAX_DEVICES;
        }
        TEST(nDevices > 0, "Didnt find any CUDA capable devices. Stopping.");

        for (int i = 0; i < nDevices; i++) 
            CUDA_DEBUG_TEST(cudaGetDeviceProperties(&devices[i], i));

        //compute maximum in each tracked category to
        // be able to properly select the best device forthe job!
        cudaDeviceProp max_prop = {0};
        double max_peak_memory = 0;
        for (int i = 0; i < nDevices; i++) 
        {
            cudaDeviceProp prop = devices[i];
            max_prop.warpSize = MAX(max_prop.warpSize, prop.warpSize);
            max_prop.multiProcessorCount = MAX(max_prop.multiProcessorCount, prop.multiProcessorCount);
            max_prop.concurrentKernels = MAX(max_prop.concurrentKernels, prop.concurrentKernels);
            max_prop.memoryClockRate = MAX(max_prop.memoryClockRate, prop.memoryClockRate);
            max_prop.memoryBusWidth = MAX(max_prop.memoryBusWidth, prop.memoryBusWidth);
            max_prop.totalGlobalMem = MAX(max_prop.totalGlobalMem, prop.totalGlobalMem);
            max_prop.sharedMemPerBlock = MAX(max_prop.sharedMemPerBlock, prop.sharedMemPerBlock);
            peak_memory[i] = 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6;

            max_peak_memory = MAX(max_peak_memory, peak_memory[i]);
        }

        double max_score = 0;
        int max_score_i = 0;
        for (int i = 0; i < nDevices; i++) 
        {
            cudaDeviceProp prop = devices[i];
            double score = 0
                + 0.40 * prop.warpSize/max_prop.warpSize
                + 0.40 * prop.multiProcessorCount/max_prop.multiProcessorCount
                + 0.05 * prop.concurrentKernels/max_prop.concurrentKernels
                + 0.05 * peak_memory[i]/max_peak_memory
                + 0.05 * prop.totalGlobalMem/max_prop.totalGlobalMem
                + 0.05 * prop.sharedMemPerBlock/max_prop.sharedMemPerBlock
                ;

            scores[i] = score;
            if(max_score < score)
            {
                max_score = score;
                max_score_i = i;
            }
        }
        cudaDeviceProp selected = devices[max_score_i];
        info.prop = selected;
        info.device_id = max_score_i;
        was_setup = true;
        CUDA_TEST(cudaSetDevice(info.device_id));

        LOG_INFO("CUDA", "Listing devices below (%i):\n", nDevices);
        for (int i = 0; i < nDevices; i++)
            LOG_INFO(">CUDA", "[%i] %s (score: %lf) %s\n", i, devices[i].name, scores[i], i == max_score_i ? "[selected]" : "");

        LOG_INFO("CUDA", "Selected '%s':\n", selected.name);
        LOG_INFO("CUDA", "  Multi Processor count: %i\n", selected.multiProcessorCount);
        LOG_INFO("CUDA", "  Warp-size: %i\n", selected.warpSize);
        LOG_INFO("CUDA", "  Max thread dim: %i %i %i\n", selected.maxThreadsDim[0], selected.maxThreadsDim[1], selected.maxThreadsDim[2]);
        LOG_INFO("CUDA", "  Max threads per block: %i\n", selected.maxThreadsPerBlock);
        LOG_INFO("CUDA", "  Max threads per multi processor: %i\n", selected.maxThreadsPerMultiProcessor);
        LOG_INFO("CUDA", "  Max blocks per multi processor: %i\n", selected.maxBlocksPerMultiProcessor);
        LOG_INFO("CUDA", "  Memory Clock Rate (MHz): %i\n", selected.memoryClockRate/1024);
        LOG_INFO("CUDA", "  Memory Bus Width (bits): %i\n", selected.memoryBusWidth);
        LOG_INFO("CUDA", "  Peak Memory Bandwidth (GB/s): %.1f\n", peak_memory[max_score_i]);
        LOG_INFO("CUDA", "  Global memory (Gbytes) %.1f\n",(float)(selected.totalGlobalMem)/1024.0/1024.0/1024.0);
        LOG_INFO("CUDA", "  Shared memory per block (Kbytes) %.1f\n",(float)(selected.sharedMemPerBlock)/1024.0);
        LOG_INFO("CUDA", "  Constant memory (Kbytes) %.1f\n",(float)(selected.totalConstMem)/1024.0);
        LOG_INFO("CUDA", "  minor-major: %i-%i\n", selected.minor, selected.major);
        LOG_INFO("CUDA", "  Concurrent kernels: %s\n", selected.concurrentKernels ? "yes" : "no");
        LOG_INFO("CUDA", "  Concurrent computation/communication: %s\n\n",selected.deviceOverlap ? "yes" : "no");
    }

    return info;
}