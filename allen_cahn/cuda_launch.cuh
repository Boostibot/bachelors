#pragma once
#include "cuda_util.cuh"
#include "cuda_device.cuh"

//This fille attempts to find "optimal" kernel launch configuration. 
//
//This process is split into three parts in order:
// 1. Cuda_Launch_Constraints - Specifies the constraints we have on the kernel (min/max sizes, shared memory ...). Can be cached.
// 2. Cuda_Launch_Bounds - Calculates the bounds for each of the launch paramters. Can be cached.
// 3. Cuda_Launch_Config - Calculates the actaul paramters used to launch a kernel from Cuda_Launch_Bounds and Cuda_Launch_Config for tuning.
//                         Cannot be cached when allowing for custom Cuda_Launch_Config. Is very fast.
// 
// The above approachw as chosen because:
// - Calculating launch paramters is non trivial, thus we want to factor it into a few universal functions.
// - Querrying, recaluclating the bounds and searching for optimal parameters is expensive in terms of cpu time. 
//   Thus we need to separate the procedure into a few steps where intermediate results can be cached.
// - Tunability. All of the kernels we define are entirely customizable through lambdas. Because of this the runtime
//   characteristics of the resulting kernel + inlined lambda is expected to warry a lot and with that the optimal launch
//   configuration. We cannot inspect the passed in lambda in any way so wel let the caller provide their own Cuda_Launch_Params.

struct Cuda_Launch_Params {
    // Value in range [0, 1] selecting on of the preferd sizes. 
    // Prefered sizes are valid sizes that achieve maximum utilization of hardware, usually powers of two.
    double preferd_block_size_ratio = 1; 

    // If greater than zero and within the valid block size range used as the block size regardless of
    // anything else. If is not given or not within range block size is determined by preferd_block_size_ratio instead.
    // uint   preferd_block_size = 0;
    uint preferd_block_size = {0};

    //The cap on number of blocks. Can be used to tune sheduler.
    uint   max_block_count = UINT_MAX;

    //The stream used for the launch
    cudaStream_t stream = 0;
};

struct Cuda_Launch_Config {
    uint block_size;
    uint block_count;
    uint shared_memory;
    uint dynamic_shared_memory;

    uint desired_block_count;
    uint max_concurent_blocks;
};  

struct Cuda_Launch_Bounds {
    uint min_block_size;
    uint max_block_size;
    uint preferd_block_sizes[12];
    uint preferd_block_sizes_count;

    double used_shared_memory_per_thread;
    uint used_shared_memory_per_block;
};  

struct Cuda_Launch_Constraints {
    double used_shared_memory_per_thread = 0;
    uint used_shared_memory_per_block = 0;
    
    uint used_register_count_per_block = 0;
    uint used_constant_memory = 0;

    uint max_shared_mem = UINT_MAX;
    uint max_block_size = UINT_MAX;
    uint min_block_size = 0;

    cudaFuncAttributes attributes;
};

static Cuda_Launch_Config cuda_get_launch_config(csize N, Cuda_Launch_Bounds bounds, Cuda_Launch_Params params)
{
    Cuda_Info info = cuda_one_time_setup();
    Cuda_Launch_Config launch = {0};
    if(params.preferd_block_size > 0)
    {
        if(bounds.min_block_size <= params.preferd_block_size && params.preferd_block_size <= bounds.max_block_size)
            launch.block_size = params.preferd_block_size;
    }

    if(launch.block_size == 0)
    {
        ASSERT(bounds.preferd_block_sizes_count >= 1);
        ASSERT(0 <= params.preferd_block_size_ratio && params.preferd_block_size_ratio <= 1);
        uint index = (uint) round(params.preferd_block_size_ratio*(bounds.preferd_block_sizes_count - 1));
        launch.block_size = bounds.preferd_block_sizes[index];
    }

    launch.dynamic_shared_memory = (uint) (bounds.used_shared_memory_per_thread*launch.block_size + 0.5);
    launch.shared_memory = launch.dynamic_shared_memory + bounds.used_shared_memory_per_block;
    
    launch.max_concurent_blocks = UINT_MAX;
    if(launch.shared_memory > 0)
        launch.max_concurent_blocks = info.prop.multiProcessorCount*(info.prop.sharedMemPerMultiprocessor/launch.shared_memory); 

    //Only fire as many blocks as the hardware could ideally support at once. 
    // Any more than that is pure overhead for the sheduler
    launch.desired_block_count = DIV_CEIL((uint) N, launch.block_size);
    launch.block_count = MIN(MIN(launch.desired_block_count, launch.max_concurent_blocks), params.max_block_count);

    return launch;
}

static Cuda_Launch_Bounds cuda_get_launch_bounds(Cuda_Launch_Constraints constraints)
{
    Cuda_Launch_Bounds out = {0};
    Cuda_Info info = cuda_one_time_setup();

    uint max_shared_mem = MIN(info.prop.sharedMemPerBlock, constraints.max_shared_mem);
    uint block_size_hw_upper_bound = info.prop.maxThreadsPerBlock;
    uint block_size_hw_lower_bound = DIV_CEIL(info.prop.maxThreadsPerMultiProcessor + 1, info.prop.maxBlocksPerMultiProcessor + 1);

    uint block_size_max = MIN(block_size_hw_upper_bound, constraints.max_block_size);
    uint block_size_min = MAX(block_size_hw_lower_bound, constraints.min_block_size);
    
    block_size_max = ROUND_DOWN(block_size_max, WARP_SIZE);
    block_size_min = ROUND_UP(block_size_min, WARP_SIZE);

    out.min_block_size = block_size_min;
    out.max_block_size = block_size_max;
    if(constraints.used_shared_memory_per_thread > 0)
    {
        uint shared_memory_max_block_size = (max_shared_mem - constraints.used_shared_memory_per_block)/constraints.used_shared_memory_per_thread;
        out.max_block_size = MIN(out.max_block_size, shared_memory_max_block_size);
    }

    if(out.max_block_size < out.min_block_size || out.max_block_size == 0)
    {
        LOG_ERROR("kernel", "no matching launch info found! @TODO: print constraints");
        return out;
    }

    //Find the optimal block_size. It must
    // 1) be twithin range [block_size_min, block_size_max] (to be feasible)
    // 2) be divisible by warpSize (W) (to have good block utlization)
    // 3) divide prop.maxThreadsPerMultiProcessor (to have good streaming multiprocessor utilization - requiring less SMs)
    for(uint curr_size = out.min_block_size; curr_size <= out.max_block_size; curr_size += WARP_SIZE)
    {
        if(info.prop.maxThreadsPerMultiProcessor % curr_size == 0)
        {   
            if(out.preferd_block_sizes_count < STATIC_ARRAY_SIZE(out.preferd_block_sizes))
                out.preferd_block_sizes[out.preferd_block_sizes_count++] = curr_size;
            else
                out.preferd_block_sizes[out.preferd_block_sizes_count - 1] = curr_size;
        }
    }

    if(out.preferd_block_sizes_count == 0)
        out.preferd_block_sizes[0] = out.max_block_size;

    out.used_shared_memory_per_block = constraints.used_shared_memory_per_block;
    out.used_shared_memory_per_thread = constraints.used_shared_memory_per_thread;
    return out;
}

static Cuda_Launch_Constraints cuda_constraints_launch_constraints(const void* kernel)
{
    cudaFuncAttributes attributes = {};
    CUDA_TEST(cudaFuncGetAttributes(&attributes, kernel));

    Cuda_Launch_Constraints constraints = {};
    constraints.max_shared_mem = attributes.maxDynamicSharedSizeBytes;
    constraints.max_block_size = attributes.maxThreadsPerBlock;
    constraints.used_constant_memory = attributes.constSizeBytes;
    constraints.used_shared_memory_per_block = attributes.sharedSizeBytes;
    constraints.attributes = attributes;
    return constraints;
}