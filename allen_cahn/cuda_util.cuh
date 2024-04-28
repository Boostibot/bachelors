#pragma once
#define SHARED __host__ __device__

//For the moment we compile these onese as well but
// just as static so we can safely link
#define EXPORT static
#include "assert.h"
#include "defines.h"
#include "log.h"

#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdarg.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

typedef int csize;

//Can be defined to something else if we wish to for example use size_t or uint
static bool _test_cuda_(cudaError_t error, const char* expression, int line, const char* file, const char* function, const char* format, ...)
{
    if(error != cudaSuccess)
    {
        log_message("CUDA", LOG_FATAL, line, file, function, "CUDA_TEST(%s) failed with '%s'! %s %s:%i\n", expression, cudaGetErrorString(error), function, file, line);
        if(format != NULL && strlen(format) != 0)
        {
            va_list args;               
            va_start(args, format);     
            vlog_message(">CUDA", LOG_FATAL, line, file, function, format, args);
            va_end(args);  
        }

        log_flush();
    }
    return error == cudaSuccess;
}

#define CUDA_TEST(status, ...) (_test_cuda_((status), #status,  __LINE__, __FUNCTION__, __FILE__, "" __VA_ARGS__) ? (void) 0 : abort())

#ifdef DO_DEBUG
    #define CUDA_DEBUG_TEST(status, ...) (0 ? printf("" __VA_ARGS__) : (status))
#else
    #define CUDA_DEBUG_TEST(status, ...) CUDA_TEST(status, __VA_ARGS__)
#endif


enum {
    WARP_SIZE = 32

    // We use a constant. If this is not the case we will 'need' a different algorhimt anyway. 
    // The PTX codegen treats warpSize as 'runtime immediate constant' which from my undertanding
    // is a special constat accesible through its name 'mov.u32 %r6, WARP_SZ;'. 
    // In many contexts having it be a immediate constant is not enough. 
    // For example when doing 'x % warpSize == 0' the code will emit 'rem.s32' instruction which is
    // EXTREMELY costly (on GPUs even more than CPUs!) compared to the single binary 'and.b32' emmited 
    // in case of 'x % WARP_SIZE == 0'.

    // If you would need to make the below code a bit more "future proof" you could make WARP_SIZE into a 
    // template argument. However we really dont fear this to cahnge anytime soon since many functions
    // such as __activemask(), __ballot_sync() or anything else producing lane mask, operates on u32 which 
    // assumes WARP_SIZE == 32.

    // Prior to launching the kernel we check if warpSize == WARP_SIZE. If it does not we error return.
};



struct Cuda_Info {
    int device_id;
    bool has_broken_driver;
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
        // be able to properly select the best device for
        // the job!
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
        info.has_broken_driver = strcmp(selected.name, "NVIDIA GeForce MX450") == 0;
        was_setup = true;
        CUDA_TEST(cudaSetDevice(info.device_id));

        LOG_INFO("CUDA", "Listing devices below (%i):\n", nDevices);
        for (int i = 0; i < nDevices; i++)
            LOG_INFO(">CUDA", "[%i] %s (score: %lf) %s\n", i, devices[i].name, scores[i], i == max_score_i ? "[selected]" : "");

        // selected.maxThreadsDim
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

struct Cuda_Launch_Params {
    // Value in range [0, 1] selecting on of the preferd sizes. 
    // Prefered sizes are valid sizes that achieve maximum utilization of hardware, usually powers of two.
    double preferd_block_size_ratio = 1; 

    // If greater than zero and within the valid block size range used as the block size regardless of
    // anything else. If is not given or not within range block size is determined by preferd_block_size_ratio instead.
    uint   preferd_block_size = 0;

    //The cap on number of blocks. Can be used to tune sheduler.
    uint   max_block_count = UINT_MAX;
    
    //when greater then zero caps on number of blocks used for drivers with broken sheduler (such as the MX450).
    uint   max_block_count_for_broken_drivers = 0;

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

struct Cuda_Launch_Query {
    double used_shared_memory_per_thread = 0;
    uint used_shared_memory_per_block = 0;
    
    uint used_register_count_per_block = 0;
    uint used_constant_memory = 0;

    uint max_shared_mem = UINT_MAX;
    uint max_block_size = UINT_MAX;
    uint min_block_size = 0;

    cudaFuncAttributes attributes;
};

//Recalculates launch bounds using a different N
Cuda_Launch_Config cuda_get_launch_config(csize N, Cuda_Launch_Bounds bounds, Cuda_Launch_Params params)
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
    if(info.has_broken_driver && params.max_block_count_for_broken_drivers)
        launch.block_count = MIN(launch.block_count, params.max_block_count_for_broken_drivers);

    return launch;
}

Cuda_Launch_Bounds cuda_query_launch_bounds(Cuda_Launch_Query query)
{
    Cuda_Launch_Bounds out = {0};
    Cuda_Info info = cuda_one_time_setup();

    uint max_shared_mem = MIN(info.prop.sharedMemPerBlock, query.max_shared_mem);
    uint block_size_hw_upper_bound = info.prop.maxThreadsPerBlock;
    uint block_size_hw_lower_bound = DIV_CEIL(info.prop.maxThreadsPerMultiProcessor + 1, info.prop.maxBlocksPerMultiProcessor + 1);

    uint block_size_max = MIN(block_size_hw_upper_bound, query.max_block_size);
    uint block_size_min = MAX(block_size_hw_lower_bound, query.min_block_size);
    
    block_size_max = (block_size_max/WARP_SIZE)*WARP_SIZE;
    block_size_min = DIV_CEIL(block_size_min,WARP_SIZE)*WARP_SIZE;

    out.min_block_size = block_size_min;
    out.max_block_size = block_size_max;
    if(query.used_shared_memory_per_thread > 0)
    {
        uint shared_memory_max_block_size = (max_shared_mem - query.used_shared_memory_per_block)/query.used_shared_memory_per_thread;
        out.max_block_size = MIN(out.max_block_size, shared_memory_max_block_size);
    }

    if(out.max_block_size < out.min_block_size || out.max_block_size == 0)
    {
        LOG_ERROR("kernel", "no matching launch info found! @TODO: print query");
        return out;
    }

    //Find the optimal block_size. It must
    // 1) be twithin range [block_size_min, block_size_max] (to be feasible)
    // 2) be divisible by warpSize (W) (to have good block utlization)
    // 3) be as big as possible (should be at least WARP_SIZE^2 = 1024 for maximum reduction (also right now 1024 is the max size of block on all cards))
    // 4) divide prop.maxThreadsPerMultiProcessor (to have good streaming multiprocessor utilization - requiring less SMs)
    for(uint curr_size = out.max_block_size; curr_size >= out.min_block_size; curr_size -= WARP_SIZE)
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

    out.used_shared_memory_per_block = query.used_shared_memory_per_block;
    out.used_shared_memory_per_thread = query.used_shared_memory_per_thread;
    return out;
}

Cuda_Launch_Query cuda_launch_query_from_kernel(const void* kernel)
{
    cudaFuncAttributes attributes = {};
    CUDA_TEST(cudaFuncGetAttributes(&attributes, kernel));

    Cuda_Launch_Query query = {};
    query.max_shared_mem = attributes.maxDynamicSharedSizeBytes;
    query.max_block_size = attributes.maxThreadsPerBlock;
    query.used_constant_memory = attributes.constSizeBytes;
    query.used_shared_memory_per_block = attributes.sharedSizeBytes;
    query.attributes = attributes;
    return query;
}

enum {
    REALLOC_COPY = 1,
    REALLOC_ZERO = 2,
};

static void* _cuda_realloc(void* old_ptr, size_t new_size, size_t old_size, int flags, const char* file, const char* function, int line)
{
    Cuda_Info info = cuda_one_time_setup();
    LOG_INFO("CUDA", "realloc " MEMORY_FMT "-> " MEMORY_FMT " %s %s:%i\n",
            MEMORY_PRINT(old_size), 
            MEMORY_PRINT(new_size),
            function, file, line);

    static int64_t used_bytes = 0;
    void* new_ptr = NULL;
    if(new_size != 0)
    {
        CUDA_TEST(cudaMalloc(&new_ptr, new_size), 
            "Out of CUDA memory! Requested " MEMORY_FMT ". Using " MEMORY_FMT " / " MEMORY_FMT ". %s %s:%i", 
            MEMORY_PRINT(new_size), 
            MEMORY_PRINT(used_bytes), 
            MEMORY_PRINT(info.prop.totalGlobalMem),
            function, file, line);

        size_t min_size = MIN(old_size, new_size);
        if((flags & REALLOC_ZERO) && !(flags & REALLOC_COPY))
            CUDA_DEBUG_TEST(cudaMemset(new_ptr, 0, new_size));
        else
        {
            if(flags & REALLOC_COPY)
                CUDA_DEBUG_TEST(cudaMemcpy(new_ptr, old_ptr, min_size, cudaMemcpyDeviceToDevice));
            if(flags & REALLOC_ZERO)
                CUDA_DEBUG_TEST(cudaMemset((uint8_t*) new_ptr + min_size, 0, new_size - min_size));
        }
    }


    CUDA_DEBUG_TEST(cudaFree(old_ptr), 
        "Invalid pointer passed to cuda_realloc! %s:%i", file, line);

    used_bytes += (int64_t) new_size - (int64_t) old_size;
    assert(used_bytes >= 0);
    return new_ptr;
}

static void _cuda_realloc_in_place(void** ptr_ptr, size_t new_size, size_t old_size, int flags, const char* file, const char* function, int line)
{
    *ptr_ptr = _cuda_realloc(*ptr_ptr, new_size, old_size, flags, file, function, line);
}

#define cuda_realloc(old_ptr, new_size, old_size, flags)          _cuda_realloc(old_ptr, new_size, old_size, flags, __FILE__, __FUNCTION__, __LINE__)
#define cuda_realloc_in_place(ptr_ptr, new_size, old_size, flags) _cuda_realloc_in_place(ptr_ptr, new_size, old_size, flags, __FILE__, __FUNCTION__, __LINE__)

typedef struct Source_Info {
    int line;
    const char* function;
    const char* file;
} Source_Info;

#define SOURCE_INFO() BRACE_INIT(Source_Info){__LINE__, __FUNCTION__, __FILE__}

typedef struct Cache_Index {
    int index;
    int bucket;
} Cache_Index;

typedef struct Cache_Allocation {
    void* ptr;
    bool used;
    uint64_t generation;
    Cache_Index next;
    Source_Info source;
} Cache_Allocation;

typedef struct Size_Bucket {
    size_t bucket_size;
    int used_count;
    Cache_Index first_free;
    int allocations_capacity;
    int allocations_size;
    Cache_Allocation* allocations;
} Size_Bucket;
//Not a fully general allocator. It should however be a lot faster for our use
// case then the standard cudaMalloc
typedef struct Allocation_Cache {
    uint64_t generation;
    size_t alloced_bytes;
    size_t max_alloced_bytes;
    
    int bucket_count;
    //We store flat max because we are lazy and 256 seems like enough.
    Size_Bucket buckets[256];
} Allocation_Cache;

static inline Allocation_Cache* _global_allocation_cache() {
    thread_local static Allocation_Cache c;
    return &c;
}

typedef struct Cache_Tag {
    uint64_t generation;
    Cache_Index last;
} Cache_Tag;

static Cache_Tag cache_tag_make()
{
    Cache_Tag out = {0};
    out.generation = ++_global_allocation_cache()->generation;
    return out;
}

static void* _cache_alloc(size_t bytes, Cache_Tag* tag, Source_Info source)
{
    Allocation_Cache* cache = _global_allocation_cache();
    if(bytes == 0)
        return NULL;

    //Find correcctly sized bucket
    int bucket_i = -1;
    for(int i = 0; i < cache->bucket_count; i++)
    {
        if(cache->buckets[i].bucket_size == bytes)
        {
            bucket_i = i;
            break;
        }
    }
    if(bucket_i == -1)
    {
        bucket_i = cache->bucket_count++;
        LOG_INFO("CUDA", "Alloc cache made bucket [%i] " MEMORY_FMT, bucket_i, MEMORY_PRINT(bytes));
        TEST(cache->bucket_count < STATIC_ARRAY_SIZE(cache->buckets), "Unexepectedly high ammount of buckets");
        cache->buckets[bucket_i].bucket_size = bytes;
    }

    //Find not used allocation
    Size_Bucket* bucket = &cache->buckets[bucket_i];
    if(bucket->first_free.index <= 0)
    {
        //If is missing free slot grow slots
        if(bucket->allocations_size >= bucket->allocations_capacity)
        {
            size_t count = MAX(16, bucket->allocations_capacity*4/3 + 8);
            LOG_INFO("CUDA", "Alloc cache bucket [%i] growing slots %i -> %i", bucket_i, (int) bucket->allocations_capacity, (int) count);
            Cache_Allocation* new_data = (Cache_Allocation*) realloc(bucket->allocations, count*sizeof(Cache_Allocation));
            TEST(new_data);
            bucket->allocations_capacity = count;
            bucket->allocations = new_data;
        }

        LOG_INFO("CUDA", "Alloc cache bucket [%i] allocated " MEMORY_FMT, bucket_i, MEMORY_PRINT(bytes));
        //Fill the allocation appropriately
        int alloc_index = bucket->allocations_size ++;
        Cache_Allocation* allocation = &bucket->allocations[alloc_index];
        CUDA_TEST(cudaMalloc(&allocation->ptr, bytes));
        bucket->used_count += 1;
        cache->max_alloced_bytes += bytes;

        allocation->next = bucket->first_free;
        bucket->first_free.bucket = bucket_i;
        bucket->first_free.index = alloc_index + 1;
    }

    //Realink the allocation
    Cache_Index index = bucket->first_free;
    CHECK_BOUNDS(index.index - 1, bucket->allocations_size);

    Cache_Allocation* allocation = &bucket->allocations[index.index - 1];
    ASSERT(allocation->ptr != NULL);

    bucket->first_free = allocation->next;
    bucket->used_count += 1;

    allocation->generation = tag->generation;
    allocation->source = source;
    allocation->next = tag->last;
    allocation->used = true;
    tag->last = index;

    return allocation->ptr;
}

static void cache_free(Cache_Tag* tag)
{
    Allocation_Cache* cache = _global_allocation_cache();
    while(tag->last.index != 0)
    {
        Size_Bucket* bucket = &cache->buckets[tag->last.bucket];
        Cache_Allocation* allocation = &bucket->allocations[tag->last.index - 1];
        ASSERT(allocation->generation == tag->generation && allocation->used);
        Cache_Index curr_allocated = tag->last;
        Cache_Index next_allocated = allocation->next;
        allocation->used = false;
        allocation->next = bucket->first_free;
        bucket->first_free = curr_allocated;
        bucket->used_count -= 1;
        ASSERT(bucket->used_count >= 0);
        tag->last = next_allocated;
    }
}

#define cache_alloc(Type, count, tag_ptr) (Type*) _cache_alloc(sizeof(Type) * (size_t) (count), (tag_ptr), SOURCE_INFO())