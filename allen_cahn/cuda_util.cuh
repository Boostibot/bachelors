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

struct Cuda_Info {
    int device_id;
    cudaDeviceProp prop;
};

Cuda_Info cuda_one_time_setup()
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
        was_setup = true;
        CUDA_TEST(cudaSetDevice(info.device_id));

        LOG_INFO("CUDA", "Listing devices below (%d):\n", nDevices);
        for (int i = 0; i < nDevices; i++)
            LOG_INFO(">CUDA", "[%i] %s (score: %lf) %s\n", i, devices[i].name, scores[i], i == max_score_i ? "[selected]" : "");

        LOG_INFO("CUDA", "Selected %s:\n", selected.name);
        LOG_INFO("CUDA", "  Multi Processor count: %i\n", selected.multiProcessorCount);
        LOG_INFO("CUDA", "  Warp-size: %d\n", selected.warpSize);
        LOG_INFO("CUDA", "  Memory Clock Rate (MHz): %d\n", selected.memoryClockRate/1024);
        LOG_INFO("CUDA", "  Memory Bus Width (bits): %d\n", selected.memoryBusWidth);
        LOG_INFO("CUDA", "  Peak Memory Bandwidth (GB/s): %.1f\n", peak_memory[max_score_i]);
        LOG_INFO("CUDA", "  Total global memory (Gbytes) %.1f\n",(float)(selected.totalGlobalMem)/1024.0/1024.0/1024.0);
        LOG_INFO("CUDA", "  Shared memory per block (Kbytes) %.1f\n",(float)(selected.sharedMemPerBlock)/1024.0);
        LOG_INFO("CUDA", "  minor-major: %d-%d\n", selected.minor, selected.major);
        LOG_INFO("CUDA", "  Concurrent kernels: %s\n", selected.concurrentKernels ? "yes" : "no");
        LOG_INFO("CUDA", "  Concurrent computation/communication: %s\n\n",selected.deviceOverlap ? "yes" : "no");
    }

    return info;
}

enum {
    REALLOC_COPY = 1,
    REALLOC_ZERO = 2,
};


void* _cuda_realloc(void* old_ptr, size_t new_size, size_t old_size, int flags, const char* file, const char* function, int line)
{
    LOG_INFO("CUDA", "realloc " MEMORY_FMT "-> " MEMORY_FMT " %s %s:%i\n",
            MEMORY_PRINT(old_size), 
            MEMORY_PRINT(new_size),
            function, file, line);

    static int64_t used_bytes = 0;
    void* new_ptr = NULL;
    if(new_size != 0)
    {
        Cuda_Info info = cuda_one_time_setup();
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

void _cuda_realloc_in_place(void** ptr_ptr, size_t new_size, size_t old_size, int flags, const char* file, const char* function, int line)
{
    *ptr_ptr = _cuda_realloc(*ptr_ptr, new_size, old_size, flags, file, function, line);
}

#define cuda_realloc(old_ptr, new_size, old_size, flags)          _cuda_realloc(old_ptr, new_size, old_size, flags, __FILE__, __FUNCTION__, __LINE__)
#define cuda_realloc_in_place(ptr_ptr, new_size, old_size, flags) _cuda_realloc_in_place(ptr_ptr, new_size, old_size, flags, __FILE__, __FUNCTION__, __LINE__)

// #include "hash_index.h"
// #include "hash.h"

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
    //We store flat max because we are lazy and 32 seems like enough.
    Size_Bucket buckets[32];
} Allocation_Cache;

Allocation_Cache _global_allocation_cache;


typedef struct Cache_Tag {
    uint64_t generation;
    Cache_Index last;
} Cache_Tag;

Cache_Tag cache_tag_make()
{
    Cache_Tag out = {0};
    _global_allocation_cache.generation += 1;
    out.generation = _global_allocation_cache.generation;
    return out;
}

void* _cache_alloc(size_t bytes, Cache_Tag* tag, Source_Info source)
{
    Allocation_Cache* cache = &_global_allocation_cache;
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

void cache_free(Cache_Tag* tag)
{
    Allocation_Cache* cache = &_global_allocation_cache;
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