#pragma once

// This file creates a system for fast "cache allocation" of gpu memory in a manner similar to an arena. 
// It does this by keeping track of the allocated blocks blocks by the system and placing these blocks in a simple free list
// bucketed by size. When the caller asks for memory we find the appropriate sized bucket and return a free block. Only when no such
// block is found we alloc more memory.
// 
// The interface uses a construct called Cache_Tag through which all cache allocations are made. When no longer needed all the allocations
// can be freed at once by freeing this tag. Internally Cache_Tag is simply a handle to a linked list of blocks 
// (we reuse the links from the free list, thus this is interely free).
//
// @TODO: remove cuda_realloc!
// Lastly we declare a wrapper around cudaMalloc/cudaFree called cuda_realloc which prints all allocations made. We will switch away from this system shortly.
//
// Hopefully in the future I will figure out how to do proper growing arena on the gpu as the current system does not allow at all for merging of blocks. 
// As such once we allocate a block of lets say 375 bytes we can only ever acess that memory if we again allocate 375 bytes. So far this hasnt been much 
// of a problem however in the future the constaints this allocator happens to exist may change. 

#include "cuda_device.cuh"

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

typedef struct Cache_Tag {
    uint64_t generation;
    Cache_Index last;
} Cache_Tag;

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
        LOG_INFO("CUDA", "Alloc cache made bucket [%i] %s", bucket_i, format_bytes((size_t) bytes).str);
        TEST(cache->bucket_count < (int) ARRAY_LEN(cache->buckets), "Unexepectedly high ammount of buckets");
        cache->buckets[bucket_i].bucket_size = bytes;
    }

    //Find not used allocation
    Size_Bucket* bucket = &cache->buckets[bucket_i];
    if(bucket->first_free.index <= 0)
    {
        //If is missing free slot grow slots
        if(bucket->allocations_size >= bucket->allocations_capacity)
        {
            int64_t count = MAX(16, bucket->allocations_capacity*4/3 + 8);
            LOG_INFO("CUDA", "Alloc cache bucket [%i] growing slots %i -> %i", bucket_i, (int) bucket->allocations_capacity, (int) count);
            Cache_Allocation* new_data = (Cache_Allocation*) realloc(bucket->allocations, (size_t) count*sizeof(Cache_Allocation));
            TEST(new_data);
            bucket->allocations_capacity = count;
            bucket->allocations = new_data;
        }

        LOG_INFO("CUDA", "Alloc cache bucket [%i] allocated %s", bucket_i, format_bytes((size_t) bytes).str);
        //Fill the allocation appropriately
        int alloc_index = bucket->allocations_size ++;
        Cache_Allocation* allocation = &bucket->allocations[alloc_index];
        CUDA_TEST(cudaMalloc(&allocation->ptr, (size_t) bytes));
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


enum {
    REALLOC_COPY = 1,
    REALLOC_ZERO = 2,
};

static void* _cuda_realloc(void* old_ptr, size_t new_size, size_t old_size, int flags, const char* file, const char* function, int line)
{
    Cuda_Info info = cuda_one_time_setup();
    LOG_INFO("CUDA", "realloc %s-> %s %s %s:%i\n",
            format_bytes((size_t) old_size).str, 
            format_bytes((size_t) new_size).str,
            function, file, line);

    static size_t used_bytes = 0;
    void* new_ptr = NULL;
    if(new_size != 0)
    {
        CUDA_TEST(cudaMalloc(&new_ptr, (size_t) new_size), 
            "Out of CUDA memory! Requested %s. Using %s / %s. %s %s:%i", 
            format_bytes((size_t) new_size).str, 
            format_bytes((size_t) used_bytes).str, 
            format_bytes((size_t) info.prop.totalGlobalMem).str,
            function, file, line);

        size_t min_size = MIN(old_size, new_size);
        if((flags & REALLOC_ZERO) && !(flags & REALLOC_COPY))
            CUDA_DEBUG_TEST(cudaMemset(new_ptr, 0, (size_t)new_size));
        else
        {
            if(flags & REALLOC_COPY)
                CUDA_DEBUG_TEST(cudaMemcpy(new_ptr, old_ptr, (size_t) min_size, cudaMemcpyDeviceToDevice));
            if(flags & REALLOC_ZERO)
                CUDA_DEBUG_TEST(cudaMemset((uint8_t*) new_ptr + min_size, 0, (size_t) (new_size - min_size)));
        }
    }

    CUDA_DEBUG_TEST(cudaFree(old_ptr), 
        "Invalid pointer passed to cuda_realloc! %s:%i", file, line);

    used_bytes += new_size - old_size;
    return new_ptr;
}

static void _cuda_realloc_in_place(void** ptr_ptr, size_t new_size, size_t old_size, int flags, const char* file, const char* function, int line)
{
    *ptr_ptr = _cuda_realloc(*ptr_ptr, new_size, old_size, flags, file, function, line);
}

#define cuda_realloc(old_ptr, new_size, old_size, flags)          _cuda_realloc(old_ptr, new_size, old_size, flags, __FILE__, __FUNCTION__, __LINE__)
#define cuda_realloc_in_place(ptr_ptr, new_size, old_size, flags) _cuda_realloc_in_place((void**) ptr_ptr, new_size, old_size, flags, __FILE__, __FUNCTION__, __LINE__)
