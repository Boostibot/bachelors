
#include <cuda_runtime.h>
#include "lib/assert.h"

void _test_cuda(cudaError_t error, const char* expression, Source_Info info)
{
    if(error != cudaSuccess)
    {
        assertion_report(expression, info, "cuda failed with error %s", cudaGetErrorString(error));
        platform_trap();
        platform_abort();
    }
}

#define CUDA_TEST(status) _test_cuda((status), #status, SOURCE_INFO())
#define CUDA_ERR_AND(err) (err) != cudaSuccess ? (err) :

#include "lib/allocator.h"

typedef struct Cuda_Allocator {
    Allocator allocator;
} Cuda_Allocator;

EXPORT void* cuda_allocator_allocate(Allocator* self_, isize new_size, void* old_ptr, isize old_size, isize align, Source_Info called_from)
{
    //Cuda_Allocator* self = (Cuda_Allocator*) (void*) self_; 
    ASSERT_MSG((old_ptr != NULL) == (old_size != 0), "invalid combination of size and ptr");

    if(old_ptr)
        cudaFree(old_ptr);

    void* out = NULL;
    if(new_size > 0)
    {
        cudaError_t err = cudaMalloc(&out, new_size);
        if(err != cudaSuccess)
            LOG_ERROR("Cuda_Allocator", "cuda allocator failed with message: '%s'", cudaGetErrorString(err));
    }

    return out;
}

EXPORT Allocator_Stats cuda_allocator_get_stats(Allocator* self)
{
    Allocator_Stats stats = {0};
    stats.type_name = "Cuda_Allocator";
    stats.is_top_level = true;
    return stats;
}
EXPORT void cuda_allocator_init(Cuda_Allocator* self)
{
    self->allocator.allocate = cuda_allocator_allocate;
    self->allocator.get_stats = cuda_allocator_get_stats;
}

EXPORT void cuda_allocator_deinit(Cuda_Allocator* self)
{
    (void) self;
}

Allocator* allocator_get_cuda()
{
    static Cuda_Allocator alloc = {cuda_allocator_allocate, cuda_allocator_get_stats};
    return &alloc.allocator;
}