#include <cuda_runtime.h>
#include "lib/assert.h"

void _test_cuda(cudaError_t error, const char* expression, Source_Info info)
{
    if(error != cudaSuccess)
    {
        assertion_report(expression, info, "cuda failed with error %s", cudaGetErrorString(error));
        platform_trap();
        abort();
    }
}

#define CUDA_TEST(status) _test_cuda((status), #status, SOURCE_INFO())
#define CUDA_ERR_AND(err) (err) != cudaSuccess ? (err) :