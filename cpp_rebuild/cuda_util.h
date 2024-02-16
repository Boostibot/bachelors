#pragma once

#include <cuda_runtime.h>
#include "assert.h"

static void _test_cuda(cudaError_t error, const char* expression, int line, const char* file, const char* function)
{
    if(error != cudaSuccess)
    {
        assertion_report(expression, line, file, function, "cuda failed with error %s", cudaGetErrorString(error));
        platform_debug_break();
        abort();
    }
}

#define CUDA_TEST(status) _test_cuda((status), #status,  __LINE__, __FILE__, __FUNCTION__)
#define CUDA_ERR_AND(err) (err) != cudaSuccess ? (err) :