#pragma once

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

// typedef int64_t csize;
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

//prevents unused variable type warnings messages on nvcc
#define USE_VARIABLE(x) if(&(x));

#define SHARED __host__ __device__

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
