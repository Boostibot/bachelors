#pragma once
#include "cuda_util.cuh"

enum Cuda_For_Flags {
    CUDA_FOR_NONE = 0,
    CUDA_FOR_ASYNC = 1,
};

template <typename Function>
static __global__ void _kernel_cuda_for_each(isize from, isize item_count, Function func)
{
    for (isize i = blockIdx.x * blockDim.x + threadIdx.x; i < item_count; i += blockDim.x * gridDim.x) 
        func(from + i);
}

template <typename Function>
static void cuda_for(isize from, isize to, Function func, int flags = 0)
{
    Cuda_Info info = cuda_one_time_setup();
    uint block_size = 256;
    uint block_count = (uint) MIN((to - from)/block_size, UINT_MAX);

    _kernel_cuda_for_each<<<block_count, block_size>>>(from, to-from, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
    if((flags & CUDA_FOR_ASYNC) == 0)
        CUDA_DEBUG_TEST(cudaDeviceSynchronize());
}


template <typename Function>
static __global__ void _kernel_cuda_for_each_2D(isize from_x, isize x_size, isize from_y, isize y_size, Function func)
{
    //@TODO: Whats the optimal loop order? First x or y?
    for (isize y = blockIdx.y * blockDim.y + threadIdx.y; y < y_size; y += blockDim.y * gridDim.y) 
        for (isize x = blockIdx.x * blockDim.x + threadIdx.x; x < x_size; x += blockDim.x * gridDim.x) 
            func(x + from_x, y + from_y);
}

template <typename Function>
static void cuda_for_2D(isize from_x, isize from_y, isize to_x, isize to_y, Function func, int flags = 0)
{
    Cuda_Info info = cuda_one_time_setup();
    isize volume = (to_x - from_x)*(to_y - from_y);

    uint block_size = 256;
    uint block_count = (uint) MIN(volume/block_size, UINT_MAX);

    _kernel_cuda_for_each_2D<<<block_count, block_size>>>(from_x, to_x-from_x, from_y, to_y-from_y, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
    if((flags & CUDA_FOR_ASYNC) == 0)
        CUDA_DEBUG_TEST(cudaDeviceSynchronize());
}