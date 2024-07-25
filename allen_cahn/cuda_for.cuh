#pragma once

//This file provids most basic paralelization primitives: paralel for loops. The basic versions are very simple
// designed to be as convenient as possible. The smallest possible example being:
// 
// int* data = ...;
// int size = 1024;
// cuda_for(0, size, [=]SHARED(csize i){
//      data[i] = data[i]*2 + 3;
// });
//
// We also provide 2D and 3D (@TODO) version.
//
//
// The other primitive defined here is "tiled for". It is an shared memory optimalization of cuda_for.
// On GPUs access of main memory is very slow. This is even more pronounced when doing several input item
// acesses per output item such as when performing covolution style operations. This also comes up in our code
// when calculating the discretized laplacian. We need to perform:
//      \laplacian_d p := (p_N + p_S + p_E + p_W - 4*p)/(h*h)
// where p_N, p_S, p_E, p_W stand for the items to the North, South, East, West of item p.
//
// To optimize we first load an etire "tile" worth of input items into shared memory and then perform repeated 
// reads there. This saves us for the operation above 4 global memory acesses per input item. A big optimalization.
//
// The implementation is a bit more complex due to the need to handle what happens at the borders of the tile. 
// Of course we cannot index outside the tile as that would result in bad memory acesses. We solve this by
// only performing the useful operation on the interior of the tile. The size of such safety boarded we donet 
// 'r'. r corresponds the the maxmimum offset from the output item to the input item. In our laplacian case it is 1.
// Its worth noting that r=0 makes sense and is useful for things like matrix matrix multiply.
// The complete operation is depicted on the diagram below:
//
//                       -r        0                       tn-r     tn                 
//                                           
//                        <-------------tile size (tn)--------------->
//                        <---r----><-----inner tile-------><---r---->
//    global mem          |--------|------------------------|--------|
//
//                        vvvvvvvvvvvvvvv gather func vvvvvvvvvvvvvvvv
//
//    shared mem          |--------|------------------------|--------|
//
//                                 vvvvvvv user func vvvvvvvv
//
//    global mem                   |------------------------|
//
// Here we can see that we load values from the global memory using a custom provided gather func. This function can 
// for example compress the data, pad the data with zeros, provide periodic boundary conditions etc. Then we
// take the output of gather func in shared memory and feed it just for the inner tile into the user func which performs 
// the useful computation. 
//
// As is apparent from the diagram above not all threads are used for execution of the user func. Only about
// 30*30 out of 32*32 threads (= 87%) are used. Because of this for copute intensive operations tintermideate values 
// should be saved to an auxiliary array and the actual useful result be obtained from launching a new kernel 
// (which now uses 100% of the availibel threads) over the intermediate array.
//
// Example of 2D cuda_tiled_for used to calculate the laplacian as defined above:
//
// float* input = NULL;
// float* output = NULL;
// csize size_x = 1024;
// csize size_y = 1024;
// float h = 1/size_x;

// // Declare the value r in both directions to be 1 (at compile time!) through
// // the template arguments
// cuda_tiled_for_2D_bound<1, 1>(input, size_x, size_y, 
//     [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, const int* shared){
//         // No need to worry about overread here. 
//         // The cuda_tiled_for_2D_bound loads 0 (configurable) to shared 
//         // memory when loading out of bounds. 
//         // Because we are only inside the inner region of the tile we dont need
//         // to worry about reading out of bounds.
//         float p_E = shared[tx+1 + ty*tile_size_x];
//         float p_W = shared[tx-1 + ty*tile_size_x];
//         float p_N = shared[tx   + (ty+1)*tile_size_x];
//         float p_S = shared[tx   + (ty-1)*tile_size_x];
//         float p = shared[tx + ty*tile_size_x];

//         output[x + y*size_x] = (p_N + p_S + p_E + p_W - 4*p)/(h*h);
//     }
// );

#include "cuda_util.cuh"
#include "cuda_device.cuh"
#include "cuda_alloc.cuh"

template <typename Function>
static __global__ void cuda_for_kernel(csize from, csize item_count, Function func)
{
    for (csize i = blockIdx.x * blockDim.x + threadIdx.x; i < item_count; i += blockDim.x * gridDim.x) 
        func(from + i);
}

template <typename Function>
static __global__ void cuda_for_2D_kernel(csize from_x, csize x_size, csize from_y, csize y_size, Function func)
{
    for (csize y = blockIdx.y * blockDim.y + threadIdx.y; y < y_size; y += blockDim.y * gridDim.y) 
        for (csize x = blockIdx.x * blockDim.x + threadIdx.x; x < x_size; x += blockDim.x * gridDim.x) 
            func(x + from_x, y + from_y);
}

template <typename Function>
static void cuda_for(csize from, csize to, Function func, Cuda_Launch_Params launch_params = {})
{
    static Cuda_Launch_Bounds bounds = cuda_get_launch_bounds(cuda_constraints_launch_constraints((void*) cuda_for_kernel<Function>));

    if(launch_params.preferd_block_size == 0)
        launch_params.preferd_block_size = 64;
    Cuda_Launch_Config launch = cuda_get_launch_config(to - from, bounds, launch_params);

    CUDA_DEBUG_TEST(cudaGetLastError());
    cuda_for_kernel<<<launch.block_count, launch.block_size, launch.dynamic_shared_memory, launch_params.stream>>>(from, to-from, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
}

template <typename Function>
static void cuda_for_2D(csize from_x, csize from_y, csize to_x, csize to_y, Function func, Cuda_Launch_Params launch_params = {})
{
    static Cuda_Launch_Bounds bounds = cuda_get_launch_bounds(cuda_constraints_launch_constraints((void*) cuda_for_2D_kernel<Function>));

    csize volume = (to_x - from_x)*(to_y - from_y);
    if(launch_params.preferd_block_size == 0)
        launch_params.preferd_block_size = 64;
    Cuda_Launch_Config launch = cuda_get_launch_config(volume, bounds, launch_params);

    cuda_for_2D_kernel<<<launch.block_count, launch.block_size, launch.dynamic_shared_memory, launch_params.stream>>>(from_x, to_x-from_x, from_y, to_y-from_y, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
}

//========================================== TILED FOR =====================================
enum {
    TILED_FOR_DYNAMIC_RANGE = -1 //Set the template arguments to this value to be able to specify the 'r' through function arguments 
};

template <typename T, typename Gather, typename Function, csize static_r>
static void __global__ cuda_tiled_for_kernel(csize i_offset, csize N, csize dynamic_r, Gather gather, Function func)
{
    extern __shared__ max_align_t shared_backing[];
    T* shared = (T*) (void*) shared_backing;

    csize r = 0;
    if constexpr(static_r != TILED_FOR_DYNAMIC_RANGE)
        r = static_r;
    else
        r = dynamic_r;

    csize tile_size = blockDim.x;
    csize ti = threadIdx.x;
    for (csize bi = blockIdx.x; ; bi += gridDim.x) 
    {
        csize i_base = bi*tile_size - 2*bi*r;
        if(i_base >= N)
            break;

        csize i = i_base - r + ti;
        T val = gather(i, N, r); //gather is not offset!

        shared[ti] = val;
        __syncthreads();

        if(r <= ti && ti < tile_size-r && i < N)
            func(i + i_offset, ti, tile_size, shared);
            
        __syncthreads();
    }
}

template <csize static_r, typename T, typename Function, typename Gather>
static void cuda_tiled_for(csize from_i, csize to_i, Gather gather, Function func, csize dynamic_r = 0, Cuda_Launch_Params launch_params = {})
{
    static Cuda_Launch_Bounds bounds = {};
    static Cuda_Launch_Constraints constraints = {};
    if(bounds.max_block_size == 0)
    {
        constraints = cuda_constraints_launch_constraints((void*) cuda_tiled_for_kernel<T, Gather, Function, static_r>);
        constraints.used_shared_memory_per_thread = sizeof(T);
        bounds = cuda_get_launch_bounds(constraints);
    }

    csize r = dynamic_r;
    if constexpr(static_r != TILED_FOR_DYNAMIC_RANGE)
        r = static_r;

    csize N = to_i - from_i;
    if(N <= 0)
        return;

    Cuda_Launch_Config launch = cuda_get_launch_config(N, bounds, launch_params);
    if(launch.block_size == 0)
    {
        LOG_ERROR("kernel", "couldnt find appropriate config parameters to launch '%s' with N:%lli r:%lli", __FUNCTION__, (lli)N, (lli)r);
        return;
    }

    if(0) {
        LOG_DEBUG("kernel", "cuda_tiled_for launch: N:%i block_count:%i block_size:%i dynamic_shared_memory:%i\n", 
            N, launch.block_count, launch.block_size, launch.dynamic_shared_memory);
    }

    cuda_tiled_for_kernel<T, Gather, Function, static_r>
        <<<launch.block_count, launch.block_size, launch.dynamic_shared_memory, launch_params.stream>>>(from_i, N, dynamic_r, (Gather&&) gather, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
}


template <csize static_r, typename T, typename Function>
static void cuda_tiled_for_bound(const T* data, csize from_i, csize to_i, Function func, csize dynamic_r = 0, T out_of_bounds_val = T(), Cuda_Launch_Params launch_params = {})
{
    //Gather is not offset!
    const T* offset_data = data + from_i;
    cuda_tiled_for<static_r, T, Function>(from_i, to_i, [=]SHARED(csize i, csize N, csize r){
        if(0 <= i && i < N)
            return offset_data[i];
        else
            return out_of_bounds_val;
    }, (Function&&) func, dynamic_r, launch_params);
}

template <typename T, typename Gather, typename Function, csize static_rx, csize static_ry>
static void __global__ cuda_tiled_for_2D_kernel(csize from_x, csize from_y, csize nx, csize ny, csize dynamic_rx, csize dynamic_ry, Gather gather, Function func)
{
    extern __shared__ max_align_t shared_backing[];
    T* shared = (T*) (void*) shared_backing;

    csize rx = 0;
    if constexpr(static_rx != TILED_FOR_DYNAMIC_RANGE)
        rx = static_rx;
    else
        rx = dynamic_rx;

    csize ry = 0;
    if constexpr(static_ry != TILED_FOR_DYNAMIC_RANGE)
        ry = static_ry;
    else
        ry = dynamic_ry;

    csize tile_size_x = blockDim.x;
    csize tile_size_y = blockDim.y;
    csize tx = threadIdx.x;
    csize ty = threadIdx.y;

    for (csize by = blockIdx.y; ; by += gridDim.y) 
    {
        csize base_y = by*(tile_size_y - 2*ry);
        if(base_y >= ny)
            break;

        for (csize bx = blockIdx.x; ; bx += gridDim.x) 
        {
            csize base_x = bx*(tile_size_x - 2*rx);
            if(base_x >= nx)
                break;

            csize y = base_y - ry + ty;
            csize x = base_x - rx + tx;
            T val = gather(x, y, nx, ny, rx, ry); //gather is not offset!

            shared[tx + ty*tile_size_x] = val;
            __syncthreads();

            if(rx <= tx && tx < tile_size_x-rx && x < nx)
                if(ry <= ty && ty < tile_size_y-ry && y < ny)
                    func(from_x+x, from_y+y, tx, ty, tile_size_x, tile_size_y, shared);

            __syncthreads();
        }
    }
}

template <csize static_rx, csize static_ry, typename T, typename Function, typename Gather>
static void cuda_tiled_for_2D(csize from_x, csize from_y, csize to_x, csize to_y, Gather gather, Function func, csize dynamic_rx = 0, csize dynamic_ry = 0, Cuda_Launch_Params launch_params = {})
{
    static Cuda_Launch_Bounds bounds = {};
    static Cuda_Launch_Constraints constraints = {};
    if(bounds.max_block_size == 0)
    {
        constraints = cuda_constraints_launch_constraints((void*) cuda_tiled_for_2D_kernel<T, Gather, Function, static_rx, static_ry>);
        constraints.used_shared_memory_per_thread = sizeof(T);
        bounds = cuda_get_launch_bounds(constraints);
    }

    csize nx = to_x - from_x;
    csize ny = to_y - from_y;
    if(nx <= 0 || ny <= 0)
        return;

    csize rx = dynamic_rx;
    if constexpr(static_rx != TILED_FOR_DYNAMIC_RANGE)
        rx = static_rx;

    csize ry = dynamic_ry;
    if constexpr(static_ry != TILED_FOR_DYNAMIC_RANGE)
        ry = static_ry;

    csize volume = nx*ny;
    launch_params.preferd_block_size = 256;
    Cuda_Launch_Config launch = cuda_get_launch_config(volume, bounds, launch_params);

    dim3 block_size3 = {1, 1, 1};

    if(rx == ry)
    {
        block_size3.x = (uint) round(sqrt(launch.block_size));
        block_size3.y = (uint) launch.block_size / block_size3.x;
    }
    else
    {
        block_size3.x = (uint) ROUND_UP(2*rx+1, WARP_SIZE);
        block_size3.y = (uint) launch.block_size / block_size3.x;
        if((int) block_size3.y < 2*ry+1)
        {
            LOG_ERROR("kernel", "couldnt find appropriate config parameters to launch '%s' with nx:%lli ny:%lli rx:%lli ry:%lli", __FUNCTION__, (lli)nx, (lli)ny, (lli)rx, (lli)ry);
            return;
        }
    }

    if(0) {
        LOG_DEBUG("kernel", "cuda_tiled_for_2D launch: N:{%lli %lli} block_count:%i block_size:{%i %i} dynamic_shared_memory:%i\n", 
            (lli)nx, (lli)ny, launch.block_count, block_size3.x, block_size3.y, launch.dynamic_shared_memory);
    }

    cuda_tiled_for_2D_kernel<T, Gather, Function, static_rx, static_ry>
        <<<launch.block_count, block_size3, launch.dynamic_shared_memory, launch_params.stream>>>(from_x, from_y, nx, ny, dynamic_rx, dynamic_ry, (Gather&&) gather, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
}

template <csize static_rx, csize static_ry, typename T, typename Function>
static void cuda_tiled_for_2D_bound(const T* data, csize data_width, csize from_x, csize from_y, csize to_x, csize to_y, Function func, csize dynamic_rx = 0, csize dynamic_ry = 0, T out_of_bounds_val = T(), Cuda_Launch_Params launch_params = {})
{
    //Gather is not offset!
    const T* offset_data = data + (from_x + from_y*data_width);
    cuda_tiled_for_2D<static_rx, static_ry, T, Function>(from_x, from_y, to_x, to_y, 
        [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry){
            if(0 <= x && x < nx && 0 <= y && y < ny)
                return offset_data[x + y*data_width];
            else
                return out_of_bounds_val;
        }, (Function&&) func, dynamic_rx, dynamic_ry, launch_params);
}

//================================================ TESTS ====================================================================
#if (defined(TEST_CUDA_ALL) || defined(TEST_CUDA_FOR)) && !defined(TEST_CUDA_FOR_IMPL)
#define TEST_CUDA_FOR_IMPL

#define DUMP_INT(x) printf(#x":%i \t%s:%i\n", (x), __FILE__, __LINE__)
#define _CUDA_HERE(fmt, ...) ((threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) ? printf("> %-20s %20s:%-4i " fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__) : 0)
#define CUDA_HERE(...) _CUDA_HERE("" __VA_ARGS__)

static __host__ __device__ void print_int_array(const char* before, const int* array, csize N, const char* after)
{
    printf("%s", before);
    for(csize i = 0; i < N; i++)
    {
        if(i == 0)
            printf("%3i", array[i]);
        else
            printf(", %3i", array[i]);
    }
    printf("%s", after);
}

static __host__ __device__ void print_int_array_2d(const char* before, const int* array, csize nx, csize ny, const char* after)
{
    printf("%s", before);
    if(ny > 0)
        printf("\n");
    for(csize y = 0; y < ny; y++)
    {
        printf("   ");
        for(csize x = 0; x < nx; x++)
        {
            if(x == 0)
                printf("%3i", array[x + y*nx]);
            else
                printf(", %3i", array[x + y*nx]);
        }
        printf("\n");
    }
    printf("%s", after);
}


#define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION _Pragma( "nv_diag_suppress 177" ) _Pragma( "nv_diag_suppress 550" )
#define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  _Pragma( "nv_diag_default 177" ) _Pragma( "nv_diag_suppress 550" )
static void cpu_convolution(const int* input, const int* stencil, int* output, csize N, csize r, int out_of_bounds_val)
{   
    csize sx = 2*r + 1;
    USE_VARIABLE(sx);
    for(csize i = 0; i < N; i++)
    {
        int out = 0;
        for(csize iter = -r; iter <= r; iter++)
        {
            csize i_absolute = iter + i;
            if(0 <= i_absolute && i_absolute < N)
            {
                CHECK_BOUNDS(iter + r, sx);
                CHECK_BOUNDS(i_absolute, N);
                out += input[i_absolute] * stencil[iter + r];
            }
        }

        output[i] = out;
    }
}

static void cpu_convolution_2D(const int* input, const int* stencil, int* output, csize nx, csize ny, csize rx, csize ry, int out_of_bounds_val)
{   
    csize sx = 2*rx + 1;
    csize sy = 2*ry + 1;

    USE_VARIABLE(sx);
    USE_VARIABLE(sy);
    for(csize y = 0; y < ny; y++)
        for(csize x = 0; x < nx; x++)
        {
            int out = 0;
            for(csize iter_y = -ry; iter_y <= ry; iter_y++)
                for(csize iter_x = -rx; iter_x <= rx; iter_x++)
                {
                    csize x_absolute = x + iter_x;
                    csize y_absolute = y + iter_y;

                    if(0 <= x_absolute && x_absolute < nx)
                        if(0 <= y_absolute && y_absolute < ny)
                        {
                            csize i_absolute = x_absolute + y_absolute*nx;
                            csize i_stencil = iter_x + rx + (iter_y + ry)*sx;
                            CHECK_BOUNDS(i_absolute, nx*ny);
                            CHECK_BOUNDS(i_stencil, sx*sy);
                            out += input[i_absolute] * stencil[i_stencil];
                        }
                }

            CHECK_BOUNDS(x + y*nx, nx*ny);
            output[x + y*nx] = out;
        }
}

static void test_tiled_for(uint64_t seed)
{
    csize Ns[] = {
        0, 1, 4, 15, 63, 64, 65, 127, 128, 129, 256, 1024 - 1, 1024, 
        1024*4, 1024*14, 1024*16, 1024*20, 1024*32, 1024*128, 1024*256, 1024*512, 
        1024*700, 1024*900, 1024*1024 - 1, 1024*1024
    };
    csize rs[] = {0, 1, 2, 3, 10, 15};

    csize max_N = 0;
    for(csize Ni = 0; Ni < (csize) ARRAY_LEN(Ns); Ni++)
        if(max_N < Ns[Ni])
            max_N = Ns[Ni];

    size_t max_N_bytes = (size_t) max_N*sizeof(int);
    int* allocation = (int*) malloc(max_N_bytes*4);

    Cache_Tag tag = cache_tag_make();
    int* gpu_out = cache_alloc(int, max_N, &tag);
    int* gpu_stencil = cache_alloc(int, max_N, &tag);
    int* gpu_in = cache_alloc(int, max_N, &tag);

    int input_range = 100;
    int stencil_range = 10;
    srand(seed);
    for(csize Ni = 0; Ni < (csize) ARRAY_LEN(Ns); Ni++)
    {
        for(csize ri = 0; ri < (csize) ARRAY_LEN(rs); ri++)
        {
            csize N = Ns[Ni];
            csize r = rs[ri];
            csize S = 2*r + 1;

            LOG_INFO("kernel", "test_tiled_for: N:%i r:%i\n", N, r);

            memset(allocation, 0x55, max_N_bytes*4);
            cudaMemset(gpu_out, 0x55, max_N_bytes);

            int* cpu_in =           allocation + 0*max_N;
            int* cpu_stencil =      allocation + 1*max_N;
            int* cpu_out =          allocation + 2*max_N;
            int* cpu_out_cuda =     allocation + 3*max_N;

            for(csize i = 0; i < N; i++)
                cpu_in[i] = rand() % input_range;

            for(csize i = 0; i < S; i++)
                cpu_stencil[i] = rand() % stencil_range - stencil_range/2;

            cpu_convolution(cpu_in, cpu_stencil, cpu_out, N, r, 0);

            cudaMemcpy(gpu_in, cpu_in, max_N_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_stencil, cpu_stencil, max_N_bytes, cudaMemcpyHostToDevice);

            cuda_tiled_for_bound<TILED_FOR_DYNAMIC_RANGE>(gpu_in, 0, N, [=]SHARED(csize i, csize ti, csize block_size, int* __restrict__ shared){
                int out = 0;
                csize S = 2*r + 1;
                USE_VARIABLE(S);
                for(csize iter = -r; iter <= r; iter++)
                {
                    csize i_shared = iter + ti;
                    csize i_absolute = iter + i;
                    assert(0 <= i_shared && i_shared < block_size);
                    assert(0 <= iter + r && iter + r < S);
                    if(0 <= i_absolute && i_absolute < N)
                        out += shared[i_shared] * gpu_stencil[iter + r];
                }

                assert(0 <= i && i < N);
                gpu_out[i] = out;
            }, r);

            cudaMemcpy(cpu_out_cuda, gpu_out, max_N_bytes, cudaMemcpyDeviceToHost);

            for(csize i = 0; i < N; i++)
                TEST(cpu_out[i] == cpu_out_cuda[i], 
                    "test_tiled_for failed! N:%lli i:%lli seed:%lli TEST(%i == %i)", 
                    (lli)N, (lli)i, (lli)seed, cpu_out[i], cpu_out_cuda[i]);
        }
    }

    free(allocation);
    cache_free(&tag);

    LOG_OKAY("kernel", "test_tiled_for: success!");
}

static void test_tiled_for_2D(uint64_t seed)
{
    csize ns[] = {0, 1, 15, 63, 64, 65, 127, 128, 129, 256, 1023, 1024};
    csize rs[] = {0, 1, 2, 3};

    csize max_N = 0;
    for(csize Ni = 0; Ni < (csize) ARRAY_LEN(ns); Ni++)
        if(max_N < ns[Ni])
            max_N = ns[Ni];

    max_N = max_N*max_N;
    size_t max_N_bytes = (size_t)max_N*sizeof(int);
    int* allocation = (int*) malloc(max_N_bytes*4);

    Cache_Tag tag = cache_tag_make();
    int* gpu_out = cache_alloc(int, max_N, &tag);
    int* gpu_stencil = cache_alloc(int, max_N, &tag);
    int* gpu_in = cache_alloc(int, max_N, &tag);

    int input_range = 100;
    int stencil_range = 10;

    srand(seed);
    for(csize niy = 0; niy < (csize) ARRAY_LEN(ns); niy++)
        for(csize nix = 0; nix < (csize) ARRAY_LEN(ns); nix++)
            for(csize riy = 0; riy < (csize) ARRAY_LEN(rs); riy++)
                for(csize rix = 0; rix < (csize) ARRAY_LEN(rs); rix++)
                {
                    csize nx = ns[nix];
                    csize ny = ns[niy];
                    csize rx = rs[rix];
                    csize ry = rs[riy];

                    csize sx = 2*rx+1;
                    csize sy = 2*ry+1;

                    size_t N_bytes = (size_t) (nx*ny)*sizeof(int);
                    
                    LOG_INFO("kernel", "test_tiled_for_2D: nx:%i ny:%i rx:%i ry:%i\n", nx, ny, rx, ry);

                    int* cpu_in =           allocation + 0*max_N;
                    int* cpu_stencil =      allocation + 1*max_N;
                    int* cpu_out =          allocation + 2*max_N;
                    int* cpu_out_cuda =     allocation + 3*max_N;

                    memset(cpu_out, 0x55, N_bytes);
                    CUDA_DEBUG_TEST(cudaMemset(gpu_out, 0x55, N_bytes));

                    for(csize i = 0; i < nx*ny; i++)
                        cpu_in[i] = rand() % input_range;

                    for(csize i = 0; i < sx*sy; i++)
                        cpu_stencil[i] = rand() % stencil_range - stencil_range/2;

                    cpu_convolution_2D(cpu_in, cpu_stencil, cpu_out, nx, ny, rx, ry, 0);

                    CUDA_DEBUG_TEST(cudaMemcpy(gpu_in, cpu_in, N_bytes, cudaMemcpyHostToDevice));
                    CUDA_DEBUG_TEST(cudaMemcpy(gpu_stencil, cpu_stencil, (size_t) (sx*sy)*sizeof(int), cudaMemcpyHostToDevice));

                    cuda_tiled_for_2D_bound<TILED_FOR_DYNAMIC_RANGE, TILED_FOR_DYNAMIC_RANGE>(gpu_in, nx, 0, 0, nx, ny, 
                        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, int* __restrict__ shared){
                            int out = 0;
                            for(csize ix = -rx; ix <= rx; ix++)
                                for(csize iy = -ry; iy <= ry; iy++)
                                {
                                    csize absolute_x = ix + x;
                                    csize absolute_y = iy + y;

                                    assert(0 <= ix+tx && ix+tx <= tile_size_x);
                                    assert(0 <= iy+ty && iy+ty <= tile_size_y);
                                    
                                    if(0 <= absolute_x && absolute_x < nx)
                                        if(0 <= absolute_y && absolute_y < ny)
                                        {
                                            csize shared_i = (ix+tx) + (iy+ty)*tile_size_x;
                                            assert(0 <= shared_i && shared_i < tile_size_x*tile_size_y);
                                            out += shared[shared_i] * gpu_stencil[ix+rx + (iy+ry)*sx];
                                        }
                                }

                            assert(0 <= x && x < nx);
                            assert(0 <= y && y < ny);
                            gpu_out[x+y*nx] = out;
                        }, rx, ry);
                    
                    CUDA_DEBUG_TEST(cudaMemcpy(cpu_out_cuda, gpu_out, N_bytes, cudaMemcpyDeviceToHost));

                    for(csize x = 0; x < nx; x++)
                        for(csize y = 0; y < ny; y++)
                        {
                            csize i = x + y*nx;
                            TEST(cpu_out[i] == cpu_out_cuda[i], 
                                "test_tiled_for_2D failed! nx:%lli ny:%lli seed:%lli x:%lli y:%lli TEST(%i == %i)", 
                                (lli)nx, (lli)ny, (lli)seed, (lli)x, (lli)y, cpu_out[i], cpu_out_cuda[i]);
                        }
                }

    free(allocation);
    cache_free(&tag);

    LOG_OKAY("kernel", "test_tiled_for_2D: success!");
}

#endif