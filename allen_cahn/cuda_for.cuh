#pragma once
#include "cuda_util.cuh"

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

#if 0
template <typename Function>
static void cuda_for(csize from, csize to, Function func, Cuda_Launch_Params launch_params = {})
{
    Cuda_Info info = cuda_one_time_setup();
    uint block_size = 64;
    uint block_count = (uint) DIV_CEIL(to - from ,block_size);
    if(info.has_broken_driver)
        block_count = MIN(block_count, launch_params.max_block_count_for_broken_drivers);

    cuda_for_kernel<<<block_count, block_size, 0, launch_params.stream>>>(from, to-from, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
    // if(launch_params.stream != 0)
        // CUDA_DEBUG_TEST(cudaDeviceSynchronize());
}

template <typename Function>
static void cuda_for_2D(csize from_x, csize from_y, csize to_x, csize to_y, Function func, Cuda_Launch_Params launch_params = {})
{
    Cuda_Info info = cuda_one_time_setup();
    csize volume = (to_x - from_x)*(to_y - from_y);

    uint block_size = 64;
    uint block_count = (uint) MIN(DIV_CEIL(volume, block_size), UINT_MAX);
    if(info.has_broken_driver)
        block_count = MIN(block_count, launch_params.max_block_count_for_broken_drivers);

    cuda_for_2D_kernel<<<block_count, block_size>>>(from_x, to_x-from_x, from_y, to_y-from_y, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
    // if(launch_params.stream != 0)
        // CUDA_DEBUG_TEST(cudaDeviceSynchronize());
}
#else

template <typename Function>
static void cuda_for(csize from, csize to, Function func, Cuda_Launch_Params launch_params = {})
{
    static Cuda_Launch_Bounds bounds = cuda_query_launch_bounds(cuda_launch_query_from_kernel((void*) cuda_for_kernel<Function>));

    if(launch_params.preferd_block_size == 0)
        launch_params.preferd_block_size = 64;
    if(launch_params.max_block_count_for_broken_drivers == 0)
        launch_params.max_block_count_for_broken_drivers = 500;
    Cuda_Launch_Config launch = cuda_get_launch_config(to - from, bounds, launch_params);

    cuda_for_kernel<<<launch.block_count, launch.block_size, launch.dynamic_shared_memory, launch_params.stream>>>(from, to-from, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
    // if(launch_params.stream != 0)
        // CUDA_DEBUG_TEST(cudaDeviceSynchronize());
}

template <typename Function>
static void cuda_for_2D(csize from_x, csize from_y, csize to_x, csize to_y, Function func, Cuda_Launch_Params launch_params = {})
{
    static Cuda_Launch_Bounds bounds = cuda_query_launch_bounds(cuda_launch_query_from_kernel((void*) cuda_for_2D_kernel<Function>));

    csize volume = (to_x - from_x)*(to_y - from_y);
    if(launch_params.preferd_block_size == 0)
        launch_params.preferd_block_size = 64;
    if(launch_params.max_block_count_for_broken_drivers == 0)
        launch_params.max_block_count_for_broken_drivers = 500;
    Cuda_Launch_Config launch = cuda_get_launch_config(volume, bounds, launch_params);

    cuda_for_2D_kernel<<<launch.block_count, launch.block_size, launch.dynamic_shared_memory, launch_params.stream>>>(from_x, to_x-from_x, from_y, to_y-from_y, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
    // if(launch_params.stream != 0)
        // CUDA_DEBUG_TEST(cudaDeviceSynchronize());
}

#endif

#if 1

template <typename T, typename Function>
void __global__ cuda_tiled_for_kernel(const T* input, csize N, csize r, Function func, T out_of_bounds_val)
{
    // using T = decltype(gather(0));
    extern __shared__ max_align_t shared_backing[];
    T* shared = (T*) (void*) shared_backing;

    csize tile_size = blockDim.x;
    csize ti = threadIdx.x;
    for (csize bi = blockIdx.x; ; bi += gridDim.x) 
    {
        csize i_base = bi*tile_size - 2*bi*r;
        if(i_base >= N)
            break;

        csize i = i_base - r + ti;
        T val = out_of_bounds_val;
        if(0 <= i && i < N)
            val = input[i];

        shared[ti] = val;
        __syncthreads();

        if(r <= ti && ti < tile_size-r && i < N)
            func(i, ti, shared, tile_size, N, r);
            // func(i, ti, shared, tile_size);
    }
}

#include <assert.h>
void __global__ gpu_convolution1(const int* input, const int* stencil, int* output, int N, int r, int out_of_bounds_val)
{
    extern __shared__ int shared[];

    int tile_size = blockDim.x;
    int ti = threadIdx.x;
    for (int bi = blockIdx.x; ; bi += gridDim.x) 
    {
        int i_base = bi*tile_size - 2*bi*r;
        if(i_base >= N)
            break;

        int i = i_base - r + ti;
        int val = out_of_bounds_val;
        if(0 <= i && i < N)
            val = input[i];

        shared[ti] = val;
        __syncthreads();

        if(r <= ti && ti < tile_size-r && i < N)
        {
            int out = 0;
            for(int iter = -r; iter <= r; iter++)
            {
                int i_shared = iter + ti;
                int i_absolute = iter + i;
                if(0 <= i_absolute && i_absolute < N)
                    out += shared[i_shared] * stencil[iter + r];
            }

            output[i] = out;
        }
    }
}

struct Tiled_For_Functor {
    int* __restrict__ output;
    int* __restrict__ stencil;
    int ints[20];

    __forceinline__ __device__ void operator()(int i, int ti, int* __restrict__ shared, int tile_size, int N, int r) const 
    {
        int out = 0;
        for(int iter = -r; iter <= r; iter++)
        {
            int i_shared = iter + ti;
            int i_absolute = iter + i;
            if(0 <= i_absolute && i_absolute < N)
                out += shared[i_shared] * stencil[iter + r];
        }

        output[i] = out;
    }
};

void __global__ gpu_convolution2(const int* input, Tiled_For_Functor func, int N, int r, int out_of_bounds_val)
{
    extern __shared__ int shared[];
    int tile_size = blockDim.x;
    int ti = threadIdx.x;
    for (int bi = blockIdx.x; ; bi += gridDim.x) 
    {
        int i_base = bi*tile_size - 2*bi*r;
        if(i_base >= N)
            break;

        int i = i_base - r + ti;
        int val = out_of_bounds_val;
        if(0 <= i && i < N)
            val = input[i];

        shared[ti] = val;
        __syncthreads();

        if(r <= ti && ti < tile_size-r && i < N)
            func(i, ti, shared, tile_size, N, r);
    }
}


void cpu_convolution(const int* input, const int* stencil, int* output, int N, int r, int out_of_bounds_val)
{   
    for(int i = 0; i < N; i++)
    {
        int out = 0;
        for(csize iter = -r; iter <= r; iter++)
        {
            int i_absolute = iter + i;
            if(0 <= i_absolute && i_absolute < N)
                out += input[i_absolute] * stencil[iter + r];
        }

        output[i] = out;
    }
}

template <typename T, typename Function>
void cuda_tiled_for(T* data, csize N, csize r, Function func, T out_of_bounds_val = T(), cudaStream_t stream = 0)
{
    #if 0
    if(N <= 0)
        return;

    Cuda_Launch_Query query = {};
    query.block_size_to_shared_memory_factor = sizeof(T);
    query.min_block_size = WARP_SIZE;
    query.max_block_size = 128;
    query.max_blocks = 200;

    Cuda_Launch_Config launch = cuda_query_launch_config(N, query);
    if(launch.block_size == 0)
    {
        LOG_INFO("tiled_for", "couldnt find appropriate size to launch cuda_tiled_for kernel");
        return;
    }

    printf("launch: N:%i block_count:%i block_size:%i shared_memory:%i\n", N, launch.block_count, launch.block_size, launch.shared_memory);
    // template <typename Gather_Function, typename Function, Tiled_For_Variant variant>
    // void __global__ cuda_tiled_for_kernel(T* data, csize N, csize r, Function func, T out_of_bounds_val)
    cuda_tiled_for_kernel<T, Function>
        <<<launch.block_count, launch.block_size, launch.shared_memory, stream>>>(data, N, r, (Function&&) func, out_of_bounds_val);

    CUDA_DEBUG_TEST(cudaGetLastError());
    if(stream == NULL)
        CUDA_DEBUG_TEST(cudaDeviceSynchronize());
    #endif
}
#endif




void cpu_tiled_for(int* x, csize N, csize tile_radius, void (*func)(csize i, csize ti, int* tile, void* context), void* context, int out_of_bounds_val = 0, bool modular_bounds = false)
{
    int shared[64] = {0};
    csize block_size = 64;
    csize inner_block_size = block_size - 2*tile_radius;
    assert(inner_block_size > 0);

    for(csize inner_block_from = 0; inner_block_from < N; inner_block_from += inner_block_size)
    {
        csize block_from = inner_block_from - tile_radius;
        // csize block_to = block_from + block_size;

        for(csize ti = 0; ti < block_size; ti++)
        {
            csize i = ti + block_from;
            int val = out_of_bounds_val;
            if(modular_bounds)
            {
                val = x[(i + N) % N];
            }
            else
            {
                if(i >= 0 && i < N)
                    val = x[i];
            }

            shared[ti] = val;
            //sync - must be not within a conditional

            // if(i >= 0 && i < N)      
            if(i < N)      
            {
                if(ti >= tile_radius && ti < inner_block_size+tile_radius)
                    func(i, ti, shared, context);
            }
        }
    }
}


void cpu_tiled_for2(int* data, csize N, csize r, void (*func)(csize i, csize ti, int* tile, void* context), void* context, int out_of_bounds_val = 0, bool modular_bounds = false)
{
    if(N <= 0)
        return;

    // using T = decltype(gather(0));
    int shared[64] = {0};

    csize block_size = 64;
    csize inner_block_size = block_size - 2*r;

    for(csize bi = 0; ; bi += 64)
    {
        csize block_index = bi / 64;
        csize i_base = bi - r - 2*r*block_index;
        if(i_base >= N)
            break;

        for(csize ti = 0; ti < 64; ti ++)
        {
            assert(inner_block_size > 0);

            csize i = i_base + ti;
            int val = out_of_bounds_val;
            if(modular_bounds)
            {
                csize idx = MOD(i, N);
                assert(0 <= idx && idx < N);
                val = data[idx];
            }
            else
            {
                csize idx = i;
                if(0 <= idx && idx < N)
                    val = data[idx];
            }

            shared[ti] = val;
        }
        // __syncthreads();
        for(csize ti = 0; ti < 64; ti ++)
        {
            csize i = i_base + ti;
            if(ti >= r && ti < inner_block_size+r && i < N)
                func(i, ti, shared, context);
        }
    }
}

struct Test_Tiled_Context {
    int* output;
    int* stencil;
    csize range;
    csize N;
    bool modular;
};

void test_tile_for_func(csize center_i, csize center_ti, int* tile, void* context)
{
    Test_Tiled_Context* c = (Test_Tiled_Context*) context;
    assert(0 <= center_i && center_i < c->N);

    int out = 0;
    for(csize iter = -c->range; iter <= c->range; iter++)
    {
        csize ti = iter + center_ti;
        csize i = iter + center_i;
        assert(0 <= ti && ti < 64);
        if(c->modular || (0 <= i && i < c->N))
            out += tile[ti] * c->stencil[iter + c->range];
    }

    c->output[center_i] = out;
}

void test_for_func(csize center_i, int* data, void* context)
{
    Test_Tiled_Context* c = (Test_Tiled_Context*) context;
    assert(0 <= center_i && center_i < c->N);

    int out = 0;
    for(csize iter = -c->range; iter <= c->range; iter++)
    {
        csize i = iter + center_i;
        if(c->modular)
        {
            out += data[MOD(i, c->N)] * c->stencil[iter + c->range];
        }
        else if(0 <= i && i < c->N)
            out += data[i] * c->stencil[iter + c->range];
    }

    c->output[center_i] = out;
}

void print_int_array(const char* before, const int* array, csize N, const char* after)
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

template <int size>
struct Test_Big_Struct {
    int vals[size];
};

void test_tiled_for(uint64_t seed, bool printing = false)
{
    enum {MAX_N = 256};

    seed = 1714248654424358039;
    // seed = 1714248107310112759;
    csize Ns[] = {0, 1, 4, 15, 63, 64, 65, 127, 128, 129, 256, 1024, 1024*4, 1024*14, 1024*16};
    csize rs[] = {1};
    bool modulars[] = {false};

    csize max_N = 1024*1024;
    // for(csize Ni = 0; Ni < STATIC_ARRAY_SIZE(Ns); Ni++)
        // if(max_N < Ns[Ni])
            // max_N = Ns[Ni];

    int* allocation = (int*) malloc(max_N*5*sizeof(int));

    Cache_Tag tag = cache_tag_make();
    int* out_cuda = cache_alloc(int, max_N, &tag);
    int* stencil_cuda = cache_alloc(int, max_N, &tag);
    int* in_cuda = cache_alloc(int, max_N, &tag);

    int data_range = 10;
    int stencil_range = 2;

    srand(seed);
    for(csize mi = 0; mi < STATIC_ARRAY_SIZE(modulars); mi++)
    {
        for(csize Ni = 0; Ni < 1024; Ni++)
        {
            for(csize ri = 0; ri < STATIC_ARRAY_SIZE(rs); ri++)
            {
                csize N = Ni*1024;
                csize r = rs[ri];
                bool modular = modulars[mi];

                if(printing && N <= 64)
                {
                    printf("\nN:%i r:%i\n", N, r);
                }

                memset(allocation, 0x55, max_N*5*sizeof(int));
                int* data =         allocation + 0*max_N;
                int* stencil =      allocation + 1*max_N;
                int* out_regul =    allocation + 2*max_N;
                int* out_tiled =    allocation + 3*max_N;
                int* out_cpu_cuda = allocation + 4*max_N;

                for(csize i = 0; i < N; i++)
                    data[i] = i % data_range;
                    // data[i] = rand() % data_range;

                for(csize i = 0; i < 2*r+1; i++)
                    stencil[i] = 1;
                    // stencil[i] = rand() % stencil_range - stencil_range/2;

                Test_Tiled_Context context = {0};
                context.N = N;
                context.range = r;
                context.stencil = stencil;
                context.output = out_regul;
                context.modular = modular;

                for(csize i = 0; i < N; i++)
                    test_for_func(i, data, &context);

                context.output = out_tiled;
                cpu_tiled_for2(data, N, r, test_tile_for_func, &context, 0, modular);


                cudaMemcpy(in_cuda, data, max_N*sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(stencil_cuda, stencil, max_N*sizeof(int), cudaMemcpyHostToDevice);
                if(0)
                {
                    Test_Big_Struct<30> big_struct = {1, 2, 3, 4, 5};
                    // cuda_tiled_for(in_cuda, N, r, [=]SHARED(csize i, csize ti, int* __restrict__ shared, csize block_size){
                    cuda_tiled_for(in_cuda, N, r, [=]SHARED(csize i, csize ti, int* __restrict__ shared, csize block_size, csize N, csize r){
                        int big = 0;
                        for(int k = 0; k < STATIC_ARRAY_SIZE(big_struct.vals); k++)
                            big += big_struct.vals[k];

                        int out = 0;
                        for(int iter = -r; iter <= r; iter++)
                        {
                            int i_shared = iter + ti + big / 100;
                            int i_absolute = iter + i;
                            if(0 <= i_absolute && i_absolute < N)
                                out += shared[i_shared] * stencil_cuda[iter + r];
                        }

                        out_cuda[i] = out;
                    });
                }
                else
                {
                    if(N > 0)
                    {
                        Cuda_Info info = cuda_one_time_setup();
                        cudaFuncAttributes attributes = {};
                        CUDA_TEST(cudaFuncGetAttributes(&attributes, gpu_convolution2));
                        
                        Tiled_For_Functor func = {out_cuda, stencil_cuda};
                        int max_blocks_by_constant_memory = info.prop.totalConstMem/2 / (sizeof(func) + sizeof(void*));

                        int block_size = 64;
                        int block_count = DIV_CEIL(N, block_size);
                        int shared_memory = block_size*sizeof(int);
                        if(block_count > 10)
                            block_count = 10;

                        if(block_count > max_blocks_by_constant_memory)
                            block_count = max_blocks_by_constant_memory;

                        if(0)
                            gpu_convolution1<<<block_count, block_size, shared_memory>>>(in_cuda, stencil_cuda, out_cuda, N, r, 0);
                        else
                        {
                            gpu_convolution2<<<block_count, block_size, shared_memory>>>(in_cuda, func, N, r, 0);
                        }

                        printf("launch: N:%i block_size:%i block_count:%i\n", N, block_size, block_count);
                    }
                }

                cudaMemcpy(out_cpu_cuda, out_cuda, max_N*sizeof(int), cudaMemcpyDeviceToHost);

                if(printing && N <= 64)
                {
                    print_int_array("stencil:  [", stencil, 2*r+1, "]\n");
                    print_int_array("original: [", data, N, "]\n");
                    print_int_array("regular:  [", out_regul, N, "]\n");
                    print_int_array("tiled:    [", out_tiled, N, "]\n");
                    print_int_array("cuda:     [", out_cpu_cuda, N, "]\n");
                }

                for(csize i = 0; i < N; i++)
                {
                    TEST(out_regul[i] == out_tiled[i]);
                    TEST(out_regul[i] == out_cpu_cuda[i], "i:%i seed:%lli %i != %i %i", i, seed, out_regul[i], out_cpu_cuda[i], N);
                }

                // TEST(memcmp(out_regul, out_tiled, N * sizeof *out_regul) == 0);
                // TEST(memcmp(out_regul, out_cpu_cuda, N * sizeof *out_regul) == 0);
            }
        }
    }

    free(allocation);
    cache_free(&tag);
}


template <typename Gather_Function, typename Function>
static void cuda_tiled_for_2D(csize from_x, csize from_y, csize to_x, csize to_y, csize tile_rx, csize tile_ry, Gather_Function gather, Function func, int flags = 0);