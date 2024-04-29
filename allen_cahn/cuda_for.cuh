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

template <typename Function>
static void cuda_for(csize from, csize to, Function func, Cuda_Launch_Params launch_params = {})
{
    static Cuda_Launch_Bounds bounds = cuda_query_launch_bounds(cuda_launch_query_from_kernel((void*) cuda_for_kernel<Function>));

    if(launch_params.preferd_block_size == 0)
        launch_params.preferd_block_size = 64;
    Cuda_Launch_Config launch = cuda_get_launch_config(to - from, bounds, launch_params);

    cuda_for_kernel<<<launch.block_count, launch.block_size, launch.dynamic_shared_memory, launch_params.stream>>>(from, to-from, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
}

template <typename Function>
static void cuda_for_2D(csize from_x, csize from_y, csize to_x, csize to_y, Function func, Cuda_Launch_Params launch_params = {})
{
    static Cuda_Launch_Bounds bounds = cuda_query_launch_bounds(cuda_launch_query_from_kernel((void*) cuda_for_2D_kernel<Function>));

    csize volume = (to_x - from_x)*(to_y - from_y);
    if(launch_params.preferd_block_size == 0)
        launch_params.preferd_block_size = 64;
    Cuda_Launch_Config launch = cuda_get_launch_config(volume, bounds, launch_params);

    cuda_for_2D_kernel<<<launch.block_count, launch.block_size, launch.dynamic_shared_memory, launch_params.stream>>>(from_x, to_x-from_x, from_y, to_y-from_y, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
}

template <typename T, typename Gather, typename Function, csize static_r = -1>
static void __global__ cuda_tiled_for_kernel(csize N, csize dynamic_r, Gather gather, Function func)
{
    extern __shared__ max_align_t shared_backing[];
    T* shared = (T*) (void*) shared_backing;

    csize r = 0;
    if constexpr(static_r != -1)
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
        T val = gather(i, N, r);

        shared[ti] = val;
        __syncthreads();

        if(r <= ti && ti < tile_size-r && i < N)
            func(i, ti, shared, tile_size, N, r);
            
        __syncthreads();
    }
}

template <typename T, csize static_r = -1, typename Function = int, typename Gather = int>
static void cuda_tiled_for(csize N, csize dynamic_r, Gather gather, Function func, Cuda_Launch_Params launch_params = {})
{
    static Cuda_Launch_Bounds bounds = {};
    static Cuda_Launch_Query query = {};
    if(bounds.max_block_size == 0)
    {
        query = cuda_launch_query_from_kernel((void*) cuda_tiled_for_kernel<T, Gather, Function, static_r>);
        query.used_shared_memory_per_thread = sizeof(T);
        bounds = cuda_query_launch_bounds(query);
    }

    csize r = dynamic_r;
    if constexpr(static_r != -1)
        r = static_r;

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
        <<<launch.block_count, launch.block_size, launch.dynamic_shared_memory, launch_params.stream>>>(N, dynamic_r, (Gather&&) gather, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
}


template <typename T, csize static_r = -1, typename Function = int>
static void cuda_tiled_for_bound(const T* data, csize N, csize dynamic_r, Function func, T out_of_bounds_val = T(), Cuda_Launch_Params launch_params = {})
{
    cuda_tiled_for<T, static_r, Function>(N, dynamic_r, [=]SHARED(csize i, csize N, csize r){
        if(0 <= i && i < N)
            return data[i];
        else
            return out_of_bounds_val;
    }, (Function&&) func, launch_params);
}


template <typename T, csize static_r = -1, bool small_r = false, typename Function = int>
static void cuda_tiled_for_periodic(const T* data, csize N, csize dynamic_r, Function func, T out_of_bounds_val = T(), Cuda_Launch_Params launch_params = {})
{
    cuda_tiled_for<T, static_r, Function>(N, dynamic_r, [=]SHARED(csize i, csize N, csize r){
        (void) data;
        if constexpr(small_r)
        {
            if(i < 0)
                i += N;
            else if(i >= N)
                i -= N;

            return data[i];
        }
        else
            return data[MOD(i, N)];
    }, (Function&&) func, launch_params);
}

template <typename T, typename Gather, typename Function, csize static_rx, csize static_ry>
static void __global__ cuda_tiled_for_2D_kernel(csize nx, csize ny, csize dynamic_rx, csize dynamic_ry, Gather gather, Function func)
{
    extern __shared__ max_align_t shared_backing[];
    T* shared = (T*) (void*) shared_backing;

    csize rx = 0;
    if constexpr(static_rx != -1)
        rx = static_rx;
    else
        rx = dynamic_rx;

    csize ry = 0;
    if constexpr(static_ry != -1)
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
            T val = gather(x, y, nx, ny, rx, ry);

            shared[tx + ty*tile_size_x] = val;
            __syncthreads();

            if(rx <= tx && tx < tile_size_x-rx && x < nx)
                if(ry <= ty && ty < tile_size_y-ry && y < ny)
                    func(x, y, tx, ty, shared, tile_size_x, tile_size_y, nx, ny, rx, ry);

            __syncthreads();
        }
    }
}


template <typename T, csize static_rx = -1, csize static_ry = -1, typename Function = int, typename Gather = int>
static void cuda_tiled_for_2D(csize nx, csize ny, csize dynamic_rx, csize dynamic_ry, Gather gather, Function func, Cuda_Launch_Params launch_params = {})
{
    static Cuda_Launch_Bounds bounds = {};
    static Cuda_Launch_Query query = {};
    if(bounds.max_block_size == 0)
    {
        query = cuda_launch_query_from_kernel((void*) cuda_tiled_for_2D_kernel<T, Gather, Function, static_rx, static_ry>);
        query.used_shared_memory_per_thread = sizeof(T);
        bounds = cuda_query_launch_bounds(query);
    }

    if(nx <= 0 || ny <= 0)
        return;

    csize rx = dynamic_rx;
    if constexpr(static_rx != -1)
        rx = static_rx;

    csize ry = dynamic_ry;
    if constexpr(static_ry != -1)
        ry = static_ry;

    csize volume = nx*ny;
    Cuda_Launch_Config launch = cuda_get_launch_config(volume, bounds, launch_params);

    dim3 block_size3 = {1, 1, 1};
    block_size3.x = ROUND_UP(2*rx+1, WARP_SIZE);
    block_size3.y = launch.block_size / block_size3.x;
    if(block_size3.y < 2*ry+1)
    {
        LOG_ERROR("kernel", "couldnt find appropriate config parameters to launch '%s' with nx:%lli ny:%lli rx:%lli ry:%lli", __FUNCTION__, (lli)nx, (lli)ny, (lli)rx, (lli)ry);
        return;
    }

    if(0) {
        LOG_DEBUG("kernel", "cuda_tiled_for_2D launch: N:{%lli %lli} block_count:%i block_size:{%i %i} dynamic_shared_memory:%i\n", 
            (lli)nx, (lli)ny, launch.block_count, block_size3.x, block_size3.y, launch.dynamic_shared_memory);
    }

    cuda_tiled_for_2D_kernel<T, Gather, Function, static_rx, static_ry>
        <<<launch.block_count, block_size3, launch.dynamic_shared_memory, launch_params.stream>>>(nx, ny, dynamic_rx, dynamic_ry, (Gather&&) gather, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
}

template <typename T, csize static_rx = -1, csize static_ry = -1, typename Function = int>
static void cuda_tiled_for_2D_bound(const T* data, csize nx, csize ny, csize dynamic_rx, csize dynamic_ry, Function func, T out_of_bounds_val = T(), Cuda_Launch_Params launch_params = {})
{
    cuda_tiled_for_2D<T, static_rx, static_ry, Function>(nx, ny, dynamic_rx, dynamic_ry, [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry){
        if(0 <= x && x < nx && 0 <= y && y < ny)
            return data[x + y*nx];
        else
            return out_of_bounds_val;
    }, (Function&&) func, launch_params);
}

template <typename T, csize static_rx = -1, csize static_ry = -1, bool small_r = false, typename Function = int>
static void cuda_tiled_for_2D_modular(const T* data, csize nx, csize ny, csize dynamic_rx, csize dynamic_ry, Function func, Cuda_Launch_Params launch_params = {})
{
    cuda_tiled_for_2D<T, static_rx, static_ry, Function>(nx, ny, dynamic_rx, dynamic_ry, [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry){
        csize x_mod = 0;
        csize y_mod = 0;

        if constexpr(small_r)
        {
            if(x_mod < 0)
                x_mod += nx;
            else if(x_mod >= nx)
                x_mod -= nx;

            if(y_mod < 0)
                y_mod += ny;
            else if(y_mod >= ny)
                y_mod -= ny;
        }
        else
        {
            x_mod = MOD(x, nx);
            y_mod = MOD(y, ny);
        }

        return data[x_mod + y_mod*nx];
    }, (Function&&) func, launch_params);
}

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
    for(csize Ni = 0; Ni < STATIC_ARRAY_SIZE(Ns); Ni++)
        if(max_N < Ns[Ni])
            max_N = Ns[Ni];

    csize max_N_bytes = max_N*sizeof(int);
    int* allocation = (int*) malloc(max_N_bytes*4);

    Cache_Tag tag = cache_tag_make();
    int* gpu_out = cache_alloc(int, max_N, &tag);
    int* gpu_stencil = cache_alloc(int, max_N, &tag);
    int* gpu_in = cache_alloc(int, max_N, &tag);

    int input_range = 100;
    int stencil_range = 10;
    srand(seed);
    for(csize Ni = 0; Ni < STATIC_ARRAY_SIZE(Ns); Ni++)
    {
        for(csize ri = 0; ri < STATIC_ARRAY_SIZE(rs); ri++)
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

            cuda_tiled_for_bound(gpu_in, N, r, [=]SHARED(csize i, csize ti, int* __restrict__ shared, csize block_size, csize N, csize r){
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
            });
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
    for(csize Ni = 0; Ni < STATIC_ARRAY_SIZE(ns); Ni++)
        if(max_N < ns[Ni])
            max_N = ns[Ni];

    max_N = max_N*max_N;
    csize max_N_bytes = max_N*sizeof(int);
    int* allocation = (int*) malloc(max_N_bytes*4);

    Cache_Tag tag = cache_tag_make();
    int* gpu_out = cache_alloc(int, max_N, &tag);
    int* gpu_stencil = cache_alloc(int, max_N, &tag);
    int* gpu_in = cache_alloc(int, max_N, &tag);

    int input_range = 100;
    int stencil_range = 10;

    srand(seed);
    for(csize niy = 0; niy < STATIC_ARRAY_SIZE(ns); niy++)
        for(csize nix = 0; nix < STATIC_ARRAY_SIZE(ns); nix++)
            for(csize riy = 0; riy < STATIC_ARRAY_SIZE(rs); riy++)
                for(csize rix = 0; rix < STATIC_ARRAY_SIZE(rs); rix++)
                {
                    csize nx = ns[nix];
                    csize ny = ns[niy];
                    csize rx = rs[rix];
                    csize ry = rs[riy];

                    csize sx = 2*rx+1;
                    csize sy = 2*ry+1;

                    csize N_bytes = nx*ny*sizeof(int);
                    
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
                    CUDA_DEBUG_TEST(cudaMemcpy(gpu_stencil, cpu_stencil, sx*sy*sizeof(int), cudaMemcpyHostToDevice));

                    cuda_tiled_for_2D_bound(gpu_in, nx, ny, rx, ry, [=]SHARED(
                        csize x, csize y, csize tx, csize ty, 
                        int* __restrict__ shared, csize tile_size_x, csize tile_size_y, 
                        csize nx, csize ny, csize rx, csize ry){

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
                    });
                    
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