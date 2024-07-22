
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wsign-conversion"

#include "cuda_util.cuh"
#include "cuda_random.cuh"
#include <time.h>

//TESTING CODE===========================
struct Temp_Mallocs
{
    enum {TEMP_CUDA_ALLOCS_MAX = 32};
    void* host_allocs[TEMP_CUDA_ALLOCS_MAX] = {0};
    void* device_allocs[TEMP_CUDA_ALLOCS_MAX] = {0};
    
    int count = 0;

    ~Temp_Mallocs()
    {
        for(int i = 0; i < count; i++)
            CUDA_TEST(cudaFree(device_allocs[i]));
        for(int i = 0; i < count; i++)
            free(host_allocs[i]);
    }

    void* host_device_alloc(size_t size, void** host_out)
    {
        TEST(count < TEMP_CUDA_ALLOCS_MAX);
        CUDA_TEST(cudaMalloc(&device_allocs[count], size));
        host_allocs[count] = calloc(size, 1);
        TEST(host_allocs[count] != NULL);

        *host_out = host_allocs[count];
        return device_allocs[count++];
    }

    float* host_device_random_floats(int count, float from, float to, float** host_out)
    {
        float* host = NULL;
        float* device = (float*) host_device_alloc((size_t) count*sizeof(float), (void**) &host);
        for(int i = 0; i < count; i++)
            host[i] = random_f32(from, to);
        
        CUDA_TEST(cudaMemcpy(device, host, (size_t)count*sizeof(float), cudaMemcpyHostToDevice));
        *host_out = host;
        return device;
    }
};

void cuda_test_floats_nearby(const float* device, const float* host, int count, float epsilon = 1e-5f)
{
    float* temp = (float*) malloc((size_t) count*sizeof(float));
    CUDA_TEST(cudaMemcpy(temp, device, (size_t) count*sizeof(float), cudaMemcpyDeviceToHost));

    for(int i = 0; i < count; i++)
        TEST(fabsf(temp[i] - host[i]) < epsilon, "Must be nearby at index %i device:%e host:%e", i, (double) temp[i], (double) host[i]);

    free(temp);
}

//================================================================
//                       naive parallel for
//================================================================
__global__ void saxpy_naive_kernel(float* result, float a, const float* x, const float* y)
{
    int i = threadIdx.x;
    result[i] = a*x[i] + y[i];
}

int main_saxpy_naive()
{
    Temp_Mallocs mallocs;
    float* result_host = NULL;
    float* x_host = NULL;
    float* y_host = NULL;

    int N = random_int_with_high_chance_of_extremes(1, 32)*32;
    float* result = mallocs.host_device_random_floats(N, 0, 1, &result_host);
    float* x = mallocs.host_device_random_floats(N, 0, 1, &x_host);
    float* y = mallocs.host_device_random_floats(N, 0, 1, &y_host);
    float a = random_f32(-2, 2);
    
    saxpy_naive_kernel<<<1, N>>>(result, a, x, y);
    assert(cudaGetLastError() == 0);
    
    CUDA_TEST(cudaGetLastError());
    for(int i = 0; i < N; i++)
        result_host[i] = a*x_host[i] + y_host[i];
    cuda_test_floats_nearby(result, result_host, N);
    return 0;
}


//================================================================
//                       cuda malloc
//================================================================

//If this does not crash it is declared to work
int main_cuda_malloc()
{
    int N = 100;
    int size = N * sizeof(float);
    
    float* host_mem = (float*) malloc(size);
    float* device_mem = NULL;
    cudaMalloc(&device_mem, size);

    for(int i = 0; i < N; i++)
        host_mem[i] = i;

    cudaMemcpy(device_mem, host_mem, size, cudaMemcpyHostToDevice);

    //do computation with device_mem

    cudaMemcpy(host_mem, device_mem, size, cudaMemcpyDeviceToHost);
    //for(int i = 0; i < N; i++)
        //printf("%f\n", host_mem[i]);

    cudaFree(device_mem);
    free(host_mem);

    CUDA_TEST(cudaGetLastError());
    return 0;
}

//================================================================
//                       saxpy grid stride
//================================================================
__global__ void saxpy_grid_stride(float* result, float a, const float* x, const float* y, int N)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; 
        i < N; 
        i += blockDim.x*gridDim.x) { 
        result[i] = a*x[i] + y[i];
    }
}

int main_grid_stride()
{
    Temp_Mallocs mallocs;
    float* result_host = NULL;
    float* x_host = NULL;
    float* y_host = NULL;

    int N = random_int_with_high_chance_of_extremes(1, 5000);
    float* result = mallocs.host_device_random_floats(N, 0, 1, &result_host);
    float* x = mallocs.host_device_random_floats(N, 0, 1, &x_host);
    float* y = mallocs.host_device_random_floats(N, 0, 1, &y_host);
    float a = random_f32(-2, 2);
    
    int blockSize = 256;
    int gridSize = 2; 
    saxpy_grid_stride<<<gridSize, blockSize>>>(result, a, x, y, N);
    
    CUDA_TEST(cudaGetLastError());
    for(int i = 0; i < N; i++)
        result_host[i] = a*x_host[i] + y_host[i];
    cuda_test_floats_nearby(result, result_host, N);
    return 0;
}

//================================================================
//                       parallel for
//================================================================

template <typename Function>
__global__ void parallel_for_kernel(int from, int N, Function func)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; 
        i < N; 
        i += blockDim.x*gridDim.x) 
        func(from + i);
}

template <typename Function>
void parallel_for(int from, int to, Function func)
{
    int blockSize = 256;
    int gridSize = (to - from + blockSize - 1) / blockSize; //divide round up
    parallel_for_kernel<<<gridSize, blockSize>>>(from, to-from, func);
}

int main_parallel_for()
{
    Temp_Mallocs mallocs;
    float* result_host = NULL;
    float* x_host = NULL;
    float* y_host = NULL;

    int N = random_int_with_high_chance_of_extremes(1, 5000);
    float* result = mallocs.host_device_random_floats(N, 0, 1, &result_host);
    float* x = mallocs.host_device_random_floats(N, 0, 1, &x_host);
    float* y = mallocs.host_device_random_floats(N, 0, 1, &y_host);
    float a = random_f32(-2, 2);
    
    parallel_for(0, N, [=]__device__ __host__(int i){
        result[i] = a*x[i] + y[i];
    });

    CUDA_TEST(cudaGetLastError());
    for(int i = 0; i < N; i++)
        result_host[i] = a*x_host[i] + y_host[i];
    cuda_test_floats_nearby(result, result_host, N);
    return 0;
}


//================================================================
//                    Grid stride loop 2D
//================================================================
__global__ void saxpy2D(float* result, float a, const float* x, const float* y, 
                        int N, int M)
{
    for (int i = blockIdx.y*blockDim.y + threadIdx.y; 
        i < M; i += blockDim.y*gridDim.y) 
        for (int j = blockIdx.x*blockDim.x + threadIdx.x; 
            j < N; j += blockDim.x*gridDim.x) 
        {
            int I = i*N + j;
            result[I] = a*x[I] + y[I];
        }
}

int main_saxpy_2D()
{
    Temp_Mallocs mallocs;
    float* result_host = NULL;
    float* x_host = NULL;
    float* y_host = NULL;

    int N = random_int_with_high_chance_of_extremes(1, 3000);
    int M = random_int_with_high_chance_of_extremes(1, 3000);
    float* result = mallocs.host_device_random_floats(N*M, 0, 1, &result_host);
    float* x = mallocs.host_device_random_floats(N*M, 0, 1, &x_host);
    float* y = mallocs.host_device_random_floats(N*M, 0, 1, &y_host);
    float a = random_f32(-2, 2);
    
    dim3 blockDim(16, 16);
    dim3 gridDim(4, 4);
    saxpy2D<<<gridDim, blockDim>>>(result, a, x, y, N, M);
    CUDA_TEST(cudaGetLastError());

    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            result_host[i*N + j] = a*x_host[i*N + j] + y_host[i*N + j];

    cuda_test_floats_nearby(result, result_host, N*M);
    return 0;
}


//================================================================
//                    Tridiag mul naive
//================================================================
void tridiag_mul(float* result, float a, float b, float c, const float* x, int N)
{
    parallel_for(1, N-1, [=]__device__ __host__(int i){
        result[i] = b*x[i-1] + a*x[i] + c*x[i+1];
    });  
}


//================================================================
//                         sum 64
//================================================================
__global__ void sum64(float* result, const float* x, int N)
{
    assert(blockDim.x == 64);
    __shared__ float shared[64];
    
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    shared[threadIdx.x] = x[i];
    __syncthreads();

    float sum = 0;
    for(int j = 0; j < 64; j++)
        sum += shared[j];

    result[i] = sum;
}

int main_sum64()
{  
    Temp_Mallocs mallocs;
    float* result_host = NULL;
    float* x_host = NULL;

    int N = random_int_with_high_chance_of_extremes(1, 1000)*64;
    float* result = mallocs.host_device_random_floats(N, 0, 1, &result_host);
    float* x = mallocs.host_device_random_floats(N, 0, 1, &x_host);
    
    sum64<<<N/64, 64>>>(result, x, N);
    CUDA_TEST(cudaGetLastError());

    for(int j = 0; j < N/64; j++)
    {
        int to = MIN((j+1)*64, N);
        float sum = 0;
        for(int i = j*64; i < to; i++)
            sum += x_host[i];

        for(int i = j*64; i < to; i++)
            result_host[i] = sum;
    }

    cuda_test_floats_nearby(result, result_host, N);
    return 0;
}


//================================================================
//                         sum tile
//================================================================
__global__ void sum_tile(float* result, const float* x, int N)
{
    extern __shared__ float shared[];
    for (int i_base = blockIdx.x*blockDim.x; ; i_base += gridDim.x*blockDim.x) 
    {
        if(i_base >= N)
            break;
            
        int i = i_base + threadIdx.x;
        float read = 0;
        if(i < N)
            read = x[i];

        shared[threadIdx.x] = read;
        __syncthreads();

        if(i < N)
        {
            float sum = 0;
            for(int j = 0; j < blockDim.x; j++)
                sum += shared[j];
    
            result[i] = sum;
        }
        __syncthreads();
    }
}

int main_sum_tile()
{
    Temp_Mallocs mallocs;
    float* result_host = NULL;
    float* x_host = NULL;

    int N = random_int_with_high_chance_of_extremes(1, 5000);
    float* result = mallocs.host_device_random_floats(N, 0, 1, &result_host);
    float* x = mallocs.host_device_random_floats(N, 0, 1, &x_host);
    int tileSize = random_int_with_high_chance_of_extremes(1, 1024);
    
    int blockSize = tileSize;
    int gridSize = (N + blockSize - 1)/blockSize;
    int sharedMemSize = blockSize*sizeof(float);
    sum_tile<<<gridSize, blockSize, sharedMemSize>>>(result, x, N);
    CUDA_TEST(cudaGetLastError());

    for(int j = 0; j < gridSize; j++)
    {
        int to = MIN((j+1)*blockSize, N);
        float sum = 0;
        for(int i = j*blockSize; i < to; i++)
            sum += x_host[i];

        for(int i = j*blockSize; i < to; i++)
            result_host[i] = sum;
    }

    cuda_test_floats_nearby(result, result_host, N);
    return 0;
}


//================================================================
//                         tridiag mul
//================================================================

__global__ void tridiag_mul_shared(float* result, float a, float b, float c, const float* x, int N)
{
    extern __shared__ float shared[];

    int r = 1;
    for (int bi = blockIdx.x; ; bi += gridDim.x) 
    {
        int i_base = bi*blockDim.x - 2*bi*r;
        if(i_base >= N)
            break;

        int i = i_base - r + threadIdx.x;
        float val = 0;

        if(0 <= i && i < N)
            val = x[i];

        shared[threadIdx.x] = val;
        __syncthreads();

        if(r <= threadIdx.x && threadIdx.x < blockDim.x-r)
            if(i < N)
                result[i] = b*shared[threadIdx.x-1] + a*shared[threadIdx.x] + c*shared[threadIdx.x+1];
            
        __syncthreads();
    }
}

int main_tridiag_mul_shared()
{
    Temp_Mallocs mallocs;
    float* result_host = NULL;
    float* x_host = NULL;

    int N = random_int_with_high_chance_of_extremes(1, 5000);
    float* result = mallocs.host_device_random_floats(N, 0, 1, &result_host);
    float* x = mallocs.host_device_random_floats(N, 0, 1, &x_host);
    float a = random_f32(-2, 2);
    float b = random_f32(-2, 2);
    float c = random_f32(-2, 2);
    int tileSize = random_int_with_high_chance_of_extremes(3, 1024);
    
    int blockSize = tileSize;
    int gridSize = (N + blockSize - 1)/blockSize;
    int sharedMemSize = 1024*sizeof(float);
    tridiag_mul_shared<<<gridSize, blockSize, sharedMemSize>>>(result, a, b, c, x, N);
    CUDA_TEST(cudaGetLastError());

    for(int i = 0; i < N; i++)
    {
        float back = 0;
        float forw = 0;
        if(i > 0)
            back = b*x_host[i - 1];

        if(i < N - 1)
            forw = c*x_host[i + 1];

        result_host[i] = back + a*x_host[i] + forw;
    }

    cuda_test_floats_nearby(result, result_host, N);
    return 0;
}

//================================================================
//                         reductions
//================================================================
float reduce(const float* x, int N)
{
    float max = -INFINITY;
    for(int i = 0; i < N; i++)
        max = fmaxf(max, x[i]);
        
    return max;
}

float global_mem_reduce(float* temp1, float* temp2, const float* x, int N)
{
    cudaMemcpy(temp1, x, N*sizeof(float), cudaMemcpyDeviceToDevice);
    CUDA_TEST(cudaGetLastError());
    while(N > 1)
    {
        int half = (N + 1) / 2;
        parallel_for(0, half, [=]__device__ __host__ (int i){
            if(i + half < N)
                temp2[i] = fmaxf(temp1[i], temp1[i + half]);
            else
                temp2[i] = temp1[i];
        });  

        CUDA_TEST(cudaGetLastError());
        std::swap(temp1, temp2);
        N = half;
    }

    float max = -INFINITY;
    cudaMemcpy(&max, temp1, sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_TEST(cudaGetLastError());
    return max;
}

__global__ void shared_mem_reduce_kernel(float* output, const float* input, int N) 
{
    assert((blockDim.x & (blockDim.x-1)) == 0 && "must be power of two!");
    extern __shared__ float shared[];

    float reduced = -INFINITY;
    for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x)
        reduced = fmaxf(input[i], -INFINITY);

    shared[threadIdx.x] = reduced;
    __syncthreads();

    for(int sharedSize = blockDim.x/2; sharedSize > 0; sharedSize /= 2)
    {
        if(threadIdx.x < sharedSize)
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + sharedSize]);

        __syncthreads();      
    } 

    if(threadIdx.x == 0)
        output[blockIdx.x] = shared[0];
}

float shared_mem_reduce(float* temp1, float* temp2, const float* input, int N)
{
    const float* from = input;
    enum {MIN_SIZE = 64};
    while(N > MIN_SIZE)
    {
        int blockSize = 1024;
        int sharedMemSize = blockSize*sizeof(float);
        int gridSize = (N + blockSize - 1) / blockSize;

        shared_mem_reduce_kernel
            <<<gridSize, blockSize, sharedMemSize>>>(temp2, from, N);

        std::swap(temp1, temp2);
        N = gridSize;
        from = temp1;
    }

    float cpu[MIN_SIZE] = {0};
    cudaMemcpy(cpu, from, N*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_TEST(cudaGetLastError());

    float max = -INFINITY;
    for(int i = 0; i < N; i++)
        max = fmaxf(max, cpu[i]);
    return max;
}

__global__ void warp_reduce_kernel(float* output, const float* x, int N) 
{
    enum {WARP_SIZE = 32};
    assert(blockDim.x > WARP_SIZE && blockDim.x % WARP_SIZE == 0);

    extern __shared__ float shared[];
    uint shared_size = blockDim.x / WARP_SIZE;

    float reduced = 0;
    for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x)
        reduced = fmaxf(reduced, x[i]);

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) 
        reduced = fmaxf(__shfl_down_sync(0xffffffffU, reduced, offset), reduced);

    uint ti = threadIdx.x;
    if (ti % WARP_SIZE == 0) 
        shared[ti / WARP_SIZE] = reduced;
    
    __syncthreads();
    uint ballot_mask = __ballot_sync(0xffffffffU, ti < shared_size);
    if (ti < shared_size) 
    {
        reduced = shared[ti];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) 
            reduced = fmaxf(__shfl_down_sync(ballot_mask, reduced, offset), reduced);
    }

    if (ti == 0) 
        output[blockIdx.x] = reduced;
}

//same as shared_mem_reduce but calls warp_reduce_kernel
float warp_reduce(float* temp1, float* temp2, const float* input, int N)
{
    const float* from = input;
    enum {MIN_SIZE = 64};
    while(N > MIN_SIZE)
    {
        int blockSize = 1024;
        int sharedMemSize = blockSize*sizeof(float);
        int gridSize = (N + blockSize - 1) / blockSize;

        warp_reduce_kernel
            <<<gridSize, blockSize, sharedMemSize>>>(temp2, from, N);

        std::swap(temp1, temp2);
        N = gridSize;
        from = temp1;
    }

    float cpu[MIN_SIZE] = {0};
    cudaMemcpy(cpu, from, N*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_TEST(cudaGetLastError());

    float max = -INFINITY;
    for(int i = 0; i < N; i++)
        max = fmaxf(max, cpu[i]);
    return max;
}

int main_reduce()
{
    Temp_Mallocs mallocs;
    float* x_host = NULL;

    int N = random_int_with_high_chance_of_extremes(1, 5000);
    float* x = mallocs.host_device_random_floats(N, 0, 1000, &x_host);
    float* temp1 = NULL;
    float* temp2 = NULL;
    CUDA_TEST(cudaMalloc(&temp1, N*sizeof(float)));
    CUDA_TEST(cudaMalloc(&temp2, N*sizeof(float)));

    float cpu_reduced = reduce(x_host, N);
    float global_mem_reduced = global_mem_reduce(temp1, temp2, x, N);
    float shared_mem_reduced = shared_mem_reduce(temp1, temp2, x, N);
    float warp_reduced = warp_reduce(temp1, temp2, x, N);

    TEST(global_mem_reduced == cpu_reduced);
    TEST(shared_mem_reduced == cpu_reduced);
    TEST(warp_reduced == cpu_reduced);

    cudaFree(temp1);
    cudaFree(temp2);
    return 0;
}


template <typename T, class Reduction, typename Producer>
T produce_reduce(int N, Producer produce, int cpu_reduce = 256)
{
    (void) N;
    (void) produce;
    (void) cpu_reduce;
    return T();
}

float euclid_norm(const float *a, int N)
{
    float output = produce_reduce<float, Reduce::Add>(N, [=]__host__ __device__(int i){
        return a[i]*a[i];
    });
    return sqrtf(output);
}

void test_all_examples(double seconds)
{
    cudaSetDevice(0);
    // cuda_one_time_setup();
    int iter = 0;
    for(double start = clock_s(); clock_s() < start + seconds; )
    {
        TEST(main_saxpy_naive() == 0);
        TEST(main_cuda_malloc() == 0);
        TEST(main_grid_stride() == 0);
        TEST(main_parallel_for() == 0);
        TEST(main_sum64() == 0);
        TEST(main_sum_tile() == 0);
        TEST(main_tridiag_mul_shared() == 0);
        TEST(main_reduce() == 0);

        iter += 1;
        printf("All cuda example tests passed #%i\n", iter);
    }
}
#pragma GCC diagnostic pop