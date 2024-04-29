#pragma once

#include "cuda_util.cuh"
#include "cuda_launch.cuh"
#include "cuda_alloc.cuh"

// This file provides state of the art generic reduction operations. 
// The main kernel is fairly concice but the nature of the generic 
// implementation stretches this file out a bit. 
// We achive greater or equal performance then cuda thrust on all benchmarks I have run.
// The algorhitm is tested on various sizes and random data ensuring functioning implementation.  

//The main function in this file is produce_reduce which is a generalized map_reduce. 
// produce_reduce takes a function (index)->(value) which can be easily customized to pull 
// from different sources of data at the same time making for very easy implementation of 
//  vector dot product, efficient normed distances and other functions that would normally need zipping.
//If you are not convinced this is in fact simpler approach then the traditional map_reduce
// interface compare https://github.com/NVIDIA/thrust/blob/main/examples/dot_products_with_zip.cu with our dot_product function
template <typename T, class Reduction, typename Producer, bool has_trivial_producer = false>
static T produce_reduce(csize N, Producer produce, Reduction reduce_dummy = Reduction(), csize cpu_reduce = 128, Cuda_Launch_Params launch_params = {});

//Predefined reduction functions. These are defined in special ways because:
// 1. They can be used with any type (that suports them!). No need to template them => simpler outside interface.
// 2. They can be tested for and thus the implementation can and does take advantage of custom intrinsics.
//
//Others can be defined from within custom struct types see Example_Custom_Reduce_Operation below.
struct Reduce {
    struct Add {};
    struct Mul {};
    struct Min {};
    struct Max {};
    struct Or  {}; // *
    struct And {}; // *
    struct Xor {}; // *
    //(*) - Cannot be used with floating point types

    //Type tags. Used to specify the reduction operations more conveniently. 
    //Search their usage for examples.
    static Add ADD;
    static Mul MUL;
    static Min MIN;
    static Max MAX;
    static Or  OR;
    static And AND;
    static Xor XOR;
};

template <typename T>
struct Example_Custom_Reduce_Operation 
{
    static const char* name() {
        return "Example_Custom_Reduce_Operation";
    }

    //Returns the idnetity element for this operation
    static T identity() {
        return INFINITY;
    }

    //Performs the reduction. 
    //Important: the operation needs to be comutative and associative!
    static __host__ __device__ T reduce(T a, T b) {
        //some example implementaion
        return MIN(floor(a), floor(b));
    }
};

//Then call like so (both lines work): 
// produce_reduce(N, produce, Example_Custom_Reduce_Operation<T>())
// produce_reduce<Example_Custom_Reduce_Operation<T>>(N, produce)

//Returns the identity elemt of the Reduction operation
template <class Reduction, typename T>
static __host__ __device__ __forceinline__ T _identity();

//Returns the result of redeuction using the Reduction operation
template <class Reduction, typename T>
static __host__ __device__ __forceinline__ T _reduce(T a, T b);

template <class Reduction>
static __host__ __device__ __forceinline__ const char* _name();

//Performs reduction operation within lanes of warp enabled by mask. 
//If mask includes lanes which do not participate within the reduction 
// (as a cause of divergent control flow) might cause deadlocks.
template <class Reduction, typename T>
static __device__ __forceinline__ T _warp_reduce(unsigned int mask, T value) ;


template <class Reduction, typename T, typename Producer, bool is_trivial_producer>
static __global__ void produce_reduce_kernel(T* __restrict__ output, Producer produce, csize N) 
{
    assert(blockDim.x > WARP_SIZE && blockDim.x % WARP_SIZE == 0 && "we expect the block dim to be chosen sainly");
    uint shared_size = blockDim.x / WARP_SIZE;

    extern __shared__ max_align_t shared_backing[];
    T* shared = (T*) (void*) shared_backing;

    T reduced = _identity<Reduction, T>();
    for(int i = (int) blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x)
    {
        T produced;
        if constexpr(is_trivial_producer)
            produced = produce[i];
        else 
            produced = produce(i);

        reduced = _reduce<Reduction, T>(produced, reduced);
    }

    reduced = _warp_reduce<Reduction, T>(0xffffffffU, reduced);
    uint ti = threadIdx.x;
    if (ti % WARP_SIZE == 0) 
        shared[ti / WARP_SIZE] = reduced;
    
    __syncthreads();
    uint ballot_mask = __ballot_sync(0xffffffffU, ti < shared_size);
    if (ti < shared_size) 
    {
        reduced = shared[ti];
        reduced = _warp_reduce<Reduction, T>(ballot_mask, reduced);
    }

    if (ti == 0) 
        output[blockIdx.x] = reduced;
}

template <size_t size_per_elem>
size_t reduce_block_size_to_shared_mem_size(int block_size)
{
    return DIV_CEIL(block_size, WARP_SIZE)*size_per_elem;

}

#include <cuda_occupancy.h>
#include <cuda_runtime.h>

template <typename T, class Reduction, typename Producer, bool has_trivial_producer>
static T produce_reduce(csize N, Producer produce, Reduction reduce_dummy, csize cpu_reduce, Cuda_Launch_Params launch_params)
{
    (void) reduce_dummy;
    
    // LOG_INFO("reduce", "Reduce %s N:%i", _name<Reduction>(), (int) N);
    T reduced = _identity<Reduction, T>();
    if(N > 0)
    {
        enum {
            CPU_REDUCE_MAX = 256,
            CPU_REDUCE_MIN = 32,
        };

        static Cuda_Launch_Bounds bounds = {};
        static Cuda_Launch_Constraints constraints = {};
        if(bounds.max_block_size == 0)
        {
            constraints = cuda_constraints_launch_constraints((void*) produce_reduce_kernel<Reduction, T, Producer, has_trivial_producer>);
            constraints.used_shared_memory_per_thread = (double) sizeof(T)/WARP_SIZE;
            bounds = cuda_get_launch_bounds(constraints);
        }

        cpu_reduce = CLAMP(cpu_reduce, CPU_REDUCE_MIN, CPU_REDUCE_MAX);
        Cache_Tag tag = cache_tag_make();
        T* partials[2] = {NULL, NULL};

        csize N_curr = N;
        uint iter = 0;
        for(; (iter == 0 && has_trivial_producer == false) || N_curr > cpu_reduce; iter++)
        {
            Cuda_Launch_Config launch = cuda_get_launch_config(N_curr, bounds, launch_params);
            if(bounds.max_block_size == 0 || launch.block_size == 0)
            {
                LOG_ERROR("reduce", "this device has very strange hardware and as such we cannot launch the redutcion kernel. This shouldnt happen.");
                cache_free(&tag);
                return reduced;
            }

            uint i_curr = iter%2;
            uint i_next = (iter + 1)%2;
            if(iter < 2)
                partials[i_next] = cache_alloc(T, N, &tag);

            T* __restrict__ curr_input = partials[i_curr];
            T* __restrict__ curr_output = partials[i_next];
            if(iter ==  0)
            {
                produce_reduce_kernel<Reduction, T, Producer, has_trivial_producer>
                    <<<launch.block_count, launch.block_size, launch.dynamic_shared_memory, launch_params.stream>>>(curr_output, (Producer&&) produce, N_curr);
            }
            else
            {
                produce_reduce_kernel<Reduction, T, const T* __restrict__, true>
                    <<<launch.block_count, launch.block_size, launch.dynamic_shared_memory, launch_params.stream>>>(curr_output, curr_input, N_curr);
            }

            CUDA_DEBUG_TEST(cudaGetLastError());
            // CUDA_DEBUG_TEST(cudaDeviceSynchronize());

            uint G = MIN(N_curr, launch.block_size*launch.block_count);
            uint N_next = DIV_CEIL(G, WARP_SIZE*WARP_SIZE); 
            N_curr = N_next;
        }   

        //in case N <= CPU_REDUCE and has_trivial_producer 
        // we entirely skipped the whole loop => the partials array is
        // still null. We have to init it to the produce array.
        const T* last_input = partials[iter%2];
        if constexpr(has_trivial_producer)
            if(N <= cpu_reduce)
                last_input = produce;

        T cpu[CPU_REDUCE_MAX];
        cudaMemcpy(cpu, last_input, sizeof(T)*N_curr, cudaMemcpyDeviceToHost);
        for(uint i = 0; i < N_curr; i++)
            reduced = _reduce<Reduction, T>(reduced, cpu[i]);

        cache_free(&tag);
    }

    return reduced;
}

//============================== IMPLEMENTATION OF MORE SPECIFIC REDUCTIONS ===================================

template<class Reduction, typename T, typename Map_Func>
static T map_reduce(const T *input, uint N, Map_Func map, Reduction reduce_tag = Reduction())
{
    T output = produce_reduce<T, Reduction>(N, [=]SHARED(csize i){
        return map(input[i]);
    });
    return output;
}

template<class Reduction, typename T>
static T reduce(const T *input, uint N, Reduction reduce_tag = Reduction())
{
    T output = produce_reduce<T, Reduction, const T* __restrict__, true>(N, input);
    return output;
}

template<typename T>
static T sum(const T *input, uint N)
{
    T output = produce_reduce<T, Reduce::Add, const T* __restrict__, true>(N, input);
    return output;
}

template<typename T>
static T min(const T *input, uint N)
{
    T output = produce_reduce<T, Reduce::Min, const T* __restrict__, true>(N, input);
    return output;
}

template<typename T>
static T max(const T *input, uint N)
{
    T output = produce_reduce<T, Reduce::Max, const T* __restrict__, true>(N, input);
    return output;
}

template<typename T>
static T L1_norm(const T *a, uint N)
{
    T output = produce_reduce<T, Reduce::Add>(N, [=]SHARED(csize i){
        T diff = a[i];
        return diff > 0 ? diff : -diff;
    });
    return output;
}

template<typename T>
static T L2_norm(const T *a, uint N)
{
    T output = produce_reduce<T, Reduce::Add>(N, [=]SHARED(csize i){
        T diff = a[i];
        return diff*diff;
    });
    return (T) sqrt(output);
}

template<typename T>
static T Lmax_norm(const T *a, uint N)
{
    T output = produce_reduce<T, Reduce::Max>(N, [=]SHARED(csize i){
        T diff = a[i];
        return diff > 0 ? diff : -diff;
    });
    return output;
}

// more complex binary reductions...

template<typename T>
static T L1_distance(const T *a, const T *b, uint N)
{
    T output = produce_reduce<T, Reduce::Add>(N, [=]SHARED(csize i){
        T diff = a[i] - b[i];
        return diff > 0 ? diff : -diff;
    });
    return output;
}

template<typename T>
static T L2_distance(const T *a, const T *b, uint N)
{
    T output = produce_reduce<T, Reduce::Add>(N, [=]SHARED(csize i){
        T diff = a[i] - b[i];
        return diff*diff;
    });
    return (T) sqrt(output);
}

template<typename T>
static T Lmax_distance(const T *a, const T *b, uint N)
{
    T output = produce_reduce<T, Reduce::Max>(N, [=]SHARED(csize i){
        T diff = a[i] - b[i];
        return diff > 0 ? diff : -diff;
    });
    return output;
}

template<typename T>
static T dot_product(const T *a, const T *b, uint N)
{
    T output = produce_reduce<T, Reduce::Add>(N, [=]SHARED(csize i){
        return a[i] * b[i];
    });
    return output;
}

//============================== TEMPLATE MADNESS BELOW ===================================

template <class Reduction, typename T>
static __host__ __device__ __forceinline__ T _identity()
{
    if constexpr      (std::is_same_v<Reduction, Reduce::Add>)
        return (T) 0;
    else if constexpr (std::is_same_v<Reduction, Reduce::Mul>)
        return (T) 1;
    else if constexpr (std::is_same_v<Reduction, Reduce::Min>)
        return (T) MAX(std::numeric_limits<T>::max(), std::numeric_limits<T>::infinity());
    else if constexpr (std::is_same_v<Reduction, Reduce::Max>)
        return (T) MIN(std::numeric_limits<T>::min(), -std::numeric_limits<T>::infinity());
    else if constexpr (std::is_same_v<Reduction, Reduce::And>)
        return (T) 0xFFFFFFFFFFFFULL;
    else if constexpr (std::is_same_v<Reduction, Reduce::Or>)
        return (T) 0;
    else if constexpr (std::is_same_v<Reduction, Reduce::Xor>)
        return (T) 0;
    else
        return Reduction::identity();
}

template <class Reduction, typename T>
static __host__ __device__ __forceinline__ T _reduce(T a, T b)
{
    if constexpr      (std::is_same_v<Reduction, Reduce::Add>)
        return a + b;
    else if constexpr (std::is_same_v<Reduction, Reduce::Mul>)
        return a * b;
    else if constexpr (std::is_same_v<Reduction, Reduce::Min>)
        return MIN(a, b);
    else if constexpr (std::is_same_v<Reduction, Reduce::Max>)
        return MAX(a, b);
    else if constexpr (std::is_same_v<Reduction, Reduce::And>)
        return a & b;
    else if constexpr (std::is_same_v<Reduction, Reduce::Or>)
        return a | b;
    else if constexpr (std::is_same_v<Reduction, Reduce::Xor>)
        return a ^ b;
    else
        return Reduction::reduce(a, b);
}

template <class Reduction>
static __host__ __device__ __forceinline__ const char* _name()
{
    if constexpr      (std::is_same_v<Reduction, Reduce::Add>)
        return "Add";
    else if constexpr (std::is_same_v<Reduction, Reduce::Mul>)
        return "Mul";
    else if constexpr (std::is_same_v<Reduction, Reduce::Min>)
        return "Min";
    else if constexpr (std::is_same_v<Reduction, Reduce::Max>)
        return "Max";
    else if constexpr (std::is_same_v<Reduction, Reduce::And>)
        return "And";
    else if constexpr (std::is_same_v<Reduction, Reduce::Or>)
        return "Or";
    else if constexpr (std::is_same_v<Reduction, Reduce::Xor>)
        return "Xor";
    else
        return Reduction::name();
}


template <class Reduction, typename T>
static __device__ __forceinline__ T _warp_reduce(unsigned int mask, T value) 
{
    //For integral (up to 32 bit) types we can use builtins for super fast reduction.
    constexpr bool is_okay_int = std::is_integral_v<T> && sizeof(T) <= sizeof(int); (void) is_okay_int;

    if constexpr(0) {}
    #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    else if constexpr (is_okay_int && std::is_same_v<Reduction, Reduce::Add>)
        return (T) __reduce_add_sync(mask, value);
    else if constexpr (is_okay_int && std::is_same_v<Reduction, Reduce::Min>)
        return (T) __reduce_min_sync(mask, value);
    else if constexpr (is_okay_int && std::is_same_v<Reduction, Reduce::Max>)
        return (T) __reduce_max_sync(mask, value);
    else if constexpr (is_okay_int && std::is_same_v<Reduction, Reduce::And>)
        return (T) __reduce_and_sync(mask, (unsigned) value);
    else if constexpr (is_okay_int && std::is_same_v<Reduction, Reduce::Or>)
        return (T) __reduce_or_sync(mask, (unsigned) value);
    else if constexpr (is_okay_int && std::is_same_v<Reduction, Reduce::Xor>)
        return (T) __reduce_xor_sync(mask, (unsigned) value);
    #endif
    else
    {
        //@NOTE: T must be one of: (unsigned) short, (unsigned) int, (unsigned) long long, hafl, float, double.
        //Otherwise when sizeof(T) < 8 we will need to write custom casting code
        //Otherwiese when sizeof(T) > 8 we will need to change to algorhimt to do only shared memory reduction
        #if 1
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) 
            value = _reduce<Reduction, T>(__shfl_down_sync(mask, value, offset), value);
        #else
            value = _reduce<Reduction, T>(__shfl_down_sync(mask, value, 16), value);
            value = _reduce<Reduction, T>(__shfl_down_sync(mask, value, 8), value);
            value = _reduce<Reduction, T>(__shfl_down_sync(mask, value, 4), value);
            value = _reduce<Reduction, T>(__shfl_down_sync(mask, value, 2), value);
            value = _reduce<Reduction, T>(__shfl_down_sync(mask, value, 1), value);
        #endif

        return value;
    }
}

#ifdef COMPILE_THRUST
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

template<class Reduction, typename T>
static T thrust_reduce(const T *input, int n, Reduction reduce_tag = Reduction())
{
    // wrap raw pointers to device memory with device_ptr
    thrust::device_ptr<const T> d_input(input);

    T id = _identity<Reduction, T>();
    if constexpr(std::is_same_v<Reduction, Reduce::Add>)
        return thrust::reduce(d_input, d_input+n, id, thrust::plus<T>());
    else if constexpr(std::is_same_v<Reduction, Reduce::Min>)
        return thrust::reduce(d_input, d_input+n, id, thrust::minimum<T>());
    else if constexpr(std::is_same_v<Reduction, Reduce::Max>)
        return thrust::reduce(d_input, d_input+n, id, thrust::maximum<T>());
    else if constexpr(std::is_same_v<Reduction, Reduce::And>)
        return thrust::reduce(d_input, d_input+n, id, thrust::bit_and<T>());
    else if constexpr(std::is_same_v<Reduction, Reduce::Or>)
        return thrust::reduce(d_input, d_input+n, id, thrust::bit_or<T>());
    else if constexpr(std::is_same_v<Reduction, Reduce::Xor>)
        return thrust::reduce(d_input, d_input+n, id, thrust::bit_xor<T>());
    else
        ASSERT(false, "Bad reduce type for thrust!");
    return 0;
}
#else
template<class Reduction, typename T>
static T thrust_reduce(const T *input, int N, Reduction reduce_tag = Reduction())
{
    return reduce<Reduction, T>(input, N, reduce_tag);
}
#endif

template<class Reduction, typename T>
static T cpu_reduce(const T *input, int n, Reduction reduce_tag = Reduction())
{
    enum {COPY_AT_ONCE = 256};
    T cpu[COPY_AT_ONCE] = {0};
    T sum = _identity<Reduction, T>();

    for(int k = 0; k < n; k += COPY_AT_ONCE)
    {
        int from = k;
        int to = MIN(k + COPY_AT_ONCE, n);
        cudaMemcpy(cpu, input + from, sizeof(T)*(to - from), cudaMemcpyDeviceToHost);
        
        for(int i = 0; i < to - from; i++)
            sum = _reduce<Reduction, T>(sum, cpu[i]);
    }

    return sum;
}

//============================= TESTS =================================
#if (defined(TEST_CUDA_ALL) || defined(TEST_CUDA_REDUCTION)) && !defined(TEST_CUDA_REDUCTION_IMPL)
#define TEST_CUDA_REDUCTION_IMPL

template<class Reduction, typename T>
static T cpu_fold_reduce(const T *input, int n, Reduction reduce_tag = Reduction())
{
    T sum = _identity<Reduction, T>();
    if(n > 0)
    {
        static T* copy = 0;
        static int local_capacity = 0;
        if(local_capacity < n)
        {
            copy = (T*) realloc(copy, n*sizeof(T));
            local_capacity = n;
        }

        cudaMemcpy(copy, input, sizeof(T)*n, cudaMemcpyDeviceToHost);
        for(int range = n; range > 1; range /= 2)
        {
            for(int i = 0; i < range/2 ; i ++)
                copy[i] = _reduce<Reduction, T>(copy[2*i], copy[2*i + 1]);

            if(range%2)
                copy[range/2 - 1] = _reduce<Reduction, T>(copy[range/2 - 1], copy[range - 1]);
        }

        sum  = copy[0];
    }

    return sum;
}

static bool is_near(double a, double b, double epsilon = 1e-8)
{
    //this form guarantees that is_nearf(NAN, NAN, 1) == true
    return !(fabs(a - b) > epsilon);
}

//Returns true if x and y are within epsilon distance of each other.
//If |x| and |y| are less than 1 uses epsilon directly
//else scales epsilon to account for growing floating point inaccuracy
static bool is_near_scaled(double x, double y, double epsilon = 1e-8)
{
    //This is the form that produces the best assembly
    double calced_factor = fabs(x) + fabs(y);
    double factor = 2 > calced_factor ? 2 : calced_factor;
    return is_near(x, y, factor * epsilon / 2);
}

template<typename T>
static bool _is_approx_equal(T a, T b, double epsilon = sizeof(T) == 8 ? 1e-8 : 1e-5)
{
    if constexpr(std::is_integral_v<T>)
        return a == b;
    else if constexpr(std::is_floating_point_v<T>)
        return is_near_scaled(a, b, epsilon);
    else
        return false;
}


#include "cuda_random.cuh"
#include "cuda_runtime.h"
template<typename T>
static void test_reduce_type(uint64_t seed, const char* type_name)
{
    uint Ns[] = {0, 1, 5, 31, 32, 33, 64, 65, 256, 257, 512, 513, 1023, 1024, 1025, 256*256, 1024*1024 - 1, 1024*1024};

    //Find max size 
    uint N = 0;
    for(int i = 0; i < STATIC_ARRAY_SIZE(Ns); i++)
        if(N < Ns[i])
            N = Ns[i];

    //generate max sized map of radonom data (using 64 bits for higher precision).
    Cache_Tag tag = cache_tag_make();
    uint64_t* rand_state = cache_alloc(uint64_t, N, &tag);
    T*        rand = cache_alloc(T, N, &tag);
    random_map_seed_64(rand_state, N, seed);
    random_map_64(rand, rand_state, N);

    //test each size on radom data.
    for(int i = 0; i < STATIC_ARRAY_SIZE(Ns); i++)
    {
        uint n = Ns[i];
        LOG_INFO("kernel", "test_reduce_type<%s>: n:%lli", type_name, (lli)n);

        T sum0 = cpu_reduce(rand, n, Reduce::ADD);
        T sum1 = cpu_fold_reduce(rand, n, Reduce::ADD);
        T sum2 = thrust_reduce(rand, n, Reduce::ADD);
        T sum3 = reduce(rand, n, Reduce::ADD);

        //naive cpu sum reduce diverges from the true values for large n due to
        // floating point rounding
        if(n < 256) TEST(_is_approx_equal(sum1, sum0));

        TEST(_is_approx_equal(sum1, sum2, 1e-3)); //thrust gives inaccuarte results...
        TEST(_is_approx_equal(sum1, sum3));

        T min0 = cpu_reduce(rand, n, Reduce::MIN);
        T min1 = cpu_fold_reduce(rand, n, Reduce::MIN);
        T min2 = thrust_reduce(rand, n, Reduce::MIN);
        T min3 = reduce(rand, n, Reduce::MIN);

        TEST(_is_approx_equal(min1, min0));
        TEST(_is_approx_equal(min1, min2));
        TEST(_is_approx_equal(min1, min3));

        T max0 = cpu_reduce(rand, n, Reduce::MAX);
        T max1 = cpu_fold_reduce(rand, n, Reduce::MAX);
        T max2 = thrust_reduce(rand, n, Reduce::MAX);
        T max3 = reduce(rand, n, Reduce::MAX);

        TEST(_is_approx_equal(max1, max0));
        TEST(_is_approx_equal(max1, max2));
        TEST(_is_approx_equal(max1, max3));

        if constexpr(std::is_integral_v<T>)
        {
            T and0 = cpu_reduce(rand, n, Reduce::AND);
            T and1 = cpu_fold_reduce(rand, n, Reduce::AND);
            T and2 = thrust_reduce(rand, n, Reduce::AND);
            T and3 = reduce(rand, n, Reduce::AND);

            TEST(and1 == and0);
            TEST(and1 == and2);
            TEST(and1 == and3);

            T or0 = cpu_reduce(rand, n, Reduce::OR);
            T or1 = cpu_fold_reduce(rand, n, Reduce::OR);
            T or2 = thrust_reduce(rand, n, Reduce::OR);
            T or3 = reduce(rand, n, Reduce::OR);

            TEST(or1 == or0);
            TEST(or1 == or2);
            TEST(or1 == or3);

            T xor0 = cpu_reduce(rand, n, Reduce::XOR);
            T xor1 = cpu_fold_reduce(rand, n, Reduce::XOR);
            T xor2 = thrust_reduce(rand, n, Reduce::XOR);
            T xor3 = reduce(rand, n, Reduce::XOR);

            TEST(xor1 == xor0);
            TEST(xor1 == xor2);
            TEST(xor1 == xor3);
        }
    }

    cache_free(&tag);
    LOG_OKAY("kernel", "test_reduce_type<%s>: success!", type_name);
}

static void test_identity()
{
    ASSERT((_identity<Reduce::Add, double>() == 0));
    ASSERT((_identity<Reduce::Min, double>() == INFINITY));
    ASSERT((_identity<Reduce::Max, double>() == -INFINITY));

    ASSERT((_identity<Reduce::Add, int>() == 0));
    ASSERT((_identity<Reduce::Min, int>() == INT_MAX));
    ASSERT((_identity<Reduce::Max, int>() == INT_MIN));

    ASSERT((_identity<Reduce::Add, unsigned>() == 0));
    ASSERT((_identity<Reduce::Min, unsigned>() == UINT_MAX));
    ASSERT((_identity<Reduce::Max, unsigned>() == 0));

    ASSERT((_identity<Reduce::Or, unsigned>() == 0));
    ASSERT((_identity<Reduce::And, unsigned>() == 0xFFFFFFFFU));
    ASSERT((_identity<Reduce::Xor, unsigned>() == 0));
    LOG_OKAY("kernel", "test_identity: success!");
}

static void test_reduce(uint64_t seed)
{
    test_identity();
    //When compiling with thrust enabled this thing completely halts
    // the compiler...
    #ifndef COMPILE_THRUST
    test_reduce_type<char>(seed, "char");
    test_reduce_type<unsigned char>(seed, "unsigned");
    test_reduce_type<short>(seed, "short");
    test_reduce_type<ushort>(seed, "ushort");
    test_reduce_type<int>(seed, "int");
    #endif
    test_reduce_type<uint>(seed, "uint");
    test_reduce_type<float>(seed, "float");
    test_reduce_type<double>(seed, "double");
}
#endif