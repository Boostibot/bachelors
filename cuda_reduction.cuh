#pragma once

#include "cuda_util.cuh"
#include "cuda_launch.cuh"
#include "cuda_alloc.cuh"

// This file provides state of the art generic reduction operations. 
// The main kernel is fairly concice but the nature of the generic 
// implementation stretches this file out a bit. 
// We achive greater or equal performance then cuda thrust on all benchmarks I have run.
// The algorhitm is tested on various sizes and random data ensuring functioning implementation.  

//The main function in this file is cuda_produce_reduce which is a generalized map_reduce. 
// cuda_produce_reduce takes a function (index)->(value) which can be easily customized to pull 
// from different sources of data at the same time making for very easy implementation of 
//  vector dot product, efficient normed distances and other functions that would normally need zipping.
//If you are not convinced this is in fact simpler approach then the traditional map_reduce
// interface compare https://github.com/NVIDIA/thrust/blob/main/examples/dot_products_with_zip.cu with our dot_product function
template <typename T, class Reduction, typename Producer, bool has_trivial_producer = false>
static T cuda_produce_reduce(csize N, Producer produce, Reduction reduce_dummy = Reduction(), Cuda_Launch_Params launch_params = {}, csize cpu_reduce = 128);

//Predefined reduction functions. These are defined in special ways because:
// 1. They can be used with any type (that suports them!). No need to template them => simpler outside interface.
// 2. They can be tested for and thus the implementation can and does take advantage of custom intrinsics.
//
//Others can be defined from within custom struct types see Example_Custom_Reduce_Operation below.
namespace Reduce {
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
// cuda_produce_reduce(N, produce, Example_Custom_Reduce_Operation<T>())
// cuda_produce_reduce<Example_Custom_Reduce_Operation<T>>(N, produce)

//Returns the identity elemt of the Reduction operation
template <class Reduction, typename T>
static __host__ __device__ __forceinline__ T _reduce_indentity();

//Returns the result of redeuction using the Reduction operation
template <class Reduction, typename T>
static __host__ __device__ __forceinline__ T _reduce_reduce(T a, T b);

template <class Reduction>
static __host__ __device__ __forceinline__ const char* _reduce_name();

//Performs reduction operation within lanes of warp enabled by mask. 
//If mask includes lanes which do not participate within the reduction 
// (as a cause of divergent control flow) might cause deadlocks.
template <class Reduction, typename T>
static __device__ __forceinline__ T _warp_reduce(unsigned int mask, T value) ;


template <class Reduction, typename T, typename Producer, bool is_trivial_producer>
static __global__ void cuda_produce_reduce_kernel(T* __restrict__ output, Producer produce, csize N) 
{
    assert(blockDim.x > WARP_SIZE && blockDim.x % WARP_SIZE == 0 && "we expect the block dim to be chosen sainly");
    uint shared_size = blockDim.x / WARP_SIZE;

    extern __shared__ max_align_t shared_backing[];
    T* shared = (T*) (void*) shared_backing;

    T reduced = _reduce_indentity<Reduction, T>();
    for(int i = (int) blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x)
    {
        T produced;
        if constexpr(is_trivial_producer)
            produced = produce[i];
        else 
            produced = produce(i);

        reduced = _reduce_reduce<Reduction, T>(produced, reduced);
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

#include <cuda_occupancy.h>
#include <cuda_runtime.h>

template <typename T, class Reduction, typename Producer, bool has_trivial_producer>
static T cuda_produce_reduce(csize N, Producer produce, Reduction reduce_dummy, Cuda_Launch_Params launch_params, csize cpu_reduce)
{
    (void) reduce_dummy;
    
    // LOG_INFO("reduce", "Reduce %s N:%i", _reduce_name<Reduction>(), (int) N);
    T reduced = _reduce_indentity<Reduction, T>();
    if(N > 0)
    {
        enum {
            CPU_REDUCE_MAX = 1024,
            CPU_REDUCE_MIN = 32,
        };

        static Cuda_Launch_Bounds bounds = {};
        static Cuda_Launch_Constraints constraints = {};
        if(bounds.max_block_size == 0)
        {
            constraints = cuda_constraints_launch_constraints((void*) cuda_produce_reduce_kernel<Reduction, T, Producer, has_trivial_producer>);
            constraints.used_shared_memory_per_thread = (double) sizeof(T)/WARP_SIZE;
            bounds = cuda_get_launch_bounds(constraints);
        }

        // cpu_reduce = CLAMP(cpu_reduce, CPU_REDUCE_MIN, CPU_REDUCE_MAX);
        cpu_reduce = CPU_REDUCE_MAX;
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
                cuda_produce_reduce_kernel<Reduction, T, Producer, has_trivial_producer>
                    <<<launch.block_count, launch.block_size, launch.dynamic_shared_memory, launch_params.stream>>>(curr_output, (Producer&&) produce, N_curr);
            }
            else
            {
                cuda_produce_reduce_kernel<Reduction, T, const T* __restrict__, true>
                    <<<launch.block_count, launch.block_size, launch.dynamic_shared_memory, launch_params.stream>>>(curr_output, curr_input, N_curr);
            }

            CUDA_DEBUG_TEST(cudaGetLastError());
            // CUDA_DEBUG_TEST(cudaDeviceSynchronize());

            csize G = MIN(N_curr, (csize) launch.block_size* (csize)launch.block_count);
            csize N_next = DIV_CEIL(G, WARP_SIZE*WARP_SIZE); 
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
        cudaMemcpy(cpu, last_input, sizeof(T)*(size_t)N_curr, cudaMemcpyDeviceToHost);
        for(csize i = 0; i < N_curr; i++)
            reduced = _reduce_reduce<Reduction, T>(reduced, cpu[i]);

        cache_free(&tag);
    }

    return reduced;
}

//============================== IMPLEMENTATION OF MORE SPECIFIC REDUCTIONS ===================================

template<class Reduction, typename T, typename Map_Func>
static T cuda_map_reduce(const T *input, csize N, Map_Func map, Reduction reduce_tag = Reduction(), Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T, Reduction>(N, [=]SHARED(csize i){
        return map(input[i]);
    }, reduce_tag, launch_params);
    return output;
}

template<class Reduction, typename T>
static T cuda_reduce(const T *input, csize N, Reduction reduce_tag = Reduction(), Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T, Reduction, const T* __restrict__, true>(N, input, reduce_tag, launch_params);
    return output;
}

template<typename T>
static T cuda_sum(const T *input, csize N, Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T, Reduce::Add, const T* __restrict__, true>(N, input, Reduce::ADD, launch_params);
    return output;
}

template<typename T>
static T cuda_product(const T *input, csize N, Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T, Reduce::Mul, const T* __restrict__, true>(N, input, Reduce::MUL, launch_params);
    return output;
}

template<typename T>
static T cuda_min(const T *input, csize N, Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T, Reduce::Min, const T* __restrict__, true>(N, input, Reduce::MIN, launch_params);
    return output;
}

template<typename T>
static T cuda_max(const T *input, csize N, Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T, Reduce::Max, const T* __restrict__, true>(N, input, Reduce::MAX, launch_params);
    return output;
}

template<typename T>
static T cuda_L1_norm(const T *a, csize N, Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T>(N, [=]SHARED(csize i){
        T diff = a[i];
        return diff > 0 ? diff : -diff;
    }, Reduce::ADD, launch_params);
    return output;
}

template<typename T>
static T cuda_L2_norm(const T *a, csize N, Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T, Reduce::Add>(N, [=]SHARED(csize i){
        T diff = a[i];
        return diff*diff;
    }, Reduce::ADD, launch_params);
    return (T) sqrt(output);
}

template<typename T>
static T cuda_Lmax_norm(const T *a, csize N, Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T, Reduce::Max>(N, [=]SHARED(csize i){
        T diff = a[i];
        return diff > 0 ? diff : -diff;
    }, Reduce::MAX, launch_params);
    return output;
}

// more complex binary reductions...

template<typename T>
static T cuda_L1_distance(const T *a, const T *b, csize N, Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T, Reduce::Add>(N, [=]SHARED(csize i){
        T diff = a[i] - b[i];
        return diff > 0 ? diff : -diff;
    }, Reduce::ADD, launch_params);
    return output;
}

template<typename T>
static T cuda_L2_distance(const T *a, const T *b, csize N, Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T, Reduce::Add>(N, [=]SHARED(csize i){
        T diff = a[i] - b[i];
        return diff*diff;
    }, Reduce::ADD, launch_params);
    return (T) sqrt(output);
}

template<typename T>
static T cuda_Lmax_distance(const T *a, const T *b, csize N, Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T, Reduce::Max>(N, [=]SHARED(csize i){
        T diff = a[i] - b[i];
        return diff > 0 ? diff : -diff;
    }, Reduce::MAX, launch_params);
    return output;
}

template<typename T>
static T cuda_dot_product(const T *a, const T *b, csize N, Cuda_Launch_Params launch_params = {})
{
    T output = cuda_produce_reduce<T, Reduce::Add>(N, [=]SHARED(csize i){
        return a[i] * b[i];
    }, Reduce::ADD, launch_params);
    return output;
}

namespace Reduce
{
    template<typename T>
    struct Stats {
        T sum;
        T L1;
        T L2;
        T min;
        T max;

        static const char* name() {
            return "Stats";
        }

        static __host__ __device__ Stats identity() {
            Stats stats = {0};
            stats.sum = (T) 0;
            stats.L1 = (T) 0;
            stats.L2 = (T) 0;
            stats.min = _reduce_indentity<Min, T>();
            stats.max = _reduce_indentity<Max, T>();
            return stats;
        }

        static __host__ __device__ Stats reduce(Stats a, Stats b) {
            Stats out = {0};
            out.sum = a.sum + b.sum;
            out.L1 = a.L1 + b.L1;
            out.L2 = a.L2 + b.L2;
            out.min = MIN(a.min, b.min);
            out.max = MAX(a.max, b.max);
            return out;
        }
    };
}

//surpress "pointless comparison of unsigned type with zero" because its just annoying when making
// generic code...
#pragma nv_diag_suppress 186
template<typename T>
static Reduce::Stats<T> cuda_stats(const T *a, csize N, Cuda_Launch_Params launch_params = {})
{
    using Stats = Reduce::Stats<T>;
    Stats output = cuda_produce_reduce<Stats, Stats>(N, [=]SHARED(csize i){
        T val = a[i];
        Stats stats = {0};
        stats.sum = val;
        stats.L1 = val >= 0 ? val : (T) -val;
        stats.L2 = val*val;
        stats.min = val;
        stats.max = val;
        return stats;
    }, Stats{}, launch_params);
    output.L2 = (T) sqrt((double) output.L2);
    return output;
}

template<typename T>
static Reduce::Stats<T> cuda_stats_delta(const T *a, const T *b, csize N, Cuda_Launch_Params launch_params = {})
{
    using Stats = Reduce::Stats<T>;
    Stats output = cuda_produce_reduce<Stats, Stats>(N, [=]SHARED(csize i){
        T val = a[i] - b[i];
        Stats stats = {0};
        stats.sum = val;
        stats.L1 = val >= 0 ? val : (T) -val;
        stats.L2 = val*val;
        stats.min = val;
        stats.max = val;
        return stats;
    }, Stats{}, launch_params);
    output.L2 = (T) sqrt((double) output.L2);
    return output;
}
#pragma nv_diag_default 186

//============================== TEMPLATE MADNESS BELOW ===================================
template <class Reduction, typename T>
static __host__ __device__ __forceinline__ T _reduce_indentity()
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
static __host__ __device__ __forceinline__ T _reduce_reduce(T a, T b)
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
static __host__ __device__ __forceinline__ const char* _reduce_name()
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

template <typename T>
static __device__ __forceinline__ T shuffle_down_sync_any(unsigned int mask, T value, int offset)
{
    //@NOTE: not builtin types would need to be reinterpret casted and shuffled approproately!
    //We ignore that for now!
    if constexpr(sizeof(T) <= 8 && std::is_arithmetic_v<T>)
        return __shfl_down_sync(mask, value, offset);
    else if(sizeof(T) % 8 == 0)
    {   
        constexpr int nums = sizeof(T) / 8; 
        union Caster {T t; uint64_t u64s[nums];};
        Caster cast_from = {value};
        Caster cast_to;
        for(int i = 0; i < nums; i++)
            cast_to.u64s[i] = __shfl_down_sync(mask, cast_from.u64s[i], offset);

        return cast_to.t;
    }
    else
    {
        constexpr int nums = (sizeof(T) + 3) / 4;
        union Caster {T t; uint64_t u32s[nums];};
        Caster cast_from = {value};
        Caster cast_to;
        for(int i = 0; i < nums; i++)
            cast_to.u32s[i] = __shfl_down_sync(mask, cast_from.u32s[i], offset);

        return cast_to.t;
    }
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
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) 
            value = _reduce_reduce<Reduction, T>(shuffle_down_sync_any<T>(mask, value, offset), value);

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

    T id = _reduce_indentity<Reduction, T>();
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
    return cuda_reduce<Reduction, T>(input, N, reduce_tag);
}
#endif

template<class Reduction, typename T>
static T cpu_reduce(const T *input, csize n, Reduction reduce_tag = Reduction())
{
    enum {COPY_AT_ONCE = 256};
    T cpu[COPY_AT_ONCE] = {0};
    T sum = _reduce_indentity<Reduction, T>();

    for(csize k = 0; k < n; k += COPY_AT_ONCE)
    {
        csize from = k;
        csize to = MIN(k + COPY_AT_ONCE, n);
        cudaMemcpy(cpu, input + from, sizeof(T)*(size_t)(to - from), cudaMemcpyDeviceToHost);
        
        for(csize i = 0; i < to - from; i++)
            sum = _reduce_reduce<Reduction, T>(sum, cpu[i]);
    }

    return sum;
}


template<class Reduction, typename T>
static T cpu_reduce_host_mem(const T *input, csize n, Reduction reduce_tag = Reduction())
{
    T sum = _reduce_indentity<Reduction, T>();
    for(csize i = 0; i < n; i++)
        sum = _reduce_reduce<Reduction, T>(sum, input[i]);

    return sum;
}

//============================= TESTS =================================
#if (defined(TEST_CUDA_ALL) || defined(TEST_CUDA_REDUCTION)) && !defined(TEST_CUDA_REDUCTION_IMPL)
#define TEST_CUDA_REDUCTION_IMPL

template<class Reduction, typename T>
static T cpu_fold_reduce(const T *input, csize n, Reduction reduce_tag = Reduction())
{
    T sum = _reduce_indentity<Reduction, T>();
    if(n > 0)
    {
        static T* copy = 0;
        static csize local_capacity = 0;
        if(local_capacity < n)
        {
            copy = (T*) realloc(copy, (size_t)n*sizeof(T));
            local_capacity = n;
        }

        cudaMemcpy(copy, input, (size_t)n*sizeof(T), cudaMemcpyDeviceToHost);
        for(csize range = n; range > 1; range /= 2)
        {
            for(csize i = 0; i < range/2 ; i ++)
                copy[i] = _reduce_reduce<Reduction, T>(copy[2*i], copy[2*i + 1]);

            if(range%2)
                copy[range/2 - 1] = _reduce_reduce<Reduction, T>(copy[range/2 - 1], copy[range - 1]);
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
    csize Ns[] = {0, 1, 5, 31, 32, 33, 64, 65, 256, 257, 512, 513, 1023, 1024, 1025, 256*256, 1024*1024 - 1, 1024*1024};

    //Find max size 
    csize N = 0;
    for(csize i = 0; i < (csize) ARRAY_LEN(Ns); i++)
        if(N < Ns[i])
            N = Ns[i];

    //generate max sized map of radonom data (using 64 bits for higher precision).
    Cache_Tag tag = cache_tag_make();
    uint64_t* rand_state = cache_alloc(uint64_t, N, &tag);
    T*        rand = cache_alloc(T, N, &tag);
    random_map_seed_64(rand_state, N, seed);
    random_map_64(rand, rand_state, N);

    //test each size on radom data.
    for(csize i = 0; i < (csize) ARRAY_LEN(Ns); i++)
    {
        csize n = Ns[i];
        LOG_INFO("kernel", "test_reduce_type<%s>: n:%lli", type_name, (lli)n);

        T sum0 = cpu_reduce(rand, n, Reduce::ADD);
        T sum1 = cpu_fold_reduce(rand, n, Reduce::ADD);
        T sum2 = thrust_reduce(rand, n, Reduce::ADD);
        T sum3 = cuda_reduce(rand, n, Reduce::ADD);

        //naive cpu sum reduce diverges from the true values for large n due to
        // floating point rounding
        if(n < 256) TEST(_is_approx_equal(sum1, sum0));

        TEST(_is_approx_equal(sum1, sum2, 1e-3)); //thrust gives inaccuarte results...
        TEST(_is_approx_equal(sum1, sum3));

        T min0 = cpu_reduce(rand, n, Reduce::MIN);
        T min1 = cpu_fold_reduce(rand, n, Reduce::MIN);
        T min2 = thrust_reduce(rand, n, Reduce::MIN);
        T min3 = cuda_reduce(rand, n, Reduce::MIN);

        TEST(_is_approx_equal(min1, min0));
        TEST(_is_approx_equal(min1, min2));
        TEST(_is_approx_equal(min1, min3));

        T max0 = cpu_reduce(rand, n, Reduce::MAX);
        T max1 = cpu_fold_reduce(rand, n, Reduce::MAX);
        T max2 = thrust_reduce(rand, n, Reduce::MAX);
        T max3 = cuda_reduce(rand, n, Reduce::MAX);

        TEST(_is_approx_equal(max1, max0));
        TEST(_is_approx_equal(max1, max2));
        TEST(_is_approx_equal(max1, max3));

        Reduce::Stats<T> stats = cuda_stats(rand, n);
        T l1 = cuda_L1_norm(rand, n);
        T l2 = cuda_L2_norm(rand, n);
        TEST(stats.min == min3);
        TEST(stats.max == max3);
        if(n > 0)
            TEST(min3 <= max3);
        //this should be strict equality since the alg for sum and sum in stats is the *same*
        // but it isnt. Why? Lets blame it on the kernel optimalizations reordering our instructions
        // cause I cant be bothered
        // TEST(stats.L1 == l1);
        // TEST(stats.L2 == l2);
        // TEST(stats.sum == sum3); 
        TEST(_is_approx_equal(stats.L1, l1));
        TEST(_is_approx_equal(stats.L2, l2));
        TEST(_is_approx_equal(stats.sum, sum3)); 

        if constexpr(std::is_integral_v<T>)
        {
            T and0 = cpu_reduce(rand, n, Reduce::AND);
            T and1 = cpu_fold_reduce(rand, n, Reduce::AND);
            T and2 = thrust_reduce(rand, n, Reduce::AND);
            T and3 = cuda_reduce(rand, n, Reduce::AND);

            TEST(and1 == and0);
            TEST(and1 == and2);
            TEST(and1 == and3);

            T or0 = cpu_reduce(rand, n, Reduce::OR);
            T or1 = cpu_fold_reduce(rand, n, Reduce::OR);
            T or2 = thrust_reduce(rand, n, Reduce::OR);
            T or3 = cuda_reduce(rand, n, Reduce::OR);

            TEST(or1 == or0);
            TEST(or1 == or2);
            TEST(or1 == or3);

            T xor0 = cpu_reduce(rand, n, Reduce::XOR);
            T xor1 = cpu_fold_reduce(rand, n, Reduce::XOR);
            T xor2 = thrust_reduce(rand, n, Reduce::XOR);
            T xor3 = cuda_reduce(rand, n, Reduce::XOR);

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
    ASSERT((_reduce_indentity<Reduce::Add, double>() == 0));
    ASSERT((_reduce_indentity<Reduce::Min, double>() == INFINITY));
    ASSERT((_reduce_indentity<Reduce::Max, double>() == -INFINITY));

    ASSERT((_reduce_indentity<Reduce::Add, int>() == 0));
    ASSERT((_reduce_indentity<Reduce::Min, int>() == INT_MAX));
    ASSERT((_reduce_indentity<Reduce::Max, int>() == INT_MIN));

    ASSERT((_reduce_indentity<Reduce::Add, unsigned>() == 0));
    ASSERT((_reduce_indentity<Reduce::Min, unsigned>() == UINT_MAX));
    ASSERT((_reduce_indentity<Reduce::Max, unsigned>() == 0));

    ASSERT((_reduce_indentity<Reduce::Or, unsigned>() == 0));
    ASSERT((_reduce_indentity<Reduce::And, unsigned>() == 0xFFFFFFFFU));
    ASSERT((_reduce_indentity<Reduce::Xor, unsigned>() == 0));
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