#pragma once

#include "cuda_util.cuh"
#include "cuda_for.cuh"

//This file provides some basic hashing & randomness generation functions.
//Both of which are not ment for any sort of cryptographic security!
//We provide 64 and 32 bit variants as 64 bit integers are still quite slow
// on cuda GPUs (often emulated instructions) but they provide more precision
// for testing purposes. As such the choice of 64/32 bit must be ultimately
// determined by the user.

//The main interface is given by the following two functions

//Seeds a random state filling it with values depending on the seed.
static void random_map_seed_32(uint32_t* rand_state, csize N, uint32_t seed);
static void random_map_seed_64(uint32_t* rand_state, csize N, uint64_t seed);

//Saves random values to the provided input depending on the type. Updates rand_state. 
// For integer types up to sizeof(*rand_state) the entire range is used.
// For integer types from to sizeof(*rand_state) only the range till the maximum value of rand_state type is used.
// For floating point types the values lie evenly distributed (exponent is constant!) within the range [0, 1).
template<typename T>
static void random_map_32(T* output, uint32_t* rand_state, csize N);
template<typename T>
static void random_map_64(T* output, uint64_t* rand_state, csize N);


//Hashes a 64 bit value to 64 bit hash.
//Note that this function is bijective meaning it can be reversed.
//In particular 0 maps to 0.
static SHARED uint64_t hash_bijective_64(uint64_t value) 
{
    //source: https://stackoverflow.com/a/12996028
    uint64_t hash = value;
    hash = (hash ^ (hash >> 30)) * (uint64_t) 0xbf58476d1ce4e5b9;
    hash = (hash ^ (hash >> 27)) * (uint64_t) 0x94d049bb133111eb;
    hash = hash ^ (hash >> 31);
    return hash;
}

//Hashes a 32 bit value to 32 bit hash.
//Note that this function is bijective meaning it can be reversed.
//In particular 0 maps to 0.
static SHARED uint32_t hash_bijective_32(uint32_t value) 
{
    //source: https://stackoverflow.com/a/12996028
    uint32_t hash = value;
    hash = ((hash >> 16) ^ hash) * 0x119de1f3;
    hash = ((hash >> 16) ^ hash) * 0x119de1f3;
    hash = (hash >> 16) ^ hash;
    return hash;
}

//Mixes two prevously hashed values into one. 
//Yileds good results even when one of hash1 or hash2 is hashed badly.
//source: https://stackoverflow.com/a/27952689
static SHARED uint64_t hash_mix64(uint64_t hash1, uint64_t hash2)
{
    hash1 ^= hash2 + 0x517cc1b727220a95 + (hash1 << 6) + (hash1 >> 2);
    return hash1;
}

//source: https://stackoverflow.com/a/27952689
static SHARED uint32_t hash_mix32(uint32_t hash1, uint32_t hash2)
{
    hash1 ^= hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2);
    return hash1;
}

static SHARED uint32_t hash_pcg_32(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static SHARED uint32_t random_pcg_32(uint32_t* rng_state)
{
    uint32_t state = *rng_state;
    *rng_state = *rng_state * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

//Generates next random value
//Seed can be any value
//Taken from: https://prng.di.unimi.it/splitmix64.c
static SHARED uint64_t random_splitmix_64(uint64_t* state) 
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

static SHARED float _make_f32(uint32_t sign, uint32_t expoment, uint32_t mantissa)
{
    union {
        uint32_t u32;
        float f32;
    } caster = {0};

    caster.u32 = (sign << 31) | (expoment << 23) | mantissa;
    return caster.f32;
}

static SHARED double _make_f64(uint64_t sign, uint64_t expoment, uint64_t mantissa)
{
    union {
        uint64_t u64;
        double f64;
    } caster = {0};

    caster.u64 = (sign << 63) | (expoment << 52) | mantissa;
    return caster.f64;
}

static SHARED float random_bits_to_f32(uint32_t bits)
{
    uint64_t mantissa = bits >> (32 - 23); //grab top 23 bits
    float random_f32 = _make_f32(0, 127, (uint32_t) mantissa) - 1;
    return random_f32;
}

static SHARED double random_bits_to_f64(uint64_t bits)
{
    uint64_t mantissa = bits >> (64 - 52); //grab top 52 bit
    double random_f64 = _make_f64(0, 1023, mantissa) - 1;
    return random_f64;
}

//The actual c++ interface

template<typename T>
static SHARED T random_bits_to_value_32(uint32_t bits)
{
    if constexpr(std::is_integral_v<T>)
        return (T) bits;
    else if constexpr(std::is_floating_point_v<T>)
        return (T) random_bits_to_f32((uint32_t) bits);
    else
        static_assert(sizeof(T) == 0, "Expected T to be either an int or some kind of float." 
            "(sizeof(T) == 0 is always false but due to how c++ works its necessary)");
}

template<typename T>
static SHARED T random_bits_to_value_64(uint64_t bits)
{
    if constexpr(std::is_integral_v<T>)
        return (T) bits;
    else if constexpr(std::is_floating_point_v<T>)
    {
        if constexpr(sizeof(T) == 8)
            return (T) random_bits_to_f64(bits);
        else    
            return (T) random_bits_to_f32((uint32_t) bits);
    }
    else
        static_assert(sizeof(T) == 0, "Expected T to be either an int or some kind of float." 
            "(sizeof(T) == 0 is always false but due to how c++ works its necessary)");
}


static thread_local uint64_t global_random_state = (uint64_t) clock_ns();
static float random_f32(float from, float to)
{
    uint64_t bits = random_splitmix_64(&global_random_state);
    return random_bits_to_f32(bits)*(to - from) + from;
}

static int random_int(int from, int to)
{
    if(from >= to)
        return from;

    uint64_t bits = random_splitmix_64(&global_random_state);
    return (int) (bits % (uint64_t) (to - from)) + from;
}

//has 20% chance of returning one of the extremes. Both are equally likely. 
static int random_int_with_high_chance_of_extremes(int from, int to)
{
    if(from >= to)
        return from;

    uint64_t bits = random_splitmix_64(&global_random_state);
    if(bits % 5 == 0)
    {
        if(bits % 2 == 0)
            return from;
        else
            return to;
    }
    return (int) (bits % (uint64_t) (to - from)) + from;
}

static void random_map_seed_32(uint32_t* rand_state, csize N, uint32_t seed)
{
    cuda_for(0, N, [=]SHARED(csize i) {
        uint32_t hashed_index = hash_bijective_32((uint32_t) i);
        rand_state[i] = hash_mix32(hashed_index, seed);
    });
}

static void random_map_seed_64(uint64_t* rand_state, csize N, uint64_t seed)
{
    cuda_for(0, N, [=]SHARED(csize i) {
        uint64_t hashed_index = hash_bijective_64((uint64_t) i);
        rand_state[i] = hash_mix64(hashed_index, seed);
    });
}

template<typename T>
static void random_map_32(T* output, uint32_t* rand_state, csize N)
{
    cuda_for(0, N, [=]SHARED(csize i) {
        uint32_t random_bits = random_pcg_32(&rand_state[i]); 
        output[i] = random_bits_to_value_32<T>(random_bits);
    });
}

template<typename T>
static void random_map_64(T* output, uint64_t* rand_state, csize N)
{
    cuda_for(0, N, [=]SHARED(csize i) {
        uint64_t random_bits = random_splitmix_64(&rand_state[i]); 
        output[i] = random_bits_to_value_64<T>(random_bits);
    });
}

static SHARED float lerp(float x, float y, float s)
{
    return x + s * (y-x);
}

static SHARED float smooth_lerp(float x, float y, float s)
{
    return lerp(x, y, s * s * (3-2*s));
}

static SHARED uint32_t hash_2d(uint32_t ix, uint32_t iy, uint32_t seed)
{
    uint32_t w = 8 * sizeof(uint32_t);
    uint32_t s = w / 2; // rotation width
    uint32_t a = ix ^ seed; 
    uint32_t b = iy ^ seed;
    a *= 3284157443; b ^= a << s | a >> w-s;
    b *= 1911520717; a ^= b << s | b >> w-s;
    a *= 2048419325;
    return a;
}

static SHARED float _perlin2d_gradient(float x, float y, uint32_t ix, uint32_t iy, uint32_t seed)
{
    // uint32_t rand = hash_mix32(x ^ seed, y) ^ seed;
    // uint32_t rand = hash_mix32(hash_pcg_32(ix), hash_pcg_32(iy ^ seed)) ^ seed;
    uint32_t rand = hash_2d(ix, iy, seed);
    float rotation = random_bits_to_f32(rand)*2*3.14159265f;
    float grad_x = cosf(rotation);
    float grad_y = sinf(rotation);

    float dx = x - (float)ix;
    float dy = y - (float)iy;

    return dx*grad_x + dy*grad_y;
}

static SHARED float perlin2d(float x, float y, uint32_t seed)
{
    uint32_t xi = (uint32_t) (int) x;
    uint32_t yi = (uint32_t) (int) y;
    float sx = x - xi;
    float sy = y - yi;
    
    float s = _perlin2d_gradient(x, y, xi,   yi,   seed);
    float t = _perlin2d_gradient(x, y, xi+1, yi,   seed);
    float u = _perlin2d_gradient(x, y, xi,   yi+1, seed);
    float v = _perlin2d_gradient(x, y, xi+1, yi+1, seed);

    float top = smooth_lerp(s, t, sx);
    float bot = smooth_lerp(u, v, sx);
    return (smooth_lerp(top, bot, sy) + 1)/2;
}

static SHARED float simplex2d(float x, float y, uint32_t seed)
{
    uint32_t xi = (uint32_t) (int) x;
    uint32_t yi = (uint32_t) (int) y;
    float sx = x - xi;
    float sy = y - yi;
    
    float s = random_bits_to_f32(hash_2d(xi,   yi,   seed));
    float t = random_bits_to_f32(hash_2d(xi+1, yi,   seed));
    float u = random_bits_to_f32(hash_2d(xi,   yi+1, seed));
    float v = random_bits_to_f32(hash_2d(xi+1, yi+1, seed));

    float top = smooth_lerp(s, t, sx);
    float bot = smooth_lerp(u, v, sx);
    return smooth_lerp(top, bot, sy);
}

static SHARED float perlin2d_depth(float x, float y, int depth, uint32_t seed)
{
    float amplitude = 1.0;
    float sum = 0;
    float div = 0;
    for(int i = 0; i < depth; i++)
    {
        div += amplitude;
        sum += perlin2d(x/amplitude, y/amplitude, seed) * amplitude;
        amplitude /= 2;
    }

    return sum/div;
}

static SHARED float simplex2d_depth(float x, float y, int depth, uint32_t seed)
{
    float amplitude = 1.0;
    float sum = 0;
    float div = 0;
    for(int i = 0; i < depth; i++)
    {
        div += amplitude;
        sum += simplex2d(x/amplitude, y/amplitude, seed) * amplitude;
        amplitude /= 2;
    }

    return sum/div;
}

template<typename T>
static void perlin2d_generate(T* values, csize nx, csize ny, float freq_x, float freq_y, int depth, uint32_t seed)
{
    if(depth <= 0)
        depth = 1;

    cuda_for_2D(0, 0, nx, ny, [=]SHARED(int x, int y){
        values[x + y*nx] = (T) perlin2d_depth((float) x*freq_x/nx, (float) y*freq_y/ny, depth, seed);
    });

    //remap to [0, 1]
    T min = cuda_min(values, nx*ny);
    T max = cuda_max(values, nx*ny);

    if(min < max)
    {
        cuda_for_2D(0, 0, nx, ny, [=]SHARED(int x, int y){
            values[x + y*nx] = (values[x + y*nx] - min) / (max - min);
        });
    }
}

template<typename T>
static void simplex2d_generate(T* values, csize nx, csize ny, float freq_x, float freq_y, int depth, uint32_t seed)
{
    if(depth <= 0)
        depth = 1;

    cuda_for_2D(0, 0, nx, ny, [=]SHARED(int x, int y){
        values[x + y*nx] = (T) simplex2d_depth((float) x*freq_x/nx, (float) y*freq_y/ny, depth, seed);
    });
}
