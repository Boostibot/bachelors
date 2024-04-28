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
    uint32_t composite = (sign << 31) | (expoment << 23) | mantissa;
    float out = *(float*) (void*) &composite;
    return out;
}

static SHARED double _make_f64(uint64_t sign, uint64_t expoment, uint64_t mantissa)
{
    uint64_t composite = (sign << 63) | (expoment << 52) | mantissa;
    double out = *(double*) (void*) &composite;
    return out;
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