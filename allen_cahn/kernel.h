#pragma once

#include <cuda_runtime.h>

#define DO_FLOATS 

#ifdef DO_FLOATS
    typedef float real_t;
    #define REAL_FMT "%f"
    #define REAL_FMT_LOW_PREC "%.3f"
#else
    typedef double real_t;
    #define REAL_FMT "%lf"
    #define REAL_FMT_LOW_PREC "%.3lf"
#endif

typedef struct Allen_Cahn_Params{
    int mesh_size_x;
    int mesh_size_y;
    real_t L0; //simulation region size in real units

    real_t dt; //time step
    real_t L;  //latent heat
    real_t xi; //boundary thickness
    real_t a;  //
    real_t b;
    real_t alpha;
    real_t beta;
    real_t Tm; //melting point
    real_t Tinit; //currenlty unsused

    real_t S; //anisotrophy strength
    real_t m; //anisotrophy frequency (?)
    real_t theta0; //anisotrophy orientation
    bool do_anisotropy;
} Allen_Cahn_Params;

#define ALLEN_CAHN_HISTORY 3
#define ALLEN_CAHN_DEBUG_MAPS 8
#define ALLEN_CAHN_DEBUG_MAP_NAME_LEN 32

typedef struct Allen_Cahn_Maps{
    real_t* Phi[ALLEN_CAHN_HISTORY];
    real_t* T[ALLEN_CAHN_HISTORY];

    real_t* debug_maps[ALLEN_CAHN_DEBUG_MAPS];
    char debug_names[ALLEN_CAHN_DEBUG_MAPS][ALLEN_CAHN_DEBUG_MAP_NAME_LEN + 1];
    bool debug_request[ALLEN_CAHN_DEBUG_MAPS];
    bool debug_written[ALLEN_CAHN_DEBUG_MAPS];
} Allen_Cahn_Maps;

extern "C" cudaError_t kernel_step(Allen_Cahn_Maps* maps, Allen_Cahn_Params params, size_t iter);

//helper conversion functions for opengl compatibility (opengl cannot work with double values)
extern "C" cudaError_t kernel_float_from_double(float* output, const double* input, size_t size);
extern "C" cudaError_t kernel_double_from_float(double* output, const float* input, size_t size);