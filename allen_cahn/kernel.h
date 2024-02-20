#pragma once

#include <cuda_runtime.h>

typedef float real_t;

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

#define ALLEN_CAHN_DEBUG_MAPS 8
#define ALLEN_CAHN_DEBUG_MAP_NAME_LEN 32

typedef struct Allen_Cahn_Maps{
    real_t* Phi[2];
    real_t* T[2];

    real_t* debug_maps[ALLEN_CAHN_DEBUG_MAPS];
    char debug_names[ALLEN_CAHN_DEBUG_MAPS][ALLEN_CAHN_DEBUG_MAP_NAME_LEN + 1];
    bool debug_request[ALLEN_CAHN_DEBUG_MAPS];
    bool debug_written[ALLEN_CAHN_DEBUG_MAPS];
} Allen_Cahn_Maps;

extern "C" cudaError_t kernel_step(Allen_Cahn_Maps* maps, Allen_Cahn_Params params, int device_processor_count, size_t iter);
