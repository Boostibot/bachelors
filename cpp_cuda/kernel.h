#pragma once

#include <cuda_runtime.h>

typedef float real_t;

typedef struct Allen_Cahn_Params{
    int mesh_size_x;
    int mesh_size_y;
    real_t sym_size;

    real_t dt;
    real_t L;
    real_t xi;
    real_t a;
    real_t b;
    real_t alpha;
    real_t beta;
    real_t Tm;
    real_t Tinit;
} Allen_Cahn_Params;

extern "C" cudaError_t kernel_step(real_t* phi_map_next, real_t* T_map_next, const real_t* phi_map, const real_t* T_map, Allen_Cahn_Params params, int device_processor_count, size_t iter);
