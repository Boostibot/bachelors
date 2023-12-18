#pragma once

#include "lib/defines.h"
#include <cuda_runtime.h>

typedef struct Allen_Cahn_Params{
    i32 mesh_size_x;
    i32 mesh_size_y;
    f32 sym_size;

    f32 dt;
    f32 L;
    f32 xi;
    f32 a;
    f32 b;
    f32 alpha;
    f32 beta;
    f32 Tm;
    f32 Tinit;
} Allen_Cahn_Params;


extern "C" cudaError_t kernel_step(f32* phi_map_next, f32* T_map_next, const f32* phi_map, const f32* T_map, Allen_Cahn_Params params, int device_processor_count, isize iter);

