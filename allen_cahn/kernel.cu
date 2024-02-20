#define JMAPI __host__ __device__ static inline

#include <device_launch_parameters.h>
#include "kernel.h"
#include <cmath>
#include <stdio.h>
#include <assert.h>

#define MOD(x, mod) (((x) % (mod) + (mod)) % (mod))

enum {
    DO_DEBUG = 0,
    MAP_GRAD_PHI = 0,
    MAP_GRAD_T = 1,
    MAP_REACTION = 2,
    MAP_ANISO_FACTOR = 3,
};

__host__ __device__ real_t* map_at(real_t* map, int x, int y, Allen_Cahn_Params params)
{
    int x_mod = MOD(x, params.mesh_size_x);
    int y_mod = MOD(y, params.mesh_size_y);

    return &map[x_mod + y_mod*params.mesh_size_x];
}

__host__ __device__ real_t f0(real_t phi)
{
	return phi*(1 - phi)*(phi - 1.0f/2);
}

__host__ __device__ real_t allen_cahn_reaction_term_1(real_t phi, real_t T, real_t xi, Allen_Cahn_Params params)
{
    real_t mK = 1;
	return (params.a*f0(phi) - params.b*params.beta*xi*(T - params.Tm))*mK;
}

__host__ __device__ real_t allen_cahn_reaction_term_2(real_t phi, real_t T, real_t xi, real_t grad_phi_x, real_t grad_phi_y, Allen_Cahn_Params params)
{
    real_t mK = 1;
	real_t grad_val = hypot(grad_phi_x, grad_phi_y);
	return (params.a*f0(phi) - params.b*params.beta*xi*xi*grad_val*(T - params.Tm))*mK;
}

#define ECHOF(x) printf(#x": %f\n", (x))

#define DO_ANISOTROPY

__device__ real_t g(real_t theta, real_t theta0, real_t m, real_t S)
{
    #ifdef DO_ANISOTROPY
        return 1.0f - S*cosf(m*(theta - theta0));
    #else
        return 1;
    #endif
}

#define PI 3.14159265359
#define TAU (2*PI)

__global__ void allen_cahn_simulate(real_t* Phi_map_next, real_t* T_map_next, real_t* Phi_map, real_t* T_map, Allen_Cahn_Maps maps, Allen_Cahn_Params params, size_t iter)
{
    real_t dx = (real_t) params.L0 / params.mesh_size_x;
    real_t dy = (real_t) params.L0 / params.mesh_size_y;
    real_t mK = dx * dy;

    real_t a = params.a;
    real_t b = params.b;
    real_t alpha = params.alpha;
    real_t beta = params.beta;
    real_t xi = params.xi;
    real_t Tm = params.Tm;
    real_t L = params.L; //Latent heat, not L0 (sym size) ! 
    real_t dt = params.dt;
    real_t S = params.S; //anisotrophy strength
    real_t m = params.m; //anisotrophy frequency (?)
    real_t theta0 = params.theta0;
    bool do_anisotropy = params.do_anisotropy;

    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.mesh_size_x; x += blockDim.x * gridDim.x) 
    {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.mesh_size_x; y += blockDim.y * gridDim.y) 
        {
            real_t T = *map_at(T_map, x, y, params);
            real_t Phi = *map_at(Phi_map, x, y, params);

            real_t Phi_U = *map_at(Phi_map, x, y + 1, params);
            real_t Phi_D = *map_at(Phi_map, x, y - 1, params);
            real_t Phi_R = *map_at(Phi_map, x + 1, y, params);
            real_t Phi_L = *map_at(Phi_map, x - 1, y, params);

            real_t T_U = *map_at(T_map, x, y + 1, params);
            real_t T_D = *map_at(T_map, x, y - 1, params);
            real_t T_R = *map_at(T_map, x + 1, y, params);
            real_t T_L = *map_at(T_map, x - 1, y, params);

            real_t grad_T_x = dy*(T_R - T_L);
            real_t grad_T_y = dx*(T_U - T_D);

            real_t grad_Phi_x = dy*(Phi_R - Phi_L);
            real_t grad_Phi_y = dx*(Phi_U - Phi_D);

            real_t grad_T_norm = hypotf(grad_T_x, grad_T_y);
            real_t grad_Phi_norm = hypotf(grad_Phi_x, grad_Phi_y);

            real_t g_theta = 1;
            real_t grad_Phi_y_norm = grad_Phi_y / grad_Phi_norm;
            if(do_anisotropy)
            {
                //prevent nans
                if(grad_Phi_norm > 0.0001)
                {
                    real_t theta = asinf(grad_Phi_y_norm);
                    g_theta = 1.0f - S*cosf(m*theta + theta0);
                }
            }

            real_t int_K_laplace_T   = dy/dx*(T_L - 2*T + T_R)     + dx/dy*(T_D - 2*T + T_U);
            real_t int_K_laplace_Phi = dy/dx*(Phi_L - 2*Phi + Phi_R) + dx/dy*(Phi_D - 2*Phi + Phi_U);
            real_t int_K_f = g_theta*a*mK*f0(Phi) - b*xi*xi*beta*(T - Tm)*grad_Phi_norm/2;

            real_t int_K_dt_Phi = g_theta/alpha*int_K_laplace_Phi + 1/(xi*xi * alpha)*int_K_f;
            real_t int_K_dt_T = int_K_laplace_T + L*int_K_dt_Phi;

            real_t dt_Phi = 1/mK*int_K_dt_Phi;
            real_t dt_T = 1/mK*int_K_dt_T;

            real_t Phi_next = Phi + dt*dt_Phi;
            real_t T_next = T + dt*dt_T;
        
            if(DO_DEBUG)
            {
                if(maps.debug_request[MAP_GRAD_PHI])
                    *map_at(maps.debug_maps[MAP_GRAD_PHI], x, y, params) = hypotf(Phi_R - Phi_L, Phi_U - Phi_D);

                if(maps.debug_request[MAP_GRAD_T])
                    *map_at(maps.debug_maps[MAP_GRAD_T], x, y, params) = hypotf(T_R - T_L, T_U - T_D);

                if(maps.debug_request[MAP_REACTION])
                    *map_at(maps.debug_maps[MAP_REACTION], x, y, params) = int_K_f / mK;

                if(maps.debug_request[MAP_ANISO_FACTOR])
                    *map_at(maps.debug_maps[MAP_ANISO_FACTOR], x, y, params) = g_theta;
            }

            *map_at(Phi_map_next, x, y, params) = Phi_next;
            *map_at(T_map_next, x, y, params) = T_next;
        }
    }
}

#define CUDA_ERR_AND(err) (err) != cudaSuccess ? (err) :

extern "C" cudaError_t kernel_step(Allen_Cahn_Maps* maps, Allen_Cahn_Params params, int device_processor_count, size_t iter)
{
    real_t* Phi_map_next = maps->Phi[(iter + 1) % 2];
    real_t* Phi_map = maps->Phi[(iter) % 2];
    
    real_t* T_map_next = maps->T[(iter + 1) % 2];
    real_t* T_map = maps->T[(iter) % 2];

    dim3 bs(64, 1);
    dim3 grid(device_processor_count, 1);
    allen_cahn_simulate<<<grid, bs>>>(Phi_map_next, T_map_next, Phi_map, T_map, *maps, params, iter);

    memset(maps->debug_names, 0, sizeof maps->debug_names); 
    #define ASSIGN_MAP_NAME(maps, name) \
        memcpy((maps)->debug_names[(name)], #name, sizeof(#name));
    ASSIGN_MAP_NAME(maps, MAP_GRAD_PHI);
    ASSIGN_MAP_NAME(maps, MAP_GRAD_T);
    ASSIGN_MAP_NAME(maps, MAP_REACTION);
    ASSIGN_MAP_NAME(maps, MAP_ANISO_FACTOR);

    cudaError_t out = cudaSuccess;
    out = CUDA_ERR_AND(out) cudaGetLastError();
    out = CUDA_ERR_AND(out) cudaDeviceSynchronize();
    return out;
}