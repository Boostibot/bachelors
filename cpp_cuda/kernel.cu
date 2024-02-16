#define JMAPI __host__ __device__ static inline

#include <device_launch_parameters.h>
#include "kernel.h"
#include <cmath>
#include <stdio.h>
#include <assert.h>

__host__ __device__ real_t map_at(const real_t* map, int x, int y, Allen_Cahn_Params params)
{
    int x_mod = x % params.mesh_size_x;
    int y_mod = y % params.mesh_size_y;

    return map[x_mod + y_mod*params.mesh_size_x];
}

__host__ __device__ real_t allen_cahn_reaction_term_0(real_t phi)
{
	return phi*(1 - phi)*(phi - 1.0f/2);
}

__host__ __device__ real_t allen_cahn_reaction_term_1(real_t phi, real_t T, real_t xi, Allen_Cahn_Params params)
{
    real_t mK = 1;
	return (params.a*allen_cahn_reaction_term_0(phi) - params.b*params.beta*xi*(T - params.Tm))*mK;
}

__host__ __device__ real_t allen_cahn_reaction_term_2(real_t phi, real_t T, real_t xi, real_t grad_phi_x, real_t grad_phi_y, Allen_Cahn_Params params)
{
    real_t mK = 1;
	real_t grad_val = hypot(grad_phi_x, grad_phi_y);
	return (params.a*allen_cahn_reaction_term_0(phi) - params.b*params.beta*xi*xi*grad_val*(T - params.Tm))*mK;
}

__global__ void allen_cahn_simulate(real_t* phi_map_next, real_t* T_map_next, const real_t* phi_map, const real_t* T_map, Allen_Cahn_Params params, size_t iter)
{
    real_t dx = (real_t) params.sym_size / params.mesh_size_x;
    real_t dy = (real_t) params.sym_size / params.mesh_size_y;
    real_t mK = dx * dy;
    

    //uniform grid
    real_t tau_x = 1;
    real_t tau_y = 1;
    //real_t tau_x = dy / dx;
    //real_t tau_y = dx / dy;

    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.mesh_size_x; x += blockDim.x * gridDim.x) 
    {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.mesh_size_x; y += blockDim.y * gridDim.y) 
        {
	        real_t T = map_at(T_map, x, y, params);
	        real_t phi = map_at(phi_map, x, y, params);

	        real_t phi_py = map_at(phi_map, x, y + 1, params);
	        real_t phi_my = map_at(phi_map, x, y - 1, params);
	        real_t phi_px = map_at(phi_map, x + 1, y, params);
	        real_t phi_mx = map_at(phi_map, x - 1, y, params);

	        real_t T_py = map_at(T_map, x, y + 1, params);
	        real_t T_my = map_at(T_map, x, y - 1, params);
	        real_t T_px = map_at(T_map, x + 1, y, params);
	        real_t T_mx = map_at(T_map, x - 1, y, params);

	        real_t sum_phi_neigbours = 0
		        + tau_y*(phi_py - phi)
		        + tau_y*(phi_my - phi)
		        + tau_x*(phi_px - phi)
		        + tau_x*(phi_mx - phi);
		
	        real_t sum_T_neigbours = 0
		        + tau_y*(T_py - T)
		        + tau_y*(T_my - T)
		        + tau_x*(T_px - T)
		        + tau_x*(T_mx - T);

		    real_t grad_phi_x = (phi_px - phi_mx) * dx / (2 * mK);
		    real_t grad_phi_y = (phi_py - phi_my) * dy / (2 * mK);
        
	        real_t reaction_term = allen_cahn_reaction_term_2(phi, T, params.xi, grad_phi_x, grad_phi_y, params);

	        real_t phi_dt = (sum_phi_neigbours/mK + reaction_term/(params.xi*params.xi)) / params.alpha;
	        real_t T_dt = sum_T_neigbours / mK + params.L * phi_dt;

	        real_t phi_next = phi_dt * params.dt + phi;
	        real_t T_next = T_dt * params.dt + T;
		
            phi_map_next[x + y*params.mesh_size_x] = phi_next;
            T_map_next[x + y*params.mesh_size_x] = T_next;
        }
    }
}


__global__ void allen_cahn_simulate_empty(real_t* phi_map_next, real_t* T_map_next, const real_t* phi_map, const real_t* T_map, Allen_Cahn_Params params, size_t iter)
{
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.mesh_size_x; x += blockDim.x * gridDim.x) 
    {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.mesh_size_x; y += blockDim.y * gridDim.y) 
        {
            phi_map_next[x + y*params.mesh_size_x] = 0;
            T_map_next[x + y*params.mesh_size_x] = 0;
        }
    }
}

#define CUDA_ERR_AND(err) (err) != cudaSuccess ? (err) :

extern "C" cudaError_t kernel_step(real_t* phi_map_next, real_t* T_map_next, const real_t* phi_map, const real_t* T_map, Allen_Cahn_Params params, int device_processor_count, size_t iter)
{
    dim3 bs(64, 1);
    dim3 grid(device_processor_count, 1);
    allen_cahn_simulate<<<grid, bs>>>(phi_map_next, T_map_next, phi_map, T_map, params, iter);

    cudaError_t out = cudaSuccess;
    out = CUDA_ERR_AND(out) cudaGetLastError();
    out = CUDA_ERR_AND(out) cudaDeviceSynchronize();
    return out;
}
