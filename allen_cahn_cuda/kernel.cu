#define JMAPI __host__ __device__ static inline
#include <device_launch_parameters.h>
#include "kernel.h"
#include "cuprintf.cuh"
#include "lib/math.h"

__host__ __device__ f32 map_at(const f32* map, int x, int y, Allen_Cahn_Params params)
{
    int x_mod = x % params.mesh_size_x;
    int y_mod = y % params.mesh_size_y;

    return map[x_mod + y_mod*params.mesh_size_x];
}

__host__ __device__ f32 allen_cahn_reaction_term_0(f32 phi)
{
	return phi*(1 - phi)*(phi - 1.0f/2);
}

__host__ __device__ f32 allen_cahn_reaction_term_1(f32 phi, f32 T, f32 xi, Allen_Cahn_Params params)
{
    f32 mK = 1;
	return (params.a*allen_cahn_reaction_term_0(phi) - params.b*params.beta*xi*(T - params.Tm))*mK;
}

__host__ __device__ f32 allen_cahn_reaction_term_2(f32 phi, f32 T, f32 xi, Vec2 grad_phi, Allen_Cahn_Params params)
{
    f32 mK = 1;
	f32 grad_val = vec2_len(grad_phi);
	return (params.a*allen_cahn_reaction_term_0(phi) - params.b*params.beta*xi*xi*grad_val*(T - params.Tm))*mK;
}

__global__ void allen_cahn_simulate(f32* phi_map_next, f32* T_map_next, const f32* phi_map, const f32* T_map, Allen_Cahn_Params params, isize iter)
{
    f32 dx = (f32) params.sym_size / params.mesh_size_x;
    f32 dy = (f32) params.sym_size / params.mesh_size_y;
    f32 mK = dx * dy;
    
    //uniform grid
    f32 tau_x = 1;
    f32 tau_y = 1;
    //f32 tau_x = dy / dx;
    //f32 tau_y = dx / dy;

    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.mesh_size_x; x += blockDim.x * gridDim.x) 
    {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.mesh_size_x; y += blockDim.y * gridDim.y) 
        {
	        f32 T = map_at(T_map, x, y, params);
	        f32 phi = map_at(phi_map, x, y, params);

	        f32 phi_py = map_at(phi_map, x, y + 1, params);
	        f32 phi_my = map_at(phi_map, x, y - 1, params);
	        f32 phi_px = map_at(phi_map, x + 1, y, params);
	        f32 phi_mx = map_at(phi_map, x - 1, y, params);

	        f32 T_py = map_at(T_map, x, y + 1, params);
	        f32 T_my = map_at(T_map, x, y - 1, params);
	        f32 T_px = map_at(T_map, x + 1, y, params);
	        f32 T_mx = map_at(T_map, x - 1, y, params);

	        f32 sum_phi_neigbours = 0
		        + tau_y*(phi_py - phi)
		        + tau_y*(phi_my - phi)
		        + tau_x*(phi_px - phi)
		        + tau_x*(phi_mx - phi);
		
	        f32 sum_T_neigbours = 0
		        + tau_y*(T_py - T)
		        + tau_y*(T_my - T)
		        + tau_x*(T_px - T)
		        + tau_x*(T_mx - T);

	        Vec2 grad_phi = {
		        (phi_px - phi_mx) * dx / (2 * mK),
		        (phi_py - phi_my) * dy / (2 * mK)
	        };
        
	        f32 reaction_term = allen_cahn_reaction_term_2(phi, T, params.xi, grad_phi, params);
	        f32 phi_dt = (sum_phi_neigbours/mK + reaction_term/(params.xi*params.xi)) / params.alpha;
	        f32 T_dt = sum_T_neigbours / mK + params.L * phi_dt;

	        f32 phi_next = phi_dt * params.dt + phi;
	        f32 T_next = T_dt * params.dt + T;
		
            phi_map_next[x + y*params.mesh_size_x] = phi_next;
            T_map_next[x + y*params.mesh_size_x] = T_next;

            
            phi_map_next[x + y*params.mesh_size_x] = 0;
            T_map_next[x + y*params.mesh_size_x] = 0;
        }
    }
}


__global__ void allen_cahn_simulate_empty(f32* phi_map_next, f32* T_map_next, const f32* phi_map, const f32* T_map, Allen_Cahn_Params params, isize iter)
{
    f32 dx = (f32) params.sym_size / params.mesh_size_x;
    f32 dy = (f32) params.sym_size / params.mesh_size_y;
    f32 mK = dx * dy;
    
    //uniform grid
    f32 tau_x = 1;
    f32 tau_y = 1;
    //f32 tau_x = dy / dx;
    //f32 tau_y = dx / dy;

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

extern "C" cudaError_t kernel_step(f32* phi_map_next, f32* T_map_next, const f32* phi_map, const f32* T_map, Allen_Cahn_Params params, int device_processor_count, isize iter)
{
    dim3 bs(64, 1);
    dim3 grid(device_processor_count, 1);
    allen_cahn_simulate<<<grid, bs>>>(phi_map_next, T_map_next, phi_map, T_map, params, iter);

    cudaError_t out = cudaSuccess;
    out = CUDA_ERR_AND(out) cudaGetLastError();
    out = CUDA_ERR_AND(out) cudaDeviceSynchronize();
    return out;
}

#include "cuprintf.cu"