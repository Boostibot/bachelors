#define JMAPI __host__ __device__ static inline

#include <device_launch_parameters.h>
#include "kernel.h"
#include <cmath>
#include <stdio.h>
#include <assert.h>

#define MOD(x, mod) (((x) % (mod) + (mod)) % (mod))

__host__ __device__ real_t map_at(const real_t* map, int x, int y, Allen_Cahn_Params params)
{
    int x_mod = MOD(x, params.mesh_size_x);
    int y_mod = MOD(y, params.mesh_size_y);

    if(x_mod < 0 || x_mod >= params.mesh_size_x || y_mod < 0 || y_mod >= params.mesh_size_y)
    {
        printf("BAD %d %d\n", (x_mod), x);
    }

    return map[x_mod + y_mod*params.mesh_size_x];
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

__global__ void allen_cahn_simulate(real_t* Phi_map_next, real_t* T_map_next, const real_t* Phi_map, const real_t* T_map, Allen_Cahn_Params params, size_t iter)
{
    real_t dx = (real_t) params.sym_size / params.mesh_size_x;
    real_t dy = (real_t) params.sym_size / params.mesh_size_y;
    real_t mK = dx * dy;

    real_t a = params.a;
    real_t b = params.b;
    real_t alpha = params.alpha;
    real_t beta = params.beta;
    real_t xi = params.xi;
    real_t Tm = params.Tm;
    real_t L = params.L; //Latent heat, not sym_size ! 
    real_t dt = params.dt;

    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.mesh_size_x; x += blockDim.x * gridDim.x) 
    {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.mesh_size_x; y += blockDim.y * gridDim.y) 
        {
            real_t T = map_at(T_map, x, y, params);
            real_t Phi = map_at(Phi_map, x, y, params);

            real_t Phi_U = map_at(Phi_map, x, y + 1, params);
            real_t Phi_D = map_at(Phi_map, x, y - 1, params);
            real_t Phi_R = map_at(Phi_map, x + 1, y, params);
            real_t Phi_L = map_at(Phi_map, x - 1, y, params);

            real_t T_U = map_at(T_map, x, y + 1, params);
            real_t T_D = map_at(T_map, x, y - 1, params);
            real_t T_R = map_at(T_map, x + 1, y, params);
            real_t T_L = map_at(T_map, x - 1, y, params);

            real_t grad_Phi_x = dy*(Phi_R - Phi_L);
            real_t grad_Phi_y = dx*(Phi_U - Phi_D);

            real_t int_K_f = a*mK*f0(Phi) - b*xi*xi*beta*(T - Tm)*hypotf(grad_Phi_x, grad_Phi_y)/2;
            real_t int_K_laplace_T = dy/dx*(T_L - 2*T + T_R) + dx/dy*(T_D - 2*T + T_U);

            real_t int_K_dt_Phi = 1/alpha*int_K_laplace_T + 1/(xi*xi * alpha)*int_K_f;
            real_t int_K_dt_T = int_K_laplace_T + L*int_K_dt_Phi;

            #define ECHOF(x) printf(#x": %f\n", (x))

            if(0)
            if((x == 1 && y == 3))
            {
                ECHOF((real_t) x);
                printf("T: U:%f D:%f R:%f L:%f\n", T_U, T_D, T_R, T_L);
                printf("Phi: U:%f D:%f R:%f L:%f\n", Phi_U, Phi_D, Phi_R, Phi_L);
                ECHOF(int_K_f);
                ECHOF(int_K_laplace_T);
                ECHOF(int_K_dt_Phi);
                ECHOF(int_K_dt_T);
            }

            real_t dt_Phi = 1/mK*int_K_dt_Phi;
            real_t dt_T = 1/mK*int_K_dt_T;

            real_t Phi_next = Phi + dt*dt_Phi;
            real_t T_next = T + dt*dt_T;
        
            Phi_map_next[x + y*params.mesh_size_x] = Phi_next;
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
