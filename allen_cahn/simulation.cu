#include "simulation.h"
#ifdef COMPILE_TESTS
#define TEST_CUDA_ALL
#endif

#include "exact.h"
#include "cuda_util.cuh"
#include "cuda_reduction.cuh"
#include "cuda_for.cuh"
#include "cuda_random.cuh"

#ifdef COMPILE_SIMULATION

#if 1
SHARED Real custom_hypot(Real y, Real x) { return (Real) hypotf((float)y, (float)x); }
SHARED Real custom_atan2(Real y, Real x) { return (Real) atan2f((float) y, (float) x); }
SHARED Real custom_cos(Real theta) { return (Real) cosf((float)theta); }
#else
SHARED Real custom_hypot(Real y, Real x) { return hypot(y, x); }
SHARED Real custom_atan2(Real y, Real x) { return atan2(y, x); }
SHARED Real custom_cos(Real theta) { return cos(theta); }
#endif

#include <assert.h>
#ifndef assert
#define assert(x)
#endif

SHARED Real boundary_sample(const Real* map, int x, int y, int nx, int ny, Sim_Boundary_Type bound)
{
    switch(bound)
    {
        case BOUNDARY_PERIODIC: {
            int x_mod = x;
            if(x_mod < 0)
                x_mod += nx;
            else if(x_mod >= nx)
                x_mod -= nx;

            int y_mod = y;
            if(y_mod < 0)
                y_mod += ny;
            else if(y_mod >= ny)
                y_mod -= ny;

            return map[x_mod + y_mod*nx];
        } break;

        case BOUNDARY_DIRICHLET_ZERO: {
            if(0 <= x && x < nx && 0 <= y && y < ny)
                return map[x + y*nx];
            else
            {
                int clampx = CLAMP(x, 0, nx - 1);
                int clampy = CLAMP(y, 0, ny - 1);
                assert(0 <= clampx && clampx < nx && 0 <= clampy && clampy < ny );
                return -map[clampx + clampy*nx];
            }              
        } break;

        case BOUNDARY_NEUMANN_ZERO: {
            int clampx = CLAMP(x, 0, nx - 1);
            int clampy = CLAMP(y, 0, ny - 1);
            assert(0 <= clampx && clampx < nx && 0 <= clampy && clampy < ny );
            return map[clampx + clampy*nx];
        } break;

        case BOUNDARY_ENUM_COUNT:
        default:
            return 0;
    }
}


void sim_realloc(Sim_Map* map, const char* name, int nx, int ny, double time, i64 iter)
{
    if(map->nx != nx || map->ny != ny)
    {
        cuda_realloc_in_place(&map->data, (size_t)(nx*ny)*sizeof(double), (size_t)(map->nx*map->ny)*sizeof(double), REALLOC_ZERO);
        map->nx = nx;
        map->ny = ny;
    }

    map->iter = iter;
    map->time = time;
    strncpy(map->name, name, sizeof map->name - 1);
    map->name[sizeof map->name - 1] = 0;
}

static Sim_Map* _find_claim_temp_map(Sim_Params params, const char* name, int nx, int ny, double time, i64 iter)
{
    Sim_Map* not_used = NULL;
    Sim_Map* not_used_with_size = NULL;
    for(int i = 0; i < params.temp_map_count; i++)
    {
        Sim_Map* curr = &params.temp_maps[i];
        if(curr->iter != iter)
        {
            if(not_used == NULL)
                not_used = curr;
            if(curr->nx == nx && params.temp_maps[i].nx == nx)
            {
                not_used_with_size = curr;
                break;
            }
        }
    }

    Sim_Map* out = not_used_with_size ? not_used_with_size : not_used;
    if(out != NULL)
        sim_realloc(out, name, nx, ny, time, iter);

    return out;
}

SHARED Real f0(Real phi)
{
	return phi*(1 - phi)*(phi - (Real) 0.5);
}

struct Phase_Temp {
    Real Phi;
    Real T;
};

struct Explicit_Blend_State {
    Real weight;
    const Real* F;
    const Real* U;
};

template<bool IS_EULER, typename ... States>
void explicit_solver_solve_lin_combination(Real* out_F, Real* out_U, Sim_Params params, States... state_args)
{
    int nx = params.nx;
    int ny = params.ny;

    constexpr int state_count = (int) sizeof...(state_args);
    Explicit_Blend_State states[(size_t) state_count] = {state_args...};

    Real a = (Real) params.a;
    Real b = (Real) params.b;
    Real alpha = (Real) params.alpha;
    Real beta = (Real) params.beta;
    Real xi = (Real) params.xi;
    Real Tm = (Real) params.Tm;
    Real L = (Real) params.L; //Latent heat, not L0 (sym size) ! 
    Real dt = (Real) params.dt;
    Real S0 = (Real) params.S; //anisotrophy strength
    Real m0 = (Real) params.m0; //anisotrophy frequency (?)
    Real theta0 = (Real) params.theta0;
    bool do_corrector_guess = params.do_corrector_guess;

    Real dx = (Real) params.L0 / nx;
    Real dy = (Real) params.L0 / ny;
    Real one_over_2dx = 1/(2*dx);
    Real one_over_2dy = 1/(2*dy);
    Real one_over_dx2 = 1/(dx*dx);
    Real one_over_dy2 = 1/(dy*dy);

    Real k0_factor = a/(xi*xi * alpha);
    Real k2_factor = b*beta/alpha;
    Real k1_factor = 1/alpha;
    Real dt_L = dt*L;

    Real fu = 0;
    if(params.do_exact)
    {
        Exact_Params exact_params = get_static_exact_params(params);
        fu = exact_fu(params.iter*dt, exact_params);
    }

    Sim_Boundary_Type U_bound = params.T_boundary;
    Sim_Boundary_Type F_bound = params.Phi_boundary;
    cuda_tiled_for_2D<1, 1, Phase_Temp>(0, 0, nx, ny,
        [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry) -> Phase_Temp{
            Real U = 0;
            Real F = 0;
            #pragma unroll
            for(int i = 0; i < state_count; i++)
            {
                F += boundary_sample(states[i].F, x, y, nx, ny, F_bound) * states[i].weight;
                U += boundary_sample(states[i].U, x, y, nx, ny, U_bound) * states[i].weight;
            }

            return Phase_Temp{F, U};
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Phase_Temp* shared){
            Phase_Temp C = shared[tx   + ty*tile_size_x];
            Phase_Temp E = shared[tx+1 + ty*tile_size_x];
            Phase_Temp W = shared[tx-1 + ty*tile_size_x];
            Phase_Temp N = shared[tx   + (ty+1)*tile_size_x];
            Phase_Temp S = shared[tx   + (ty-1)*tile_size_x];

            Real grad_Phi_x = (E.Phi - W.Phi)*one_over_2dx;
            Real grad_Phi_y = (N.Phi - S.Phi)*one_over_2dx;
            Real grad_Phi_norm = custom_hypot(grad_Phi_x, grad_Phi_y);

            Real theta = custom_atan2(grad_Phi_y, grad_Phi_x);
            Real g_theta = (Real) 1 - S0*custom_cos(m0*theta + theta0);

            Real laplace_Phi = (W.Phi - 2*C.Phi + E.Phi)*one_over_dx2 + (S.Phi - 2*C.Phi + N.Phi)*one_over_dy2;
            Real laplace_T = (W.T - 2*C.T + E.T)*one_over_dx2 + (S.T - 2*C.T + N.T)*one_over_dy2;

            Real k0 = g_theta*f0(C.Phi)*k0_factor;
            Real k2 = grad_Phi_norm*k2_factor;
            Real k1 = g_theta*k1_factor;
            Real corr = 1 + k2*dt_L;

            Real dt_Phi = 0;
            if(do_corrector_guess)
                dt_Phi = (k1*laplace_Phi + k0 - k2*(C.T - Tm + dt*laplace_T))/corr;
            else
                dt_Phi = k1*laplace_Phi + k0 - k2*(C.T - Tm);

            Real dt_T = laplace_T + L*dt_Phi + fu;

            if(IS_EULER)
            {
                out_F[x + y*nx] = C.Phi + dt*dt_Phi;
                out_U[x + y*nx] = C.T + dt*dt_T;
            }
            else
            {
                out_F[x + y*nx] = dt_Phi;
                out_U[x + y*nx] = dt_T;
            }
                
        });   
}

void calc_debug_values(const Real* F, const Real* U, Real* grad_F, Real* grad_U, Real* aniso, Sim_Params params)
{
    int nx = params.nx;
    int ny = params.ny;
    Real S0 = (Real) params.S;
    Real m0 = (Real) params.m0;
    Real theta0 = (Real) params.theta0;
    Sim_Boundary_Type Phi_bound = params.Phi_boundary;
    Sim_Boundary_Type T_bound = params.T_boundary;
    cuda_for_2D(0, 0, nx, ny, [=]SHARED(int x, int y){
        Real Phi_U = boundary_sample(F, x, y + 1, nx, ny, Phi_bound);
        Real Phi_D = boundary_sample(F, x, y - 1, nx, ny, Phi_bound);
        Real Phi_R = boundary_sample(F, x + 1, y, nx, ny, Phi_bound);
        Real Phi_L = boundary_sample(F, x - 1, y, nx, ny, Phi_bound);

        Real T_U = boundary_sample(U, x, y + 1, nx, ny, T_bound);
        Real T_D = boundary_sample(U, x, y - 1, nx, ny, T_bound);
        Real T_R = boundary_sample(U, x + 1, y, nx, ny, T_bound);
        Real T_L = boundary_sample(U, x - 1, y, nx, ny, T_bound);

        Real grad_Phi_x = (Phi_R - Phi_L);
        Real grad_Phi_y = (Phi_U - Phi_D);
        Real grad_Phi_norm = custom_hypot(grad_Phi_x, grad_Phi_y);

        Real grad_T_x = (T_R - T_L);
        Real grad_T_y = (T_U - T_D);
        Real grad_T_norm = custom_hypot(grad_T_x, grad_T_y);
        
        Real theta = custom_atan2(grad_Phi_y, grad_Phi_x);
        Real g_theta = (Real) 1 - S0*custom_cos(m0*theta + theta0);

        grad_F[x + y*nx] = grad_Phi_norm;
        grad_U[x + y*nx] = grad_T_norm; 
        aniso[x + y*nx] = g_theta;
        // aniso[x + y*nx] = Phi_U;
    });
}

void explicit_solver_euler_step_based(const Real* F, const Real* U, const Real* U_base, Real* next_F, Real* next_U, Sim_Params params, bool do_stats)
{
    (void) do_stats;
    if(U == U_base)
        explicit_solver_solve_lin_combination<true>(next_F, next_U, params, Explicit_Blend_State{1, F, U});
    else
    {
        Cache_Tag tag = cache_tag_make();
        int N = params.ny * params.nx;

        Real* dt_F = cache_alloc(Real, N, &tag);
        Real* dt_U = cache_alloc(Real, N, &tag);

        Real dt = (Real) params.dt;
        explicit_solver_solve_lin_combination<false>(dt_F, dt_U, params, Explicit_Blend_State{1, F, U});
        cuda_for(0, N, [=]SHARED(int i){
            next_F[i] = F[i] + dt*dt_F[i];
            next_U[i] = U_base[i] + dt*dt_U[i];
        });

        cache_free(&tag);
    }

    if(params.stats)
    {
        params.stats->Phi_iters = 1;
        params.stats->T_iters = 1;
    }
}

void explicit_solver_rk4_step(const Real* F, const Real* U, Real* next_F, Real* next_U, Sim_Params params)
{
    Cache_Tag tag = cache_tag_make();
    int N = params.ny * params.nx;

    Real* k1_F = cache_alloc(Real, N, &tag);
    Real* k2_F = cache_alloc(Real, N, &tag);
    Real* k3_F = cache_alloc(Real, N, &tag);
    Real* k4_F = cache_alloc(Real, N, &tag);

    Real* k1_U = cache_alloc(Real, N, &tag);
    Real* k2_U = cache_alloc(Real, N, &tag);
    Real* k3_U = cache_alloc(Real, N, &tag);
    Real* k4_U = cache_alloc(Real, N, &tag);

    Real dt = (Real) params.dt;
    using W = Explicit_Blend_State;
    explicit_solver_solve_lin_combination<false>(k1_F, k1_U, params, W{1, F, U});
    explicit_solver_solve_lin_combination<false>(k2_F, k2_U, params, W{1, F, U}, W{dt/2, k1_F, k1_U});
    explicit_solver_solve_lin_combination<false>(k3_F, k3_U, params, W{1, F, U}, W{dt/2, k2_F, k2_U});
    explicit_solver_solve_lin_combination<false>(k4_F, k4_U, params, W{1, F, U}, W{dt, k3_F, k3_U});

    cuda_for(0, params.ny*params.nx, [=]SHARED(int i){
        next_F[i] = F[i] + dt/6*(k1_F[i] + 2*k2_F[i] + 2*k3_F[i] + k4_F[i]);
        next_U[i] = U[i] + dt/6*(k1_U[i] + 2*k2_U[i] + 2*k3_U[i] + k4_U[i]);
    });

    if(params.stats)
    {
        params.stats->Phi_iters = 1;
        params.stats->T_iters = 1;
    }

    cache_free(&tag);
    CUDA_DEBUG_TEST(cudaDeviceSynchronize());
}

double explicit_solver_rk4_adaptive_step(const Real* F, const Real* U, Real* next_F, Real* next_U, Sim_Params params)
{
    Cache_Tag tag = cache_tag_make();
    int N = params.ny * params.nx;

    static Real _initial_step = 0;
    if(_initial_step <= 0)
        _initial_step = (Real) params.dt;

    Real tau = _initial_step;
    Real epsilon_F = 0;
    Real epsilon_U = 0;
    Real* Epsilon_F = cache_alloc(Real, N, &tag);
    Real* Epsilon_U = cache_alloc(Real, N, &tag);

    Real* k1_F = cache_alloc(Real, N, &tag);
    Real* k2_F = cache_alloc(Real, N, &tag);
    Real* k3_F = cache_alloc(Real, N, &tag);
    Real* k4_F = cache_alloc(Real, N, &tag);
    Real* k5_F = cache_alloc(Real, N, &tag);

    Real* k1_U = cache_alloc(Real, N, &tag);
    Real* k2_U = cache_alloc(Real, N, &tag);
    Real* k3_U = cache_alloc(Real, N, &tag);
    Real* k4_U = cache_alloc(Real, N, &tag);
    Real* k5_U = cache_alloc(Real, N, &tag);

    using W = Explicit_Blend_State;
    explicit_solver_solve_lin_combination<false>(k1_F, k1_U, params, W{1, F, U});

    bool converged = false;
    int i = 0;
    int max_iters = MAX(MAX(params.T_max_iters, params.Phi_max_iters), 1);
    Real used_tau = tau;
    for(; i < max_iters && converged == false; i++)
    {
        // k1 = f(t, x);
        // k2 = f(t + tau/3, x + tau/3*k1);
        // k3 = f(t + tau/3, x + tau/6*(k1 + k2));
        // k4 = f(t + tau/2, x + tau/8*(k1 + 3*k3));
        // k5 = f(t + tau/1, x + tau*(0.5f*k1 - 1.5f*k3 + 2*k4));
        
        // k1 = f(x);
        // k2 = f(x + tau/3*k1);
        // k3 = f(x + tau/6*k1 + tau/6*k2);
        // k4 = f(x + tau/8*k1 + tau*3/8*k3));
        // k5 = f(x + tau/2*k1 - tau*3/2*k3 + tau*2*k4));
        
        explicit_solver_solve_lin_combination<false>(k2_F, k2_U, params, W{1, F, U}, W{tau/3, k1_F, k1_U});
        explicit_solver_solve_lin_combination<false>(k3_F, k3_U, params, W{1, F, U}, W{tau/6, k1_F, k1_U}, W{tau/6, k2_F, k2_U});
        explicit_solver_solve_lin_combination<false>(k4_F, k4_U, params, W{1, F, U}, W{tau/8, k1_F, k1_U}, W{tau*3/8, k3_F, k3_U});
        explicit_solver_solve_lin_combination<false>(k5_F, k5_U, params, W{1, F, U}, W{tau/2, k1_F, k1_U}, W{-tau*3/2, k3_F, k3_U}, W{tau*2, k4_F, k4_U});

        cuda_for(0, params.ny*params.nx, [=]SHARED(int i){
            Real F = (Real)0.2*k1_F[i] - (Real)0.9*k3_F[i] + (Real)0.8*k4_F[i] - (Real)0.1*k5_F[i];
            Real U = (Real)0.2*k1_U[i] - (Real)0.9*k3_U[i] + (Real)0.8*k4_U[i] - (Real)0.1*k5_U[i];

            Epsilon_F[i] = F >= 0 ? F : -F;
            Epsilon_U[i] = U >= 0 ? U : -U;
        });

        // L2 instead of max norm!
        epsilon_F = cuda_L2_norm(Epsilon_F, N)/sqrt((Real)N);
        epsilon_U = cuda_L2_norm(Epsilon_U, N)/sqrt((Real)N);

        // epsilon_F = cuda_max(Epsilon_F, N);
        // epsilon_U = cuda_max(Epsilon_U, N);

        if(epsilon_F < params.Phi_tolerance && epsilon_U < params.T_tolerance)
            converged = true;

        Real epsilon = (Real) MAX(epsilon_F + epsilon_U, 1e-20);
        Real delta = (Real) MAX(params.Phi_tolerance + params.T_tolerance, 1e-20);
        used_tau = tau;
        tau = pow(delta / epsilon, (Real)0.2)*4/5*tau;
        tau = MAX(tau, params.min_dt);

        //If below the min limit there is no point in trying another loop
        if(tau <= params.min_dt && used_tau <= params.min_dt)
            break;
    }

    cuda_for(0, params.ny*params.nx, [=]SHARED(int i){
        next_F[i] = F[i] + used_tau*((Real)1/6*(k1_F[i] + k5_F[i]) + (Real)2/3*k4_F[i]);
        next_U[i] = U[i] + used_tau*((Real)1/6*(k1_U[i] + k5_U[i]) + (Real)2/3*k4_U[i]);
    });

    LOG("SOLVER", converged ? LOG_DEBUG : LOG_WARN, "rk4-adaptive %s in %i iters with error F:%lf | U:%lf | tau:%e", converged ? "converged" : "diverged", i, (double) epsilon_F, (double) epsilon_U, (double)used_tau);
    _initial_step = tau;

    if(params.stats)
    {
        params.stats->Phi_iters = i;
        params.stats->T_iters = i;
    }

    cache_free(&tag);
    CUDA_DEBUG_TEST(cudaDeviceSynchronize());
    return (double) used_tau;
}

struct Cross_Matrix_Static {
    Real C;
    Real U;
    Real D;
    Real L;
    Real R;

    int nx;
    int ny;

    Sim_Boundary_Type boundary;
};

struct Anisotrophy_Matrix {
    Real* anisotrophy;
    Real X;
    Real Y;
    Real C_minus_one;

    int nx;
    int ny;

    Sim_Boundary_Type boundary;
};

void cross_matrix_static_multiply(Real* out, const void* _A, const Real* vec, int N)
{
    Cross_Matrix_Static A = *(Cross_Matrix_Static*)_A;
    int nx = A.nx;
    int ny = A.ny;

    cuda_tiled_for_2D<1, 1, Real>(0, 0, nx, ny,
        [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry) -> Real {
            return boundary_sample(vec, x, y, nx, ny, A.boundary);
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Real* shared){
            Real val = shared[tx + ty*tile_size_x]*A.C;
            val += shared[tx+1 + ty*tile_size_x]*A.R;
            val += shared[tx-1 + ty*tile_size_x]*A.L;
            val += shared[tx   + (ty+1)*tile_size_x]*A.U;
            val += shared[tx   + (ty-1)*tile_size_x]*A.D;

            out[x + y*nx] = val;
        }
    );
}

void anisotrophy_matrix_multiply(Real* out, const void* _A, const Real* vec, int N)
{
    Anisotrophy_Matrix A = * (Anisotrophy_Matrix*)_A;
    int nx = A.nx;
    int ny = A.ny;

    cuda_tiled_for_2D<1, 1, Real>(0, 0, nx, ny,
        [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry) -> Real {
            return boundary_sample(vec, x, y, nx, ny, A.boundary);
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Real* shared){
            int i = x + y*nx;
            Real s = A.anisotrophy[i];
            Real X = A.X*s;
            Real Y = A.Y*s;
            Real C = 1 + A.C_minus_one*s;

            Real val = shared[tx + ty*tile_size_x]*C;
            val += shared[tx+1 + ty*tile_size_x]*X;
            val += shared[tx-1 + ty*tile_size_x]*X;
            val += shared[tx   + (ty+1)*tile_size_x]*Y;
            val += shared[tx   + (ty-1)*tile_size_x]*Y;

            out[i] = val;
        }
    );
}

typedef struct Conjugate_Gardient_Params {
    Real epsilon;
    Real tolerance;
    int max_iters;

    const Real* initial_value_or_null;
} Conjugate_Gardient_Params;

typedef struct Conjugate_Gardient_Convergence {
    Real error;
    int iters;
    bool converged;
} Conjugate_Gardient_Convergence;

typedef void(*Matrix_Vector_Mul_Func)(Real* out, const void* A, const Real* x, int N);

Conjugate_Gardient_Convergence conjugate_gradient_solve(const void* A, Real* x, const Real* b, int N, Matrix_Vector_Mul_Func matrix_mul_func, const Conjugate_Gardient_Params* params_or_null)
{
    i64 start = clock_ns();
    Conjugate_Gardient_Params params = {0};
    params.epsilon = (Real) 1.0e-10;
    params.tolerance = (Real) 1.0e-5;
    params.max_iters = 10;
    if(params_or_null)
        params = *params_or_null;

    Cache_Tag tag = cache_tag_make();

    Real scaled_squared_tolerance = params.tolerance*params.tolerance*N;
    Real* r = cache_alloc(Real, N, &tag);
    Real* p = cache_alloc(Real, N, &tag);
    Real* Ap = cache_alloc(Real, N, &tag);
    Real r_dot_r = 0;

    //@TODO: IMPLEMENT FULLY (add launch params for reductions etc.)!
    static cudaStream_t stream1 = NULL;
    static cudaStream_t stream2 = NULL;
    static cudaStream_t stream3 = NULL;
    static cudaStream_t stream4 = NULL;
    if(stream1 == NULL)
    {
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);
        cudaStreamCreate(&stream4);
    }

    //@TODO: streams
    if(params.initial_value_or_null)
    {
        CUDA_DEBUG_TEST(cudaMemcpyAsync(x, params.initial_value_or_null, sizeof(Real)*(size_t)N, cudaMemcpyDeviceToDevice));
        matrix_mul_func(Ap, A, params.initial_value_or_null, N);
        cuda_for(0, N, [=]SHARED(int i){
            r[i] = b[i] - Ap[i];
            p[i] = r[i];
        });

        r_dot_r = cuda_dot_product(r, r, N);
    }
    else
    {
        CUDA_DEBUG_TEST(cudaMemsetAsync(x, 0, sizeof(Real)*(size_t)N, stream1));
        CUDA_DEBUG_TEST(cudaMemcpyAsync(r, b, sizeof(Real)*(size_t)N, cudaMemcpyDeviceToDevice, stream2));
        CUDA_DEBUG_TEST(cudaMemcpyAsync(p, b, sizeof(Real)*(size_t)N, cudaMemcpyDeviceToDevice, stream3));

        r_dot_r = cuda_dot_product(b, b, N);
        // CUDA_DEBUG_TEST(cudaDeviceSynchronize());
    }

    int iter = 0;
    if(r_dot_r >= scaled_squared_tolerance || iter == 0)
    {
        for(; iter < params.max_iters; iter++)
        {
            matrix_mul_func(Ap, A, p, N);
            
            Real p_dot_Ap = cuda_dot_product(p, Ap, N);
            Real alpha = r_dot_r / MAX(p_dot_Ap, params.epsilon);
            
            cuda_for(0, N, [=]SHARED(int i){
                x[i] = x[i] + alpha*p[i];
                r[i] = r[i] - alpha*Ap[i];
            });

            Real r_dot_r_new = cuda_dot_product(r, r, N);
            if(r_dot_r_new < scaled_squared_tolerance)
            {
                r_dot_r = r_dot_r_new;
                break;
            }

            Real beta = r_dot_r_new / MAX(r_dot_r, params.epsilon);
            cuda_for(0, N, [=]SHARED(int i){
                p[i] = r[i] + beta*p[i]; 
            });

            r_dot_r = r_dot_r_new;
        }
    }

    Conjugate_Gardient_Convergence out = {0};
    out.iters = iter;
    out.converged = iter != params.max_iters;
    out.error = sqrt(r_dot_r/N);

    i64 end = clock_ns();
    LOG_DEBUG("KERNEL", "conjugate_gradient_solve(%lli) took: %.2ems", (lli)N, (double)(end - start)*1e-6);

    cache_free(&tag);
    return out;
} 

void matrix_multiply(Real* output, const Real* A, const Real* B, int A_height, int A_width, int B_height, int B_width)
{
    assert(A_width == B_height);
    for(int y = 0; y < A_height; y++)
    {
        for(int x = 0; x < B_width; x++)
        {
            Real val = 0;
            for(int k = 0; k < A_width; k++)
                val += A[k + y*A_width]*B[x + k*B_width];

            output[x + y*B_width] = val;
        }
    }
}

thread_local static cudaEvent_t _cuda_timer_start = NULL;
thread_local static cudaEvent_t _cuda_timer_stop = NULL;

void cuda_timer_start()
{
    if(_cuda_timer_start == NULL || _cuda_timer_stop == NULL)
    {
        CUDA_TEST(cudaEventCreate(&_cuda_timer_start));
        CUDA_TEST(cudaEventCreate(&_cuda_timer_stop));
    }
    CUDA_TEST(cudaEventRecord(_cuda_timer_start, 0));
}

double cuda_timer_stop()
{
    CUDA_TEST(cudaEventRecord(_cuda_timer_stop, 0));
    CUDA_TEST(cudaEventSynchronize(_cuda_timer_stop));

    float time = 0;
    CUDA_TEST(cudaEventElapsedTime(&time, _cuda_timer_start, _cuda_timer_stop));
    return (double) time / 1000;
}


void semi_implicit_solver_step_based(const Real* F, const Real* U, const Real* U_base, Real* next_F, Real* next_U, Sim_Params params, bool do_debug)
{
    Cache_Tag tag = cache_tag_make();

    i64 iter = params.iter;
    int nx = params.nx;
    int ny = params.ny;
    Real dx = (Real) params.L0 / params.nx;
    Real dy = (Real) params.L0 / params.ny;
    int N = nx*ny;

    Real a = (Real) params.a;
    Real b = (Real) params.b;
    Real alpha = (Real) params.alpha;
    Real beta = (Real) params.beta;
    Real xi = (Real) params.xi;
    Real Tm = (Real) params.Tm;
    Real L = (Real) params.L; 
    Real dt = (Real) params.dt;
    Real S0 = (Real) params.S; 
    Real m0 = (Real) params.m0; 
    Real theta0 = (Real) params.theta0;
    Real gamma = (Real) params.gamma;
    
    Real* b_F = cache_alloc(Real, N, &tag);
    Real* b_U = cache_alloc(Real, N, &tag);
    Real* anisotropy = cache_alloc(Real, N, &tag);

    Real one_over_2dx = 1/(2*dx);
    Real one_over_2dy = 1/(2*dy);
    Real one_over_dx2 = 1/(dx*dx);
    Real one_over_dy2 = 1/(dy*dy);
    Real k0_factor = a/(xi*xi * alpha);
    Real k2_factor = b*beta/alpha;
    Real k1_factor = 1/alpha;
    Real dt_L = dt*L;

    Sim_Boundary_Type U_bound = params.T_boundary;
    Sim_Boundary_Type F_bound = params.Phi_boundary;

    Anisotrophy_Matrix A_F = {0};
    A_F.anisotrophy = anisotropy;
    A_F.C_minus_one = 2*dt/(dx*dx) + 2*dt/(dy*dy);
    A_F.X = -dt/(dx*dx);
    A_F.Y = -dt/(dy*dy);
    A_F.nx = nx;
    A_F.ny = ny;
    A_F.boundary = params.Phi_boundary;

    Cross_Matrix_Static A_U = {0};
    A_U.C = 1 + 2*dt/(dx*dx) + 2*dt/(dy*dy);
    A_U.R = -dt/(dx*dx);
    A_U.L = -dt/(dx*dx);
    A_U.U = -dt/(dy*dy);
    A_U.D = -dt/(dy*dy);
    A_U.nx = nx;
    A_U.ny = ny;
    A_U.boundary = params.T_boundary;

    bool do_corrector_guess = params.do_corrector_guess;
    bool is_tiled = true;

    cuda_timer_start();
    if(do_corrector_guess)
    {
        cuda_tiled_for_2D<1, 1, Phase_Temp>(0, 0, nx, ny,
            [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry) -> Phase_Temp {
                Real Phi = boundary_sample(F, x, y, nx, ny, F_bound);
                Real T = boundary_sample(U, x, y, nx, ny, U_bound);
                return Phase_Temp{Phi, T};
            },
            [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Phase_Temp* shared){
                Phase_Temp C = shared[tx   + ty*tile_size_x];
                Phase_Temp E = shared[tx+1 + ty*tile_size_x];
                Phase_Temp W = shared[tx-1 + ty*tile_size_x];
                Phase_Temp N = shared[tx   + (ty+1)*tile_size_x];
                Phase_Temp S = shared[tx   + (ty-1)*tile_size_x];

                Real grad_Phi_x = (E.Phi - W.Phi)*one_over_2dx;
                Real grad_Phi_y = (N.Phi - S.Phi)*one_over_2dy;
                Real grad_Phi_norm = custom_hypot(grad_Phi_x, grad_Phi_y);

                Real theta = custom_atan2(grad_Phi_y, grad_Phi_x);
                Real g_theta = (Real) 1 - S0*custom_cos(m0*theta + theta0);

                Real laplace_Phi = (W.Phi - 2*C.Phi + E.Phi)*one_over_dx2 + (S.Phi - 2*C.Phi + N.Phi)*one_over_dy2;
                Real laplace_T =   (W.T - 2*C.T + E.T)*one_over_dx2 +       (S.T - 2*C.T + N.T)*one_over_dy2;

                Real k0 = g_theta*f0(C.Phi)*k0_factor;
                Real k2 = grad_Phi_norm*k2_factor;
                Real k1 = g_theta*k1_factor;
                Real corr = 1 + k2*dt_L;

                Real right = C.Phi + dt/corr*((1-gamma)*k1*laplace_Phi + k0 - k2*(C.T - Tm + dt*laplace_T));
                Real factor = gamma/corr*k1; 

                A_F.anisotrophy[x+y*nx] = (Real) factor;
                b_F[x + y*nx] = (Real) right;
            }
        );
    }
    else
    {
        cuda_tiled_for_2D<1, 1, Real>(0, 0, nx, ny,
            [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry) -> Real {
                return boundary_sample(F, x, y, nx, ny, F_bound);
            },
            [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Real* shared){
                Real C_T   = U[x + y*nx];
                Real C_Phi = shared[tx   + ty*tile_size_x];
                Real E_Phi = shared[tx+1 + ty*tile_size_x];
                Real W_Phi = shared[tx-1 + ty*tile_size_x];
                Real N_Phi = shared[tx   + (ty+1)*tile_size_x];
                Real S_Phi = shared[tx   + (ty-1)*tile_size_x];

                Real grad_Phi_x = (E_Phi - W_Phi)/(2*dx);
                Real grad_Phi_y = (N_Phi - S_Phi)/(2*dy);
                Real grad_Phi_norm = custom_hypot(grad_Phi_x, grad_Phi_y);


                Real theta = custom_atan2(grad_Phi_y, grad_Phi_x);
                Real g_theta = (Real) 1 - S0*custom_cos(m0*theta + theta0);

                Real laplace_Phi = (W_Phi - 2*C_Phi + E_Phi)*one_over_dx2 + (S_Phi - 2*C_Phi + N_Phi)*one_over_dy2;

                Real k0 = g_theta*f0(C_Phi)*k0_factor;
                Real k2 = grad_Phi_norm*k2_factor;
                Real k1 = g_theta*k1_factor;

                Real right = C_Phi + dt*((1-gamma)*k1*laplace_Phi + k0 - k2*(C_T - Tm));
                Real factor = gamma*k1; 

                A_F.anisotrophy[x+y*nx] = (Real) factor;
                b_F[x + y*nx] = (Real) right;
            }
        );
    }

    double time = cuda_timer_stop();
    LOG_DEBUG("SOLVER", "Prepare kernel time %.2ems corrector_guess:%s tiled:%s", (double)time, 
        do_corrector_guess ? "true" : "false", 
        is_tiled ? "true" : "false");

    Conjugate_Gardient_Params solver_params = {0};
    solver_params.epsilon = (Real) 1.0e-12;
    solver_params.tolerance = (Real) params.Phi_tolerance;
    solver_params.max_iters = params.Phi_max_iters;
    solver_params.initial_value_or_null = F;

    //Solve A_F*next_F = b_F
    Conjugate_Gardient_Convergence F_converged = conjugate_gradient_solve(&A_F, next_F, b_F, N, anisotrophy_matrix_multiply, &solver_params);
    LOG_DEBUG("SOLVER", "%lli F %s in %i iters with error %e\n", (lli) iter, F_converged.converged ? "converged" : "diverged", F_converged.iters, (double)F_converged.error);

    //Calculate b_U
    cuda_for(0, nx*ny, [=]SHARED(csize i){
        Real T =  U_base[i];
        Real Phi = F[i];
        Real Phi_next = next_F[i];

        b_U[i] = (Real) (T + L*(Phi_next - Phi) + dt*(1-gamma)*T);
    });

    solver_params.tolerance = (Real) params.T_tolerance;
    solver_params.max_iters = params.T_max_iters;
    solver_params.initial_value_or_null = U;

    //Solve A_U*next_U = b_U
    Conjugate_Gardient_Convergence U_converged = conjugate_gradient_solve(&A_U, next_U, b_U, N, cross_matrix_static_multiply, &solver_params);
    LOG_DEBUG("SOLVER", "%lli U %s in %i iters with error %e\n", (lli) iter, U_converged.converged ? "converged" : "diverged", U_converged.iters, (double)U_converged.error);

    if(do_debug && params.do_debug)
    {
        Sim_Map* AfF = _find_claim_temp_map(params, "AfF", params.nx, params.ny, params.time, params.iter);
        Sim_Map* AuU = _find_claim_temp_map(params, "AuU", params.nx, params.ny, params.time, params.iter);
        if(AfF && AuU)
        {
            anisotrophy_matrix_multiply(AfF->data, &A_F, next_F, N);
            cross_matrix_static_multiply(AuU->data, &A_U, next_U, N);

            Real back_error_F_max = cuda_Lmax_distance(AfF->data, b_F, N);
            Real back_error_U_max = cuda_Lmax_distance(AuU->data, b_U, N);
            LOG_DEBUG("SOLVER", "MAX ERROR | F:%e U:%e Epsilon:%e \n", (double) back_error_F_max, (double) back_error_U_max, (double) solver_params.tolerance*2);
        }
    }

    cache_free(&tag);
}

void semi_implicit_and_euler_solver_step_corrector(const Real* F, const Real* U, Real* next_F, Real* next_U, Sim_Params params)
{
    Cache_Tag tag = cache_tag_make();
    int N = params.ny * params.nx;

    Real* temp_state_F = cache_alloc(Real, N, &tag);
    Real* temp_state_U = cache_alloc(Real, N, &tag);

    static int last_num_steps = 0;

    //Init states in such a way that the resutl will probably already be in 
    // next_state (thus no need to copy)
    Real* steps_F[2] = {0};
    Real* steps_U[2] = {0};
    if(last_num_steps % 2 == 0)
    {
        steps_F[0] = next_F;
        steps_U[0] = next_U;
        steps_F[1] = temp_state_F;
        steps_U[1] = temp_state_U;
    }
    else
    {
        steps_F[1] = next_F;
        steps_U[1] = next_U;
        steps_F[0] = temp_state_F;
        steps_U[0] = temp_state_U;
    }

    size_t max_iters = (size_t) params.corrector_max_iters;
    if(params.do_corrector_loop == false)
        max_iters = 0;
    if(max_iters == 0 && params.do_stats_step_residual)
        max_iters = 1;

    //Perform first step
    if(params.solver == SOLVER_TYPE_EXPLICIT_EULER)
        explicit_solver_euler_step_based(F, U, U, steps_F[0], steps_U[0], params, true);
    else
        semi_implicit_solver_step_based(F, U, U, steps_F[0], steps_U[0], params, true);
    for(size_t k = 0; k < max_iters; k++)
    {
        int curr = MOD(k, 2);
        int next = MOD(k + 1, 2);

        log_group();
        //Cycle.
        if(params.solver == SOLVER_TYPE_EXPLICIT_EULER)
            explicit_solver_euler_step_based(F, steps_U[curr], U, steps_F[next], steps_U[next], params, false);
        else
            semi_implicit_solver_step_based(F, steps_U[curr], U, steps_F[next], steps_U[next], params, false);
        if(params.do_stats_step_residual && params.stats)
        {
            Reduce::Stats<Real> stats = cuda_stats_delta(steps_F[curr], steps_F[next], N);
            Real step_residual_max_error = MAX(stats.max, -stats.min);

            params.stats->step_res_L1[k] = (float) stats.L1;
            params.stats->step_res_L2[k] = (float) stats.L2;
            params.stats->step_res_min[k] = (float) stats.min;
            params.stats->step_res_max[k] = (float) stats.max;
            params.stats->step_res_count = max_iters;
            
            LOG_DEBUG("SOLVER", "step residual loop: %i | avg: %e | max: %e", k, 
                (double) stats.L1, (double) step_residual_max_error, params.corrector_tolerance);
        }
        log_ungroup();
        last_num_steps = k;
    }

    //If the ended on step is already next_state dont copy anything
    int final_step = MOD(last_num_steps, 2);
    if(steps_F[final_step] != next_F)
    {
        CUDA_DEBUG_TEST(cudaMemcpyAsync(next_F, steps_F[final_step], (size_t)N*sizeof(Real), cudaMemcpyDeviceToDevice));
        CUDA_DEBUG_TEST(cudaMemcpyAsync(next_U, steps_U[final_step], (size_t)N*sizeof(Real), cudaMemcpyDeviceToDevice));
    }

    //@TODO: stats!
    cache_free(&tag);
}

template <typename T>
void sim_modify_T(Real* device_memory, T* host_memory, size_t count, Sim_Modify modify)
{
    static T* static_device = NULL;
    static size_t static_size = 0;

    if(sizeof(Real) != sizeof(T))
    {
        if(static_size < count)
        {
            cuda_realloc_in_place((void**) &static_device, count*sizeof(T), static_size*sizeof(T), 0);
            static_size = count;
        }

        T* temp_device = static_device;
        if(modify == MODIFY_UPLOAD)
        {
            //Upload: host -> static -> device
            CUDA_DEBUG_TEST(cudaMemcpy(temp_device, host_memory, count*sizeof(T), cudaMemcpyHostToDevice));
            cuda_for(0, (int) count, [=]SHARED(int i){
                device_memory[i] = (Real) temp_device[i];
            });
        }
        else
        {
            //download: device -> static -> host
            cuda_for(0, (int) count, [=]SHARED(int i){
                temp_device[i] = (T) device_memory[i];
            });
            CUDA_DEBUG_TEST(cudaMemcpy(host_memory, temp_device, count*sizeof(T), cudaMemcpyDeviceToHost));
        }
    }
    else
    {
        if(modify == MODIFY_UPLOAD)
            CUDA_DEBUG_TEST(cudaMemcpy(device_memory, host_memory, count*sizeof(T), cudaMemcpyHostToDevice));
        else
            CUDA_DEBUG_TEST(cudaMemcpy(host_memory, device_memory, count*sizeof(T), cudaMemcpyDeviceToHost));
    }
}

extern "C" void sim_modify(void* device_memory, void* host_memory, size_t size, Sim_Modify modify)
{
    if(modify == MODIFY_UPLOAD)
        CUDA_DEBUG_TEST(cudaMemcpy(device_memory, host_memory, size, cudaMemcpyHostToDevice));
    else
        CUDA_DEBUG_TEST(cudaMemcpy(host_memory, device_memory, size, cudaMemcpyDeviceToHost));
}

extern "C" void sim_modify_float(Real* device_memory, float* host_memory, size_t count, Sim_Modify modify)
{   
    sim_modify_T(device_memory, host_memory, count, modify);
}

extern "C" void sim_modify_double(Real* device_memory, double* host_memory, size_t count, Sim_Modify modify)
{   
    sim_modify_T(device_memory, host_memory, count, modify);
}

void exact_solver_step(Real* next_F, Real* next_U, Sim_Params params)
{
    Real dx = (Real) params.L0 / params.nx;
    Real dy = (Real) params.L0 / params.ny;
    Real L0 = params.L0;
    int nx = params.nx;
    int ny = params.ny;
    
    Exact_Params exact_params = get_static_exact_params(params);
    Real t = params.time;
    cuda_for_2D(0, 0, nx, ny, [=]SHARED(int xi, int yi){
        Real x = ((Real) xi + 0.5)*dx - L0/2;
        Real y = ((Real) yi + 0.5)*dy - L0/2;
        Real r = hypot(x, y);

        Real u = exact_u(t, r, exact_params);
        Real phi = exact_phi(t, r, exact_params);
        next_U[xi + yi*nx] = u;
        next_F[xi + yi*nx] = phi;
    });
}

double sim_step(Sim_Map F, Sim_Map U, Sim_Map* next_F, Sim_Map* next_U, Sim_Params params)
{
    double dt = 0;
    if(F.nx != params.nx || F.ny != params.ny || U.nx != params.nx || U.ny != params.ny)
        LOG_ERROR("SOLVER", "Bad sizes provided F:%ix%i U:%ix%i", F.nx, F.ny, U.nx, U.ny);
    else
    {
        sim_realloc(next_F, "F", params.nx, params.ny, params.time, params.iter+1);
        sim_realloc(next_U, "U", params.nx, params.ny, params.time, params.iter+1);

        switch(params.solver)
        {
            case SOLVER_TYPE_SEMI_IMPLICIT:
            case SOLVER_TYPE_EXPLICIT_EULER: {
                semi_implicit_and_euler_solver_step_corrector(F.data, U.data, next_F->data, next_U->data, params);
                dt = params.dt;
            } break;

            case SOLVER_TYPE_EXPLICIT_RK4: {
                explicit_solver_rk4_step(F.data, U.data, next_F->data, next_U->data, params);
                dt = params.dt;
            } break;

            case SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE: {
                dt = explicit_solver_rk4_adaptive_step(F.data, U.data, next_F->data, next_U->data, params);
            } break;

            case SOLVER_TYPE_EXACT: {
                exact_solver_step(next_F->data, next_U->data, params);
                dt = params.dt;
            } break;

            default: break;
        }

        if(params.do_stats && params.stats)
        {
            Reduce::Stats<Real> F_stats = cuda_stats_delta(F.data, next_F->data, params.nx*params.ny);
            Reduce::Stats<Real> U_stats = cuda_stats_delta(U.data, next_U->data, params.nx*params.ny);
            params.stats->iter = params.iter;
            params.stats->time = params.time;

            params.stats->Phi_delta_L1 = (float) F_stats.L1;
            params.stats->Phi_delta_L2 = (float) F_stats.L2;
            params.stats->Phi_delta_min = (float) F_stats.min;
            params.stats->Phi_delta_max = (float) F_stats.max;

            params.stats->T_delta_L1 = (float) U_stats.L1;
            params.stats->T_delta_L2 = (float) U_stats.L2;
            params.stats->T_delta_min = (float) U_stats.min;
            params.stats->T_delta_max = (float) U_stats.max;
        }

        if(params.do_debug)
        {
            Sim_Map* grad_Phi = _find_claim_temp_map(params, "grad_Phi", params.nx, params.ny, params.time, params.iter+1);
            Sim_Map* grad_T = _find_claim_temp_map(params, "grad_T", params.nx, params.ny, params.time, params.iter+1);
            Sim_Map* aniso = _find_claim_temp_map(params, "aniso", params.nx, params.ny, params.time, params.iter+1);
            if(grad_Phi && grad_T && aniso)
                calc_debug_values(next_F->data, next_U->data, grad_Phi->data, grad_T->data, aniso->data, params);
        }

        CUDA_DEBUG_TEST(cudaDeviceSynchronize());
    }
    return dt;
}

#else

#endif

#include "cuda_reduction.cuh"
#include "cuda_random.cuh"

#ifdef COMPILE_BENCHMARKS
static void cache_prepare(int count, int item_size, int N)
{
    Cache_Tag tag = cache_tag_make();
    for(int i = 0; i < count; i++)
        _cache_alloc((size_t) (item_size*N), &tag, SOURCE_INFO());
    cache_free(&tag);
}

extern "C" bool run_benchmarks(int N_)
{
    csize N = (csize) N_;
    cache_prepare(3, sizeof(int), N);
    cache_prepare(3, sizeof(float), N);
    cache_prepare(3, sizeof(double), N);

    Cache_Tag tag = cache_tag_make();
    uint* rand_state = cache_alloc(uint, N, &tag);
    random_map_seed_32(rand_state, N, (uint32_t) clock_ns());

    int GB = 1024*1024*1024;
    {
        double* rand_map = cache_alloc(double, N, &tag);
        random_map_32(rand_map, rand_state, N);
        
        double cpu_time = benchmark(3, [=]{ cpu_reduce(rand_map, N, Reduce::ADD); });
        double thrust_time = benchmark(3, [=]{ thrust_reduce(rand_map, N, Reduce::ADD); });
        double custom_time = benchmark(3, [=]{ cuda_reduce(rand_map, N, Reduce::ADD); });
        double total_gb = (double) N / GB * sizeof(double);
        LOG_OKAY("BENCH", "double (gb/s): cpu %5.2lf | thrust: %5.2lf | custom: %5.2lf (N:%i %s)", 
            total_gb/cpu_time, total_gb/thrust_time, total_gb/custom_time, N, format_bytes((size_t)N * sizeof(double)).str);
        LOG_OKAY("BENCH", "double (time): cpu: %e | thrust: %e | custom: %e", N, cpu_time, thrust_time, custom_time);
    }
    {
        float* rand_map = cache_alloc(float, N, &tag);
        random_map_32(rand_map, rand_state, N);
        
        double cpu_time = benchmark(3, [=]{ cpu_reduce(rand_map, N, Reduce::ADD); });
        double thrust_time = benchmark(3, [=]{ thrust_reduce(rand_map, N, Reduce::ADD); });
        double custom_time = benchmark(3, [=]{ cuda_reduce(rand_map, N, Reduce::ADD); });
        double total_gb = (double) N / GB * sizeof(float);
        LOG_OKAY("BENCH", "float (gb/s) : cpu %5.2lf | thrust: %5.2lf | custom: %5.2lf (N:%i %s)", 
            total_gb/cpu_time, total_gb/thrust_time, total_gb/custom_time, N, format_bytes((size_t)N * sizeof(float)).str);
        LOG_OKAY("BENCH", "float (time) : cpu: %e | thrust: %e | custom: %e", N, cpu_time, thrust_time, custom_time);
    }

    {
        double cpu_time = benchmark(3, [=]{ cpu_reduce(rand_state, N, Reduce::ADD); });
        double thrust_time = benchmark(3, [=]{ thrust_reduce(rand_state, N, Reduce::ADD); });
        double custom_time = benchmark(3, [=]{ cuda_reduce(rand_state, N, Reduce::ADD); });
        double total_gb = (double) N / GB * sizeof(uint);
        LOG_OKAY("BENCH", "uint (gb/s)  : cpu %5.2lf | thrust: %5.2lf | custom: %5.2lf (N:%i %s)", 
            total_gb/cpu_time, total_gb/thrust_time, total_gb/custom_time, N, format_bytes((size_t)N * sizeof(uint)).str);
        LOG_OKAY("BENCH", "uint (time)  : cpu: %e | thrust: %e | custom: %e", N, cpu_time, thrust_time, custom_time);
    }

    cache_free(&tag);
    return true;
}
#else
extern "C" bool run_benchmarks(int N)
{
    (void) N;
    return false;
}
#endif

#include "cuda_examples.cuh"
extern "C" bool run_tests()
{
    test_all_examples(3);
    #ifdef TEST_CUDA_FOR_IMPL
    test_tiled_for((uint64_t) clock_ns());
    test_tiled_for_2D((uint64_t) clock_ns());
    #endif
    #ifdef TEST_CUDA_REDUCTION_IMPL
    test_reduce((uint64_t) clock_ns());
    #endif

    return true;
}