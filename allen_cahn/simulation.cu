// ==================== SWITCHES ==========================

// #define COMPILE_BENCHMARKS
// #define COMPILE_TESTS
#define COMPILE_SIMULATION
// #define COMPILE_THRUST

#define USE_CUSTOM_REDUCE
#define USE_TILED_FOR 
#define USE_FLOATS

//==================== IMPLEMENTATION ===================
#ifdef COMPILE_TESTS
#define TEST_CUDA_ALL
#endif

#include "simulation.h"
#include "cuda_util.cuh"
#include "cuda_reduction.cuh"
#include "cuda_for.cuh"
#include "cuda_random.cuh"

#ifdef COMPILE_SIMULATION

#ifndef USE_CUSTOM_REDUCE
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#endif

#if 1
SHARED Real custom_hypot(Real y, Real x) { return (Real) hypotf((float)y, (float)x); }
SHARED Real custom_atan2(Real y, Real x) { return (Real) atan2f((float) y, (float) x); }
SHARED Real custom_cos(Real theta) { return (Real) cosf((float)theta); }
#else
SHARED Real custom_hypot(Real y, Real x) { return hypot(y, x); }
SHARED Real custom_atan2(Real y, Real x) { return atan2(y, x); }
SHARED Real custom_cos(Real theta) { return cos(theta); }
#endif

Real vector_dot_product(const Real *a, const Real *b, int ny)
{
    #ifdef USE_CUSTOM_REDUCE
    return cuda_dot_product(a, b, ny);
    #else
    // wrap raw pointers to device memory with device_ptr
    thrust::device_ptr<const Real> d_a(a);
    thrust::device_ptr<const Real> d_b(b);

    // inner_product implements a mathematical dot product
    return thrust::inner_product(d_a, d_a + ny, d_b, 0.0);
    #endif
}

Real vector_max(const Real *a, int N)
{
    #ifdef USE_CUSTOM_REDUCE
    return cuda_max(a, N);
    #else
    thrust::device_ptr<const Real> d_a(a);
    return *(thrust::max_element(d_a, d_a + N));
    #endif
}

Real vector_get_l2_dist(const Real* a, const Real* b, int N)
{
    #ifdef USE_CUSTOM_REDUCE
    return cuda_L2_distance(a, b, N)/ sqrt((Real) N);
    #else
    Cache_Tag tag = cache_tag_make();
    Real* temp = cache_alloc(Real, N, &tag);
    cuda_for(0, N, [=]SHARED(int i){
        temp[i] = a[i] - b[i];
    });

    Real temp_dot_temp = vector_dot_product(temp, temp, N);
    Real error = sqrt(temp_dot_temp/N);
    cache_free(&tag);
    return error;
    #endif
}

Real vector_get_max_dist(const Real* a, const Real* b, int N)
{
    #ifdef USE_CUSTOM_REDUCE
    return cuda_L2_distance(a, b, N);
    #else
    Cache_Tag tag = cache_tag_make();
    Real* temp = cache_alloc(Real, N, &tag);
    cuda_for(0, N, [=]SHARED(int i){
        temp[i] = a[i] - b[i];
    });

    Real temp_dot_temp = vector_max(temp, N);
    Real error = sqrt(temp_dot_temp/N);
    cache_free(&tag);
    return error;
    #endif
}


Real vector_euclid_norm(const Real* vector, int N)
{
    #ifdef USE_CUSTOM_REDUCE
    return cuda_L2_norm(vector, N)/ sqrt((Real) N);
    #else
    Real dot = vector_dot_product(vector, vector, N);
    return sqrt(dot / N);
    #endif
}


SHARED Real* at_mod(Real* map, int x, int y, int nx, int ny)
{
    #define AT_MOD_MODE 1
    #if AT_MOD_MODE == 0
        //95 ms
        int x_mod = MOD(x, nx);
        int y_mod = MOD(y, ny);
    #elif AT_MOD_MODE == 1
        //63 ms
        //@NOTE: this only works for x, y maximaly nx, ny respectively otuside of their proper range.
        // In our application this is enough.
        //@NOTE: this seems to be the fastest mode
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

    #elif AT_MOD_MODE == 2
        //85 ms
        int x_mod = (x + nx) % nx;
        int y_mod = (y + ny) % ny;
    #endif
    return &map[x_mod + y_mod*nx];
}

void explicit_solver_resize(Explicit_Solver* solver, int nx, int ny)
{
    size_t N = (size_t)nx*(size_t)ny;
    size_t N_old = (size_t)solver->nx*(size_t)solver->ny;
    if(solver->nx != nx || solver->ny != ny)
    {
        //Big evil programming practices because we are cool and we know
        // what we are doing and dont care much about what others have to
        // say
        Real* debug_maps = (Real*) (void*) &solver->debug_maps;
        for(uint i = 0; i < sizeof(solver->debug_maps) / sizeof(Real*); i++)
            cuda_realloc_in_place((void**) &debug_maps[i], N*sizeof(Real), N_old*sizeof(Real), REALLOC_ZERO);

        solver->nx = nx;
        solver->ny = ny;
    }
}

void explicit_state_resize(Explicit_State* state, int nx, int ny)
{
    size_t N = (size_t)nx*(size_t)ny;
    size_t N_old = (size_t)state->nx*(size_t)state->ny;
    if(state->nx != nx || state->ny != ny)
    {
        cuda_realloc_in_place((void**) &state->F, N*sizeof(Real), N_old*sizeof(Real), REALLOC_ZERO);
        cuda_realloc_in_place((void**) &state->U, N*sizeof(Real), N_old*sizeof(Real), REALLOC_ZERO);

        state->nx = nx;
        state->ny = ny;
    }
}

struct Bundled {
    Real Phi;
    Real T;
};

struct Explicit_Solve_Result {
    Real dt_Phi;
    Real dt_T;
};

struct Explicit_Solve_Debug {
    Real grad_Phi;
    Real grad_T;
    Real g_theta;
    Real theta;
    Real reaction_term;
};

SHARED Real f0(Real phi)
{
	return phi*(1 - phi)*(phi - 1.0f/2);
}

extern "C" void explicit_solver_newton_step(Explicit_Solver* solver, Explicit_State state, Explicit_State* next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    int nx = params.nx;
    int ny = params.ny;

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

    Real* in_F = state.F;
    Real* in_U = state.U;

    Real* out_F = next_state->F;
    Real* out_U = next_state->U;

    // Cache_Tag tag = cache_tag_make();
    // int N = params.nx * params.ny;
    // Real* out_F = cache_alloc(Real, N, &tag);
    // Real* out_U = cache_alloc(Real, N, &tag);

    cuda_tiled_for_2D<1, 1, Bundled>(0, 0, params.nx, params.ny,
        [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry) -> Bundled{
            csize x_mod = x;
            csize y_mod = y;

            if(x_mod < 0)
                x_mod += nx;
            else if(x_mod >= nx)
                x_mod -= nx;

            if(y_mod < 0)
                y_mod += ny;
            else if(y_mod >= ny)
                y_mod -= ny;

            int I = x_mod + y_mod*nx;

            Real Phi = in_F[I];
            Real T = in_U[I];
            return Bundled{Phi, T};
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Bundled* shared){
            Bundled C = shared[tx   + ty*tile_size_x];
            Bundled E = shared[tx+1 + ty*tile_size_x];
            Bundled W = shared[tx-1 + ty*tile_size_x];
            Bundled N = shared[tx   + (ty+1)*tile_size_x];
            Bundled S = shared[tx   + (ty-1)*tile_size_x];

            Real grad_Phi_x = (E.Phi - W.Phi)*one_over_2dx;
            Real grad_Phi_y = (N.Phi - S.Phi)*one_over_2dx;
            Real grad_Phi_norm = custom_hypot(grad_Phi_x, grad_Phi_y);

            Real theta = custom_atan2(grad_Phi_y, grad_Phi_x);
            Real g_theta = (Real) 1 - S0*custom_cos(m0*theta + theta0);

            Real laplace_Phi = (W.Phi - 2*C.Phi + E.Phi)*one_over_dx2 + (S.Phi - 2*C.Phi + N.Phi)*one_over_dy2;
            Real laplace_T = (W.T - 2*C.T + E.T)*one_over_dx2 +     (S.T - 2*C.T + N.T)*one_over_dy2;

            Real k0 = g_theta*f0(C.Phi)*k0_factor;
            Real k2 = grad_Phi_norm*k2_factor;
            Real k1 = g_theta*k1_factor;
            Real corr = 1 + k2*dt_L;

            Real dt_Phi = 0;
            if(do_corrector_guess)
                dt_Phi = (k1*laplace_Phi + k0 - k2*(C.T - Tm + dt*laplace_T))/corr;
            else
                dt_Phi = k1*laplace_Phi + k0 - k2*(C.T - Tm);

            Real dt_T = laplace_T + L*dt_Phi; 

            out_F[x + y*nx] = C.Phi + dt_Phi*dt;
            out_U[x + y*nx] = C.T + dt_T*dt;
        });

    // CUDA_DEBUG_TEST(cudaMemcpy(next_state->F, out_F, N*sizeof(Real), cudaMemcpyDeviceToDevice));
    // CUDA_DEBUG_TEST(cudaMemcpy(next_state->U, out_U, N*sizeof(Real), cudaMemcpyDeviceToDevice));
    // cache_free(&tag);
}

struct Explicit_Blend_State {
    Real weight;
    Explicit_State state;
};

template<typename ... States>
void explicit_solver_solve_lin_combination(Explicit_State* out, Allen_Cahn_Params params, States... state_args)
{
    int nx = params.nx;
    int ny = params.ny;
    Real* out_F = out->F;
    Real* out_U = out->U;

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

    #if 1
    cuda_tiled_for_2D<1, 1, Bundled>(0, 0, nx, ny,
        [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry) -> Bundled{
            csize x_mod = x;
            csize y_mod = y;

            if(x_mod < 0)
                x_mod += nx;
            else if(x_mod >= nx)
                x_mod -= nx;

            if(y_mod < 0)
                y_mod += ny;
            else if(y_mod >= ny)
                y_mod -= ny;

            int I = x_mod + y_mod*nx;

            Real T = 0;
            Real Phi = 0;
            #pragma unroll
            for(int i = 0; i < state_count; i++)
            {
                Phi += states[i].state.F[I] * states[i].weight;
                T += states[i].state.U[I] * states[i].weight;
            }

            return Bundled{Phi, T};
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Bundled* shared){
            Bundled C = shared[tx   + ty*tile_size_x];
            Bundled E = shared[tx+1 + ty*tile_size_x];
            Bundled W = shared[tx-1 + ty*tile_size_x];
            Bundled N = shared[tx   + (ty+1)*tile_size_x];
            Bundled S = shared[tx   + (ty-1)*tile_size_x];

            Real grad_Phi_x = (E.Phi - W.Phi)*one_over_2dx;
            Real grad_Phi_y = (N.Phi - S.Phi)*one_over_2dx;
            Real grad_Phi_norm = custom_hypot(grad_Phi_x, grad_Phi_y);

            Real theta = custom_atan2(grad_Phi_y, grad_Phi_x);
            Real g_theta = (Real) 1 - S0*custom_cos(m0*theta + theta0);

            Real laplace_Phi = (W.Phi - 2*C.Phi + E.Phi)*one_over_dx2 + (S.Phi - 2*C.Phi + N.Phi)*one_over_dy2;
            Real laplace_T = (W.T - 2*C.T + E.T)*one_over_dx2 +     (S.T - 2*C.T + N.T)*one_over_dy2;

            Real k0 = g_theta*f0(C.Phi)*k0_factor;
            Real k2 = grad_Phi_norm*k2_factor;
            Real k1 = g_theta*k1_factor;
            Real corr = 1 + k2*dt_L;

            Real dt_Phi = 0;
            if(do_corrector_guess)
                dt_Phi = (k1*laplace_Phi + k0 - k2*(C.T - Tm + dt*laplace_T))/corr;
            else
                dt_Phi = k1*laplace_Phi + k0 - k2*(C.T - Tm);

            Real dt_T = laplace_T + L*dt_Phi; 

            out_F[x + y*nx] = dt_Phi;
            out_U[x + y*nx] = dt_T;
        });
    #else

    Cache_Tag tag = cache_tag_make();
    int N = params.nx * params.ny;
    Real* F = cache_alloc(Real, N, &tag);
    Real* U = cache_alloc(Real, N, &tag);
    Real* laplace_U = cache_alloc(Real, N, &tag);
    Real* laplace_F = cache_alloc(Real, N, &tag);
    Real* grad_F_x_2dx = cache_alloc(Real, N, &tag);
    Real* grad_F_y_2dy = cache_alloc(Real, N, &tag);

    cuda_tiled_for_2D<1, 1, Bundled>(0, 0, params.nx, params.ny,
        [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry) -> Bundled{
            csize x_mod = x;
            csize y_mod = y;

            if(x_mod < 0)
                x_mod += nx;
            else if(x_mod >= nx)
                x_mod -= nx;

            if(y_mod < 0)
                y_mod += ny;
            else if(y_mod >= ny)
                y_mod -= ny;

            int I = x_mod + y_mod*nx;

            Real T = 0;
            Real Phi = 0;
            #pragma unroll
            for(int i = 0; i < state_count; i++)
            {
                Phi += states[i].state.F[I] * states[i].weight;
                T += states[i].state.U[I] * states[i].weight;
            }

            U[I] = T;
            F[I] = Phi;
            return Bundled{Phi, T};
        },
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Bundled* shared){
            Bundled C = shared[tx   + ty*tile_size_x];
            Bundled E = shared[tx+1 + ty*tile_size_x];
            Bundled W = shared[tx-1 + ty*tile_size_x];
            Bundled N = shared[tx   + (ty+1)*tile_size_x];
            Bundled S = shared[tx   + (ty-1)*tile_size_x];

            csize i = x + y*nx;
            grad_F_x_2dx[i] = (E.Phi - W.Phi);
            grad_F_y_2dy[i] = (N.Phi - S.Phi);
            laplace_F[i] = (W.Phi - 2*C.Phi + E.Phi)*one_over_dx2 + (S.Phi - 2*C.Phi + N.Phi)*one_over_dy2;
            laplace_U[i] = (W.T - 2*C.T + E.T)*one_over_dx2 +       (S.T - 2*C.T + N.T)*one_over_dy2;
        });

    cuda_for(0, N, [=]SHARED(csize i){
        Real T =  U[i];
        Real Phi = F[i];

        Real grad_Phi_x = grad_F_x_2dx[i]*one_over_2dx;
        Real grad_Phi_y = grad_F_y_2dy[i]*one_over_2dy;
        Real grad_Phi_norm = custom_hypot(grad_Phi_x, grad_Phi_y);

        Real laplace_Phi = laplace_F[i];
        Real laplace_T = laplace_U[i];

        Real theta = custom_atan2(grad_Phi_y, grad_Phi_x);
        Real g_theta = (Real) 1 - S0*custom_cos(m0*theta + theta0);

        Real k0 = g_theta*f0(Phi)*k0_factor;
        Real k2 = grad_Phi_norm*k2_factor;
        Real k1 = g_theta*k1_factor;
        Real corr = 1 + k2*dt_L;

        Real dt_Phi = 0;
        if(do_corrector_guess)
            dt_Phi = (k1*laplace_Phi + k0 - k2*(T - Tm + dt*laplace_T))/corr;
        else
            dt_Phi = k1*laplace_Phi + k0 - k2*(T - Tm);

        Real dt_T = laplace_T + L*dt_Phi; 

        out_F[i] = dt_Phi;
        out_U[i] = dt_T;
    });
    cache_free(&tag);
    #endif
}
void explicit_solver_debug_step(Explicit_Solver* solver, Explicit_State state, Allen_Cahn_Params params)
{
    int nx = params.nx;
    int ny = params.ny;
    Real* F = state.F;
    Real* U = state.U;
    Real* grad_F = solver->debug_maps.grad_phi;
    Real* grad_U = solver->debug_maps.grad_T;
    Real* aniso = solver->debug_maps.aniso_factor;
    Real* perlin = solver->debug_maps.perlin;
    Real* simplex = solver->debug_maps.simplex;
    static uint32_t _iter = 0;
    uint32_t iter = _iter++;
    Real S0 = (Real) params.S;
    Real m0 = (Real) params.m0;
    Real theta0 = (Real) params.theta0;
    cuda_for_2D(0, 0, nx, ny, [=]SHARED(int x, int y){
        Real T = *at_mod(U, x, y, nx, ny);
        Real Phi = *at_mod(F, x, y, nx, ny);

        Real Phi_U = *at_mod(F, x, y + 1, nx, ny);
        Real Phi_D = *at_mod(F, x, y - 1, nx, ny);
        Real Phi_R = *at_mod(F, x + 1, y, nx, ny);
        Real Phi_L = *at_mod(F, x - 1, y, nx, ny);

        Real T_U = *at_mod(U, x, y + 1, nx, ny);
        Real T_D = *at_mod(U, x, y - 1, nx, ny);
        Real T_R = *at_mod(U, x + 1, y, nx, ny);
        Real T_L = *at_mod(U, x - 1, y, nx, ny);

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
    });

    perlin2d_generate(perlin, nx, ny, 1, 1, 8, iter);
    simplex2d_generate(simplex, nx, ny, 3, 3, 8, iter);
}

void explicit_solver_rk4_step(Explicit_Solver* solver, Explicit_State state, Explicit_State* next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    Cache_Tag tag = cache_tag_make();
    int N = params.ny * params.nx;

    Explicit_State steps[4] = {0};
    for(int i = 0; i < (int) STATIC_ARRAY_SIZE(steps); i++)
    {
        steps[i].F = cache_alloc(Real, N, &tag);
        steps[i].U = cache_alloc(Real, N, &tag);
        steps[i].nx = params.nx;
        steps[i].ny = params.ny;
    }

    Explicit_State k1 = steps[0];
    Explicit_State k2 = steps[1];
    Explicit_State k3 = steps[2];
    Explicit_State k4 = steps[3];

    Real dt = (Real) params.dt;
    using W = Explicit_Blend_State;
    explicit_solver_solve_lin_combination(&k1, params, W{1, state});
    explicit_solver_solve_lin_combination(&k2, params, W{1, state}, W{dt * (Real) 0.5, k1});
    explicit_solver_solve_lin_combination(&k3, params, W{1, state}, W{dt * (Real) 0.5, k2});
    explicit_solver_solve_lin_combination(&k4, params, W{1, state}, W{dt * 1, k3});

    Real* out_F = next_state->F;
    Real* out_U = next_state->U;
    cuda_for(0, params.ny*params.nx, [=]SHARED(int i){
        out_F[i] =  state.F[i] + dt/6*(k1.F[i] + 2*k2.F[i] + 2*k3.F[i] + k4.F[i]);
        out_U[i] =  state.U[i] + dt/6*(k1.U[i] + 2*k2.U[i] + 2*k3.U[i] + k4.U[i]);
    });

    if(do_debug)
        explicit_solver_debug_step(solver, state, params);

    cache_free(&tag);
    CUDA_DEBUG_TEST(cudaDeviceSynchronize());
}

double explicit_solver_rk4_adaptive_step(Explicit_Solver* solver, Explicit_State state, Explicit_State* next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    Cache_Tag tag = cache_tag_make();
    int N = params.ny * params.nx;

    static Real _initial_step = 0;
    if(iter == 0)
        _initial_step = (Real) params.dt;

    Real tau = _initial_step;
    Explicit_State steps[5] = {0};
    for(int i = 0; i < (int) STATIC_ARRAY_SIZE(steps); i++)
    {
        steps[i].F = cache_alloc(Real, N, &tag);
        steps[i].U = cache_alloc(Real, N, &tag);
        steps[i].nx = params.nx;
        steps[i].ny = params.ny;
    }

    Real* Epsilon_F = cache_alloc(Real, N, &tag);
    Real* Epsilon_U = cache_alloc(Real, N, &tag);
    Real epsilon_F = 0;
    Real epsilon_U = 0;

    Explicit_State k1 = steps[0];
    Explicit_State k2 = steps[1];
    Explicit_State k3 = steps[2];
    Explicit_State k4 = steps[3];
    Explicit_State k5 = steps[4];

    using W = Explicit_Blend_State;
    explicit_solver_solve_lin_combination(&k1, params, W{1, state});

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
        
        explicit_solver_solve_lin_combination(&k2, params, W{1, state}, W{tau/3, k1});
        explicit_solver_solve_lin_combination(&k3, params, W{1, state}, W{tau/6, k1}, W{tau/6, k2});
        explicit_solver_solve_lin_combination(&k4, params, W{1, state}, W{tau/8, k1}, W{tau*3/8, k3});
        explicit_solver_solve_lin_combination(&k5, params, W{1, state}, W{tau/2, k1}, W{-tau*3/2, k3}, W{tau*2, k4});

        cuda_for(0, params.ny*params.nx, [=]SHARED(int i){
            Real F = (Real)0.2*k1.F[i] - (Real)0.9*k3.F[i] + (Real)0.8*k4.F[i] - (Real)0.1*k5.F[i];
            Real U = (Real)0.2*k1.U[i] - (Real)0.9*k3.U[i] + (Real)0.8*k4.U[i] - (Real)0.1*k5.U[i];

            Epsilon_F[i] = F >= 0 ? F : -F;
            Epsilon_U[i] = U >= 0 ? U : -U;
        });

        epsilon_F = vector_max(Epsilon_F, N);
        epsilon_U = vector_max(Epsilon_U, N);

        if(epsilon_F < params.Phi_tolerance && epsilon_U < params.T_tolerance)
            converged = true;

        Real epsilon = (Real) MAX(epsilon_F + epsilon_U, 1e-8);
        Real delta = (Real) MAX(params.Phi_tolerance + params.T_tolerance, 1e-8);
        used_tau = tau;
        tau = pow(delta / epsilon, (Real)0.2)*4/5*tau;
    }

    Real* next_F = next_state->F;
    Real* next_U = next_state->U;
    cuda_for(0, params.ny*params.nx, [=]SHARED(int i){
        next_F[i] = state.F[i] + used_tau*((Real)1.0/6*(k1.F[i] + k5.F[i]) + (Real)2.0/3*k4.F[i]);
        next_U[i] = state.U[i] + used_tau*((Real)1.0/6*(k1.U[i] + k5.U[i]) + (Real)2.0/3*k4.U[i]);
    });

    LOG("SOLVER", converged ? LOG_DEBUG : LOG_WARN, "rk4-adaptive %s in %i iters with error F:%lf | U:%lf | tau:%e", converged ? "converged" : "diverged", i, (double) epsilon_F, (double) epsilon_U, (double)used_tau);
    _initial_step = tau;

    if(do_debug)
        explicit_solver_debug_step(solver, state, params);

    cache_free(&tag);
    CUDA_DEBUG_TEST(cudaDeviceSynchronize());
    return (double) used_tau;
}

double explicit_solver_choose(Solver_Type type, Explicit_Solver* solver, Explicit_State state, Explicit_State* next_state, Allen_Cahn_Params params, size_t iter)
{
    if(type == SOLVER_TYPE_EXPLICIT)
    {
        explicit_solver_newton_step(solver, state, next_state, params, iter, params.do_debug);
        return params.dt;
    }
    if(type == SOLVER_TYPE_EXPLICIT_RK4)
    {
        explicit_solver_rk4_step(solver, state, next_state, params, iter, params.do_debug);
        return params.dt;
    }
    if(type == SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE)
    {
        return explicit_solver_rk4_adaptive_step(solver, state, next_state, params, iter, params.do_debug);
    }

    assert(false);
    return false;
}


double explicit_solver_choose_and_copute_step_residual(Solver_Type type, Explicit_Solver* solver, Explicit_State state, Explicit_State* next_state, Allen_Cahn_Params params, size_t iter, Allen_Cahn_Stats* stats_or_null)
{
    double advance_by = explicit_solver_choose(type, solver, state, next_state, params, iter);
    if(params.do_stats_step_residual && stats_or_null)
    {
        Explicit_State combined_state = state;
        combined_state.U = next_state->U;

        Explicit_State corrected_next = {0};
        Cache_Tag tag = cache_tag_make();
        int N = params.ny * params.nx;
        corrected_next.F = cache_alloc(Real, N, &tag);
        corrected_next.U = cache_alloc(Real, N, &tag);
        corrected_next.nx = params.nx;
        corrected_next.ny = params.ny;

        Allen_Cahn_Params changed_params = params;
        changed_params.do_debug = false;
        changed_params.do_stats = false;
        changed_params.do_stats_step_residual = false;

        explicit_solver_choose(type, solver, combined_state, &corrected_next, changed_params, iter);
        stats_or_null->L2_step_residuals[0] = vector_get_l2_dist(corrected_next.F, next_state->F, N);
        stats_or_null->Lmax_step_residuals[0] = vector_get_max_dist(corrected_next.F, next_state->F, N);
        stats_or_null->step_residuals = 1;

        LOG_DEBUG("SOLVER", "%lli step residual | avg: %e | max: %e", (long long) iter, (double) stats_or_null->L2_step_residuals[0], (double) stats_or_null->Lmax_step_residuals[0]);

        cache_free(&tag);
    }

    return advance_by;
}

void explicit_solver_get_maps(Explicit_Solver* solver, Explicit_State state, Sim_Map* maps, int map_count)
{
    int __map_i = 0;
    memset(maps, 0, sizeof maps[0] * (size_t) map_count);

    #define ASSIGN_MAP_NAMED(var_ptr, var_name) \
        if(__map_i < map_count) \
        { \
            maps[__map_i].data = var_ptr; \
            maps[__map_i].name = var_name; \
            maps[__map_i].nx = solver->nx; \
            maps[__map_i].ny = solver->ny; \
            __map_i += 1; \
        }\

    #define ASSIGN_MAP(var_ptr) ASSIGN_MAP_NAMED(var_ptr, #var_ptr) 

    ASSIGN_MAP_NAMED(state.F, "Phi");            
    ASSIGN_MAP_NAMED(state.U, "T");            
    ASSIGN_MAP_NAMED(solver->debug_maps.grad_phi, "grad_phi");
    ASSIGN_MAP_NAMED(solver->debug_maps.grad_T, "grad_T");
    ASSIGN_MAP_NAMED(solver->debug_maps.aniso_factor, "aniso_factor");
    ASSIGN_MAP_NAMED(solver->debug_maps.reaction, "reaction");
    ASSIGN_MAP_NAMED(solver->debug_maps.step_residual, "step_residual");
    ASSIGN_MAP_NAMED(solver->debug_maps.perlin, "perlin");
    ASSIGN_MAP_NAMED(solver->debug_maps.simplex, "simplex");
}

struct Cross_Matrix_Static {
    Real C;
    Real U;
    Real D;
    Real L;
    Real R;

    int nx;
    int ny;
};

struct Cross_Matrix {
    Real* C;
    Real* U;
    Real* D;
    Real* L;
    Real* R;

    int nx;
    int ny;
};

struct Anisotrophy_Matrix {
    Real* anisotrophy;
    Real X;
    Real Y;
    Real C_minus_one;

    int nx;
    int ny;
};

void cross_matrix_static_multiply(Real* out, const void* _A, const Real* vec, int N)
{
    Cross_Matrix_Static A = *(Cross_Matrix_Static*)_A;
    int nx = A.nx;
    int ny = A.ny;

    #ifdef USE_TILED_FOR
    cuda_tiled_for_2D_modular<1, 1, TILED_FOR_PERIODIC_SMALL_R>(vec, nx, ny, 
        [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Real* shared){
            Real val = shared[tx + ty*tile_size_x]*A.C;
            val += shared[tx+1 + ty*tile_size_x]*A.R;
            val += shared[tx-1 + ty*tile_size_x]*A.L;
            val += shared[tx   + (ty+1)*tile_size_x]*A.U;
            val += shared[tx   + (ty-1)*tile_size_x]*A.D;

            out[x + y*nx] = val;
        }
    );
    #else
    cuda_for_2D(0, 0, nx, ny, [=]SHARED(int x, int y){
        int i = x + y*nx;
        Real val = vec[i]*A.C;
        val += *at_mod((Real*) vec, x+1, y, nx, ny)*A.R;
        val += *at_mod((Real*) vec, x-1, y, nx, ny)*A.L;
        val += *at_mod((Real*) vec, x, y+1, nx, ny)*A.U;
        val += *at_mod((Real*) vec, x, y-1, nx, ny)*A.D;
        out[i] = val;
    });
    #endif
}

void cross_matrix_multiply(Real* out, const void* _A, const Real* x, int N)
{
    Cross_Matrix A = *(Cross_Matrix*)_A;
    int nx = A.nx;
    cuda_for(0, N, [=]SHARED(int i){
        Real val = 0;
        val += x[i]*A.C[i];
        if(i+1 < N)  val += x[i+1]*A.R[i];
        if(i-1 >= 0) val += x[i-1]*A.L[i];
        if(i+nx < N)  val += x[i+nx]*A.U[i];
        if(i-nx >= 0) val += x[i-nx]*A.D[i];

        out[i] = val;
    });
}

void anisotrophy_matrix_multiply(Real* out, const void* _A, const Real* vec, int N)
{
    Anisotrophy_Matrix A = * (Anisotrophy_Matrix*)_A;
    int nx = A.nx;
    int ny = A.ny;

    #ifdef USE_TILED_FOR
    cuda_tiled_for_2D_modular<1, 1, TILED_FOR_PERIODIC_SMALL_R>(vec, nx, ny, 
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
    #else
    cuda_for_2D(0, 0, nx, ny, [=]SHARED(int x, int y){
        int i = x + y*nx;
        Real s = A.anisotrophy[i];
        Real X = A.X*s;
        Real Y = A.Y*s;
        Real C = 1 + A.C_minus_one*s;

        Real val = vec[i]*C;
        val += *at_mod((Real*) vec, x+1, y, nx, ny)*X;
        val += *at_mod((Real*) vec, x-1, y, nx, ny)*X;
        val += *at_mod((Real*) vec, x, y+1, nx, ny)*Y;
        val += *at_mod((Real*) vec, x, y-1, nx, ny)*Y;
        out[i] = val;
    });
    #endif
}

typedef struct Conjugate_Gardient_Params {
    Real epsilon;
    Real tolerance;
    int max_iters;

    Real* initial_value_or_null;
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

        r_dot_r = vector_dot_product(r, r, N);
    }
    else
    {
        CUDA_DEBUG_TEST(cudaMemsetAsync(x, 0, sizeof(Real)*(size_t)N, stream1));
        CUDA_DEBUG_TEST(cudaMemcpyAsync(r, b, sizeof(Real)*(size_t)N, cudaMemcpyDeviceToDevice, stream2));
        CUDA_DEBUG_TEST(cudaMemcpyAsync(p, b, sizeof(Real)*(size_t)N, cudaMemcpyDeviceToDevice, stream3));

        r_dot_r = vector_dot_product(b, b, N);
        // CUDA_DEBUG_TEST(cudaDeviceSynchronize());
    }

    int iter = 0;
    if(r_dot_r >= scaled_squared_tolerance || iter == 0)
    {
        for(; iter < params.max_iters; iter++)
        {
            matrix_mul_func(Ap, A, p, N);
            
            Real p_dot_Ap = vector_dot_product(p, Ap, N);
            Real alpha = r_dot_r / MAX(p_dot_Ap, params.epsilon);
            
            cuda_for(0, N, [=]SHARED(int i){
                x[i] = x[i] + alpha*p[i];
                r[i] = r[i] - alpha*Ap[i];
            });

            Real r_dot_r_new = vector_dot_product(r, r, N);
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

void semi_implicit_solver_resize(Semi_Implicit_Solver* solver, int nx, int ny)
{
    if(solver->nx != nx || solver->ny != ny)
    {
        //Big evil programming practices because we are cool and we know
        // what we are doing and dont care much about what others have to
        // say
        //@TODO: make this on demand load
        size_t N = (size_t) (ny*nx);
        size_t N_old = (size_t) (solver->ny*solver->nx);

        void** maps = (void**) (void*) &solver->maps;
        for(uint i = 0; i < sizeof(solver->maps) / sizeof(Real*); i++)
            cuda_realloc_in_place(&maps[i], N*sizeof(Real), N_old*sizeof(Real), REALLOC_ZERO);

        void** debug_maps = (void**) (void*) &solver->debug_maps;
        for(uint i = 0; i < sizeof(solver->debug_maps) / sizeof(Real*); i++)
            cuda_realloc_in_place(&debug_maps[i], N*sizeof(Real), N_old*sizeof(Real), REALLOC_ZERO);

        solver->nx = nx;
        solver->ny = ny;
    }
}

void semi_implicit_solver_step_based(Semi_Implicit_Solver* solver, Real* F, Real* U, Real* U_base, Semi_Implicit_State next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    Real dx = (Real) params.L0 / solver->nx;
    Real dy = (Real) params.L0 / solver->ny;

    int nx = solver->nx;
    int ny = solver->ny;
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
    
    Real* F_next = next_state.F;
    Real* U_next = next_state.U;
    
    Real* b_F = solver->maps.b_F;
    Real* b_U = solver->maps.b_U;

    Real one_over_2dx = 1/(2*dx);
    Real one_over_2dy = 1/(2*dy);
    Real one_over_dx2 = 1/(dx*dx);
    Real one_over_dy2 = 1/(dy*dy);
    Real k0_factor = a/(xi*xi * alpha);
    Real k2_factor = b*beta/alpha;
    Real k1_factor = 1/alpha;
    Real dt_L = dt*L;

    Anisotrophy_Matrix A_F = {0};
    A_F.anisotrophy = solver->maps.anisotrophy;
    A_F.C_minus_one = 2*dt/(dx*dx) + 2*dt/(dy*dy);
    A_F.X = -dt/(dx*dx);
    A_F.Y = -dt/(dy*dy);
    A_F.nx = nx;
    A_F.ny = ny;

    Cross_Matrix_Static A_U = {0};
    A_U.C = 1 + 2*dt/(dx*dx) + 2*dt/(dy*dy);
    A_U.R = -dt/(dx*dx);
    A_U.L = -dt/(dx*dx);
    A_U.U = -dt/(dy*dy);
    A_U.D = -dt/(dy*dy);
    A_U.nx = nx;
    A_U.ny = ny;

    bool do_corrector_guess = params.do_corrector_guess;
    bool is_tiled = true;
    Cache_Tag tag = cache_tag_make();

    //@TODO: factor out
    static cudaEvent_t start = NULL;
    static cudaEvent_t stop = NULL;
    if(start == NULL || stop == NULL)
    {
        CUDA_TEST(cudaEventCreate(&start));
        CUDA_TEST(cudaEventCreate(&stop));
    }
    CUDA_TEST(cudaEventRecord(start, 0));

    if(do_corrector_guess)
    {
        #if 1
        Real* laplace_U = cache_alloc(Real, N, &tag);
        Real* laplace_F = cache_alloc(Real, N, &tag);
        Real* grad_F_x_2dx = cache_alloc(Real, N, &tag);
        Real* grad_F_y_2dy = cache_alloc(Real, N, &tag);

        cuda_tiled_for_2D<1, 1, Bundled>(0, 0, nx, ny,
            [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry) -> Bundled{
                csize x_mod = x;
                csize y_mod = y;

                if(x_mod < 0)
                    x_mod += nx;
                else if(x_mod >= nx)
                    x_mod -= nx;

                if(y_mod < 0)
                    y_mod += ny;
                else if(y_mod >= ny)
                    y_mod -= ny;

                Real T = (Real) U[x_mod + y_mod*nx];
                Real Phi = (Real) F[x_mod + y_mod*ny];
                return Bundled{Phi, T};
            },
            [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Bundled* shared){
                Bundled C = shared[tx   + ty*tile_size_x];
                Bundled E = shared[tx+1 + ty*tile_size_x];
                Bundled W = shared[tx-1 + ty*tile_size_x];
                Bundled N = shared[tx   + (ty+1)*tile_size_x];
                Bundled S = shared[tx   + (ty-1)*tile_size_x];

                csize i = x + y*nx;
                grad_F_x_2dx[i] = (E.Phi - W.Phi);
                grad_F_y_2dy[i] = (N.Phi - S.Phi);
                laplace_F[i] = (W.Phi - 2*C.Phi + E.Phi)*one_over_dx2 + (S.Phi - 2*C.Phi + N.Phi)*one_over_dy2;
                laplace_U[i] = (W.T - 2*C.T + E.T)*one_over_dx2 +       (S.T - 2*C.T + N.T)*one_over_dy2;
            }
        );
        
        cuda_for(0, N, [=]SHARED(csize i){
            Real T =  U[i];
            Real Phi = F[i];

            Real grad_Phi_x = grad_F_x_2dx[i]*one_over_2dx;
            Real grad_Phi_y = grad_F_y_2dy[i]*one_over_2dy;
            Real grad_Phi_norm = custom_hypot(grad_Phi_x, grad_Phi_y);

            Real theta = custom_atan2(grad_Phi_y, grad_Phi_x);
            Real g_theta = (Real) 1 - S0*custom_cos(m0*theta + theta0);

            Real k0 = g_theta*f0(Phi)*k0_factor;
            Real k2 = grad_Phi_norm*k2_factor;
            Real k1 = g_theta*k1_factor;
            Real corr = 1 + k2*dt_L;

            Real right = Phi + dt/corr*((1-gamma)*k1*laplace_F[i] + k0 - k2*(T - Tm + dt*laplace_U[i]));
            Real factor = gamma/corr*k1; 

            A_F.anisotrophy[i] = (Real) factor;
            b_F[i] = (Real) right;
        });
        
        #else
        cuda_tiled_for_2D<1, 1, Bundled>(0, 0, nx, ny,
            [=]SHARED(csize x, csize y, csize nx, csize ny, csize rx, csize ry) -> Bundled{
                csize x_mod = x;
                csize y_mod = y;

                if(x_mod < 0)
                    x_mod += nx;
                else if(x_mod >= nx)
                    x_mod -= nx;

                if(y_mod < 0)
                    y_mod += ny;
                else if(y_mod >= ny)
                    y_mod -= ny;

                Real T = (Real) U[x_mod + y_mod*nx];
                Real Phi = (Real) F[x_mod + y_mod*ny];
                return Bundled{Phi, T};
            },
            [=]SHARED(csize x, csize y, csize tx, csize ty, csize tile_size_x, csize tile_size_y, Bundled* shared){
                Bundled C = shared[tx   + ty*tile_size_x];
                Bundled E = shared[tx+1 + ty*tile_size_x];
                Bundled W = shared[tx-1 + ty*tile_size_x];
                Bundled N = shared[tx   + (ty+1)*tile_size_x];
                Bundled S = shared[tx   + (ty-1)*tile_size_x];

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
        #endif
    }
    else
    {
        cuda_tiled_for_2D_modular<1, 1, TILED_FOR_PERIODIC_SMALL_R>(F, nx, ny,
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

    CUDA_TEST(cudaEventRecord(stop, 0));
    CUDA_TEST(cudaEventSynchronize(stop));

    float time = 0;
    CUDA_TEST(cudaEventElapsedTime(&time, start, stop));
    LOG_DEBUG("SOLVER", "Prepare kernel time %.2ems corrector_guess:%s tiled:%s", (double)time, 
        do_corrector_guess ? "true" : "false", 
        is_tiled ? "true" : "false");

    Conjugate_Gardient_Params solver_params = {0};
    solver_params.epsilon = (Real) 1.0e-12;
    solver_params.tolerance = (Real) params.Phi_tolerance;
    solver_params.max_iters = params.Phi_max_iters;
    solver_params.initial_value_or_null = F;

    //Solve A_F*F_next = b_F
    Conjugate_Gardient_Convergence F_converged = conjugate_gradient_solve(&A_F, F_next, b_F, N, anisotrophy_matrix_multiply, &solver_params);
    LOG_DEBUG("SOLVER", "%lli F %s in %i iters with error %e\n", (lli) iter, F_converged.converged ? "converged" : "diverged", F_converged.iters, (double)F_converged.error);

    //Calculate b_U
    cuda_for(0, nx*ny, [=]SHARED(csize i){
        Real T =  U_base[i];
        Real Phi = F[i];
        Real Phi_next = F_next[i];

        b_U[i] = (Real) (T + L*(Phi_next - Phi) + dt*(1-gamma)*T);
    });

    solver_params.tolerance = (Real) params.T_tolerance;
    solver_params.max_iters = params.T_max_iters;
    solver_params.initial_value_or_null = U;

    //Solve A_U*U_next = b_U
    Conjugate_Gardient_Convergence U_converged = conjugate_gradient_solve(&A_U, U_next, b_U, N, cross_matrix_static_multiply, &solver_params);
    LOG_DEBUG("SOLVER", "%lli U %s in %i iters with error %e\n", (lli) iter, U_converged.converged ? "converged" : "diverged", U_converged.iters, (double)U_converged.error);

    if(do_debug)
    {
        Real* AfF = solver->debug_maps.AfF;
        Real* AuU = solver->debug_maps.AuU;
        //Back test
        if(1)
        {
            anisotrophy_matrix_multiply(AfF, &A_F, F_next, N);
            cross_matrix_static_multiply(AuU, &A_U, U_next, N);

            Real back_error_F = vector_get_l2_dist(AfF, b_F, N);
            Real back_error_U = vector_get_l2_dist(AuU, b_U, N);

            Real back_error_F_max = vector_get_max_dist(AfF, b_F, N);
            Real back_error_U_max = vector_get_max_dist(AuU, b_U, N);

            LOG_DEBUG("SOLVER", "AVG | F:%e U:%e Epsilon:%e \n", (double) back_error_F, (double) back_error_U, (double) solver_params.tolerance*2);
            LOG_DEBUG("SOLVER", "MAX | F:%e U:%e Epsilon:%e \n", (double) back_error_F_max, (double) back_error_U_max, (double) solver_params.tolerance*2);
        }

        Real* grad_F = solver->debug_maps.grad_phi;
        Real* grad_U = solver->debug_maps.grad_T;
        cuda_for_2D(0, 0, nx, ny, [=]SHARED(int x, int y){
            Real T = *at_mod(U, x, y, nx, ny);
            Real Phi = *at_mod(F, x, y, nx, ny);

            Real Phi_U = *at_mod(F, x, y + 1, nx, ny);
            Real Phi_D = *at_mod(F, x, y - 1, nx, ny);
            Real Phi_R = *at_mod(F, x + 1, y, nx, ny);
            Real Phi_L = *at_mod(F, x - 1, y, nx, ny);

            Real T_U = *at_mod(U, x, y + 1, nx, ny);
            Real T_D = *at_mod(U, x, y - 1, nx, ny);
            Real T_R = *at_mod(U, x + 1, y, nx, ny);
            Real T_L = *at_mod(U, x - 1, y, nx, ny);

            Real grad_Phi_x = (Phi_R - Phi_L);
            Real grad_Phi_y = (Phi_U - Phi_D);
            Real grad_Phi_norm = custom_hypot(grad_Phi_x, grad_Phi_y);

            Real grad_T_x = (T_R - T_L);
            Real grad_T_y = (T_U - T_D);
            Real grad_T_norm = custom_hypot(grad_T_x, grad_T_y);
            
            grad_F[x + y*nx] = grad_Phi_norm;
            grad_U[x + y*nx] = grad_T_norm;
        });
    }

    cache_free(&tag);
}

void semi_implicit_solver_step(Semi_Implicit_Solver* solver, Semi_Implicit_State state, Semi_Implicit_State next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    semi_implicit_solver_step_based(solver, state.F, state.U, state.U, next_state, params, iter, do_debug);
}

void semi_implicit_solver_step_corrector(Semi_Implicit_Solver* solver, Semi_Implicit_State state, Semi_Implicit_State next_state, Allen_Cahn_Params params, size_t iter, Allen_Cahn_Stats* stats_or_null)
{
    Cache_Tag tag = cache_tag_make();
    int N = params.ny * params.nx;

    Explicit_State temp_state = {0};
    temp_state.F = cache_alloc(Real, N, &tag);
    temp_state.U = cache_alloc(Real, N, &tag);
    temp_state.nx = params.nx;
    temp_state.ny = params.ny;

    static int last_placement = 0;

    //Init states in such a way that the resutl will already be in 
    // next_state (thus no need to copy)
    Explicit_State steps[2] = {0};
    if(last_placement % 2 == 0)
    {
        steps[0] = next_state;
        steps[1] = temp_state;
    }
    else
    {
        steps[1] = next_state;
        steps[0] = temp_state;
    }

    Real* step_resiudal = cache_alloc(Real, N, &tag);
    
    Real step_residual_avg_error = 0;
    Real step_residual_max_error = 0;
    bool converged = false;
    USE_VARIABLE(converged);
    
    int k = 0;
    int max_iters = params.corrector_max_iters;
    if(params.do_corrector_loop == false)
        max_iters = 0;

    if(max_iters == 0 && params.do_stats_step_residual)
        max_iters = 1;

    bool do_debug = false;
    if(max_iters == 0)
        do_debug = params.do_debug;

    //Perform first step
    semi_implicit_solver_step(solver, state, steps[0], params, iter, do_debug);
    for(; k < max_iters; k++)
    {
        Explicit_State step_curr = steps[MOD(k, 2)];
        Explicit_State step_next = steps[MOD(k + 1, 2)];

        log_group();
        semi_implicit_solver_step_based(solver, state.F, step_curr.U, state.U, step_next, params, iter, false);

        cuda_for(0, N, [=]SHARED(int i){
            //@NOTE:fabs is broken and linking the wrong function which results in
            // illegal memory access ?!
            //@NOTE: abs mostly for debug view
            Real diff = step_curr.F[i] - step_next.F[i]; 
            step_resiudal[i] = diff >= 0 ? diff : -diff;
        });

        //@NOTE: no explicit sync!
        if(k < (int) STATIC_ARRAY_SIZE(solver->debug_maps.step_residuals))
            CUDA_DEBUG_TEST(cudaMemcpyAsync(solver->debug_maps.step_residuals[k], step_resiudal, (size_t)N*sizeof(Real), cudaMemcpyDeviceToDevice));

        step_residual_avg_error = vector_euclid_norm(step_resiudal, N);
        step_residual_max_error = vector_max(step_resiudal, N);
        if(k < (int) STATIC_ARRAY_SIZE(stats_or_null->L2_step_residuals) && stats_or_null)
        {
            stats_or_null->L2_step_residuals[k] = step_residual_avg_error;
            stats_or_null->Lmax_step_residuals[k] = step_residual_max_error;
            stats_or_null->step_residuals = k + 1;
        }

        LOG_DEBUG("SOLVER", "step residual loop: %i | avg: %e | max: %e | tolerance: %lf", k, 
            (double) step_residual_avg_error, (double) step_residual_max_error, params.corrector_tolerance);
        if(step_residual_avg_error < params.corrector_tolerance)
        {
            k ++;
            converged = true;
            break;
        }

        log_ungroup();
    }
    
    last_placement = k;

    //Debug only print
    step_residual_max_error = vector_max(step_resiudal, N);
    // if(max_iters > 0)
        // LOG_DEBUG("SOLVER", "step residual %s iters: %i | avg: %lf | max: %lf | tolerance: %lf", 
            // converged ? "converged" : "diverged", k + 1, 
            // (double) step_residual_avg_error, (double) step_residual_max_error, params.corrector_tolerance);

    //If the ended on step is already next_state dont copy anything
    Explicit_State final_step = steps[MOD(k, 2)];
    if(final_step.F != next_state.F)
    {
        CUDA_DEBUG_TEST(cudaMemcpyAsync(next_state.F, final_step.F, (size_t)N*sizeof(Real), cudaMemcpyDeviceToDevice));
        CUDA_DEBUG_TEST(cudaMemcpyAsync(next_state.U, final_step.U, (size_t)N*sizeof(Real), cudaMemcpyDeviceToDevice));
    }

    cache_free(&tag);
}

void semi_implicit_solver_get_maps(Semi_Implicit_Solver* solver, Semi_Implicit_State state, Sim_Map* maps, int map_count)
{
    int __map_i = 0;
    memset(maps, 0, sizeof maps[0] * (size_t)map_count);
    ASSIGN_MAP_NAMED(state.F, "Phi");            
    ASSIGN_MAP_NAMED(state.U, "T");            
    // ASSIGN_MAP_NAMED(solver->maps.b_F, "b_F");           
    // ASSIGN_MAP_NAMED(solver->debug_maps.AfF, "AfF");           
    // ASSIGN_MAP_NAMED(solver->maps.b_U, "b_U");           
    // ASSIGN_MAP_NAMED(solver->debug_maps.AuU, "AuU");           
    ASSIGN_MAP_NAMED(solver->debug_maps.grad_phi, "grad_phi");           
    ASSIGN_MAP_NAMED(solver->debug_maps.grad_T, "grad_T");           
    ASSIGN_MAP_NAMED(solver->maps.anisotrophy, "Anisotrophy");

    CHECK_BOUNDS(2, STATIC_ARRAY_SIZE(solver->debug_maps.step_residuals)); 
    ASSIGN_MAP_NAMED(solver->debug_maps.step_residuals[0], "step_residual1");          
    ASSIGN_MAP_NAMED(solver->debug_maps.step_residuals[1], "step_residual2");           
    ASSIGN_MAP_NAMED(solver->debug_maps.step_residuals[2], "step_residual3");           
}

struct Semi_Implicit_Coupled_Cross_Matrix {
    Anisotrophy_Matrix A_F; //A anisotrophy scaled cross matrix
    Real* B_U; //A changing diagonal 

    Cross_Matrix_Static A_U; //Static cross matrix
    Real B_F; //A single value diagonal

    int nx;
    int ny;
};

void semi_implicit_coupled_solver_resize(Semi_Implicit_Coupled_Solver* solver, int nx, int ny)
{
    if(solver->nx != nx || solver->ny != ny)
    {
        int N = ny*nx;
        int N_old = solver->ny*solver->nx;
        cuda_realloc_in_place((void**) &solver->b_C, 2*(size_t)N*sizeof(Real), 2*(size_t)N_old*sizeof(Real), REALLOC_ZERO);
        cuda_realloc_in_place((void**) &solver->aniso, (size_t)N*sizeof(Real), (size_t)N_old*sizeof(Real), REALLOC_ZERO);
        cuda_realloc_in_place((void**) &solver->B_U, (size_t)N*sizeof(Real), (size_t)N_old*sizeof(Real), REALLOC_ZERO);

        solver->nx = nx;
        solver->ny = ny;
    }
}

void semi_implicit_coupled_state_resize(Semi_Implicit_Coupled_State* state, int nx, int ny)
{
    if(state->nx != nx || state->ny != ny)
    {
        int N = ny*nx;
        int N_old = state->ny*state->nx;
        cuda_realloc_in_place((void**) &state->C, 2*(size_t)N*isizeof(Real), 2*(size_t)N_old*isizeof(Real), REALLOC_ZERO);
        state->nx = nx;
        state->ny = ny;
    }
}

void semi_implicit_coupled_matrix_multiply(Real* out, const void* A_, const Real* x, int vec_size)
{
    Semi_Implicit_Coupled_Cross_Matrix A = *(Semi_Implicit_Coupled_Cross_Matrix*)A_;

    int nx = A.nx;
    int ny = A.ny;
    int N = nx*ny;

    Real* F = (Real*) x;
    Real* U = (Real*) x + N; 

    Real* out_F = out;
    Real* out_U = out + N;

    //F equation
    cuda_for_2D(0, 0, nx, ny, [=]SHARED(int x, int y){
        int i = x + y*nx;
        Real s = A.A_F.anisotrophy[i];
        Real X = A.A_F.X*s;
        Real Y = A.A_F.Y*s;
        Real C = 1 + A.A_F.C_minus_one*s;

        Real F_val = F[i]*C;
        F_val += *at_mod(F, x + 1, y, nx, ny)*X;
        F_val += *at_mod(F, x - 1, y, nx, ny)*X;
        F_val += *at_mod(F, x , y + 1, nx, ny)*Y;
        F_val += *at_mod(F, x , y - 1, nx, ny)*Y;

        Real U_val = A.B_U[i]*U[i]; 

        out_F[i] = F_val + U_val;
    });

    //U equation
    cuda_for_2D(0, 0, nx, ny, [=]SHARED(int x, int y){
        int i = x + y*nx;

        Real U_val = U[i]*A.A_U.C;
        U_val += *at_mod(U, x + 1, y, nx, ny)*A.A_U.R;
        U_val += *at_mod(U, x - 1, y, nx, ny)*A.A_U.L;
        U_val += *at_mod(U, x , y + 1, nx, ny)*A.A_U.U;
        U_val += *at_mod(U, x , y - 1, nx, ny)*A.A_U.D;

        Real F_val = F[i]*A.B_F;
        out_U[i] = F_val + U_val;
    });
}

void semi_implicit_coupled_solver_step(Semi_Implicit_Coupled_Solver* solver, Semi_Implicit_Coupled_State state, Semi_Implicit_Coupled_State next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    Real dx = (Real) params.L0 / solver->nx;
    Real dy = (Real) params.L0 / solver->ny;

    int nx = solver->nx;
    int ny = solver->ny;
    int N = nx*ny;

    Real a = (Real) params.a;
    Real b = (Real) params.b;
    Real alpha = (Real) params.alpha;
    Real beta = (Real) params.beta;
    Real xi = (Real) params.xi;
    Real Tm = (Real) params.Tm;
    Real L = (Real) params.L; 
    Real dt = (Real) params.dt;
    Real S = (Real) params.S; 
    Real m0 = (Real) params.m0; 
    Real theta0 = (Real) params.theta0;

    Real* F = state.C;
    Real* U = state.C + N;
    
    Real* b_F = solver->b_C;
    Real* b_U = solver->b_C + N;

    Real* aniso = solver->aniso;
    Real* B_U = solver->B_U;

    //Prepare dynamic data
    cuda_for_2D(0, 0, nx, ny, [=]SHARED(int x, int y){
        Real T = U[x + y*nx];
        Real Phi = F[x + y*nx];

        Real Phi_U = *at_mod(F, x, y + 1, nx, ny);
        Real Phi_D = *at_mod(F, x, y - 1, nx, ny);
        Real Phi_R = *at_mod(F, x + 1, y, nx, ny);
        Real Phi_L = *at_mod(F, x - 1, y, nx, ny);

        Real grad_Phi_x = (Phi_R - Phi_L)/(2*dx);
        Real grad_Phi_y = (Phi_U - Phi_D)/(2*dy);
        Real grad_Phi_norm = custom_hypot(grad_Phi_x, grad_Phi_y);
 
        Real g_theta = 1;
        {
            Real theta = custom_atan2(grad_Phi_y, grad_Phi_x);
            g_theta = 1.0f - S*custom_cos(m0*theta + theta0);
        }

        // g_theta = 1;
        Real f_tilda = b*xi*xi*beta*grad_Phi_norm;
        Real f = g_theta*a*f0(Phi) - b*xi*xi*beta*(T - Tm)*grad_Phi_norm;
        Real T_factor = dt*f_tilda/(xi*xi*alpha);
        B_U[x+y*nx] = -T_factor;
        aniso[x+y*nx] = g_theta;
        b_F[x + y*nx] = Phi + dt*g_theta/(xi*xi*alpha)*f0(Phi) - T_factor*Tm;
        b_U[x + y*nx] = T + L*Phi;
    });

    Anisotrophy_Matrix A_F = {0};
    A_F.anisotrophy = aniso;
    A_F.C_minus_one = 2*dt/(alpha*dx*dx) + 2*dt/(alpha*dy*dy);
    A_F.X = -dt/(alpha*dx*dx);
    A_F.Y = -dt/(alpha*dy*dy);

    Cross_Matrix_Static A_U = {0};
    A_U.C = 1 + 2*dt/(dx*dx) + 2*dt/(dy*dy);
    A_U.R = -dt/(dx*dx);
    A_U.L = -dt/(dx*dx);
    A_U.U = -dt/(dy*dy);
    A_U.D = -dt/(dy*dy);

    #define ECHOF(x)    printf(#x ": %e\n", (double) (x))
    Real B_U_norm = vector_dot_product(B_U, B_U, N);
    B_U_norm = sqrt(B_U_norm / N);
    ECHOF(B_U_norm);
    ECHOF(A_F.C_minus_one + 1);
    ECHOF(A_F.X);
    ECHOF(A_U.C);
    ECHOF(A_U.U);

    Semi_Implicit_Coupled_Cross_Matrix A_C = {0};
    A_C.A_F = A_F;
    A_C.B_U = B_U;
    A_C.A_U = A_U;
    A_C.B_F = -L;
    A_C.nx = nx;
    A_C.ny = ny;

    Conjugate_Gardient_Params solver_params = {0};
    solver_params.epsilon = (Real) 1.0e-10;
    solver_params.tolerance = (Real) 1.0e-7;
    solver_params.max_iters = 200;
    solver_params.initial_value_or_null = state.C;

    Conjugate_Gardient_Convergence conv = conjugate_gradient_solve(&A_C, next_state.C, solver->b_C, 2*N, semi_implicit_coupled_matrix_multiply, &solver_params);
    printf("%lli C %s in %i iters with error %lf\n", (long long) iter, conv.converged ? "converged" : "diverged", conv.iters, (double) conv.error);
}

void semi_implicit_coupled_solver_get_maps(Semi_Implicit_Coupled_Solver* solver, Semi_Implicit_Coupled_State state, Sim_Map* maps, int map_count)
{
    int N = solver->nx*solver->ny;
    int __map_i = 0;
    memset(maps, 0, sizeof maps[0] * (size_t)map_count);
    ASSIGN_MAP_NAMED(state.C, "Phi");            
    ASSIGN_MAP_NAMED(state.C + N, "T");            
    ASSIGN_MAP_NAMED(solver->b_C, "b_F");           
    ASSIGN_MAP_NAMED(solver->b_C + N, "b_U");           
    ASSIGN_MAP_NAMED(solver->B_U, "B_U");           
    ASSIGN_MAP_NAMED(solver->aniso, "Anisotrophy");  
}

extern "C" void kernel_float_from_double(float* output, const double* input, size_t size)
{
    cuda_for(0, (int) size, [=]SHARED(int i){
        output[i] = (float) input[i];
    });
}
extern "C" void kernel_double_from_float(double* output, const float* input, size_t size)
{
    cuda_for(0, (int) size, [=]SHARED(int i){
        output[i] = (double) input[i];
    });
}

extern "C" void sim_modify(void* device_memory, void* host_memory, size_t size, Sim_Modify modify)
{
    if(modify == MODIFY_UPLOAD)
        CUDA_DEBUG_TEST(cudaMemcpy(device_memory, host_memory, size, cudaMemcpyHostToDevice));
    else
        CUDA_DEBUG_TEST(cudaMemcpy(host_memory, device_memory, size, cudaMemcpyDeviceToHost));
}

extern "C" void sim_modify_float(Real* device_memory, float* host_memory, size_t size, Sim_Modify modify)
{
    static float* static_device = NULL;
    static size_t static_size = 0;

    if(sizeof(Real) != sizeof(float))
    {
        if(static_size < size)
        {
            cuda_realloc_in_place((void**) &static_device, size*sizeof(float), static_size*sizeof(float), 0);
            static_size = size;
        }

        if(modify == MODIFY_UPLOAD)
        {
            //Upload: host -> static -> device
            CUDA_DEBUG_TEST(cudaMemcpy(static_device, host_memory, size*sizeof(float), cudaMemcpyHostToDevice));
            kernel_double_from_float((double*) (void*) device_memory, static_device, size);
        }
        else
        {
            //download: device -> static -> host
            kernel_float_from_double(static_device, (double*) (void*) device_memory, size);
            CUDA_DEBUG_TEST(cudaMemcpy(host_memory, static_device, size*sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    else
    {
        if(modify == MODIFY_UPLOAD)
            CUDA_DEBUG_TEST(cudaMemcpy(device_memory, host_memory, size*sizeof(float), cudaMemcpyHostToDevice));
        else
            CUDA_DEBUG_TEST(cudaMemcpy(host_memory, device_memory, size*sizeof(float), cudaMemcpyDeviceToHost));
    }
}


extern "C" void sim_solver_reinit(Sim_Solver* solver, Solver_Type type, int nx, int ny)
{
    if(solver->type != type && solver->type != SOLVER_TYPE_NONE)
        sim_solver_reinit(solver, solver->type, 0, 0);

    switch(type) {
        case SOLVER_TYPE_NONE: {
            ny = 0;
            nx = 0;
        } break;

        case SOLVER_TYPE_EXPLICIT: 
        case SOLVER_TYPE_EXPLICIT_RK4:
        case SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE: {
            explicit_solver_resize(&solver->expli, nx, ny);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT: {
            semi_implicit_solver_resize(&solver->impli, nx, ny);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: {
            semi_implicit_coupled_solver_resize(&solver->impli_coupled, nx, ny);
        } break;

        default: {
            assert(false);
        }
    };

    solver->type = type;
    solver->nx = nx;
    solver->ny = ny;
}

void sim_state_reinit(Sim_State* states, Solver_Type type, int nx, int ny)
{
    if(states->type != type && states->type != SOLVER_TYPE_NONE)
        sim_state_reinit(states, states->type, 0, 0);

    switch(type) {
        case SOLVER_TYPE_NONE: {
            ny = 0;
            nx = 0;
        } break;

        case SOLVER_TYPE_EXPLICIT: 
        case SOLVER_TYPE_EXPLICIT_RK4:
        case SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE: {
            explicit_state_resize(&states->expli, nx, ny);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT: {
            //For the moemnt these are the same
            explicit_state_resize(&states->impli, nx, ny);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: {
            semi_implicit_coupled_state_resize(&states->impli_coupled, nx, ny);
        } break;

        default: {
            assert(false);
        }
    };

    states->type = type;
    states->nx = nx;
    states->ny = ny;
}

extern "C" void sim_states_reinit(Sim_State* states, int state_count, Solver_Type type, int nx, int ny)
{
    for(int i = 0; i < state_count; i++)
        sim_state_reinit(&states[i], type, nx, ny);
}

extern "C" double sim_solver_step(Sim_Solver* solver, Sim_State* states, int states_count, int iter, Allen_Cahn_Params params, Allen_Cahn_Stats* stats_or_null)
{
    int required_history = solver_type_required_history(solver->type);
    const char* solver_name = solver_type_to_cstring(solver->type);
    
    bool okay = true;
    if(states_count < required_history)
    {
        okay = false;
        LOG_INFO("SOLVER", "Step: Not enough history for solver %s! Required %i. Got %i", solver_name, states_count, required_history);
    }
    else
    {
        for(int i = 0; i < states_count; i++)
        {
            if(states[i].type != solver->type)
            {
                LOG_INFO("SOLVER", "Step: state[%i] is of bad type %s. Expected %s", solver_type_to_cstring(states[i].type), solver_name);
                okay = false;
            }
        }
    }
    
    double step_by = 0;
    if(okay)
    {
        step_by = params.dt;
        ASSERT(states_count > 0);
        Sim_State state = states[MOD(iter, states_count)];
        Sim_State next_state = states[MOD(iter + 1, states_count)];
        switch(solver->type) {
            case SOLVER_TYPE_NONE: {
                LOG_INFO("SOLVER", "Step: stepping as solver type none has no effect");
                // nothing
            } break;

            case SOLVER_TYPE_EXPLICIT: 
            case SOLVER_TYPE_EXPLICIT_RK4: 
            case SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE: {
                step_by = explicit_solver_choose_and_copute_step_residual(solver->type, &solver->expli, state.expli, &next_state.expli, params, (size_t) iter, stats_or_null);
            } break;

            case SOLVER_TYPE_SEMI_IMPLICIT: {
                semi_implicit_solver_step_corrector(&solver->impli, state.impli, next_state.impli, params, (size_t) iter, stats_or_null);
            } break;

            case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: {
                semi_implicit_coupled_solver_step(&solver->impli_coupled, state.impli_coupled, next_state.impli_coupled, params, (size_t) iter, params.do_debug);
            } break;

            default: assert(false);
        };
    }

    CUDA_DEBUG_TEST(cudaDeviceSynchronize());
    return step_by;
}

extern "C" void sim_solver_get_maps(Sim_Solver* solver, Sim_State* states, int states_count, int iter, Sim_Map* maps, int map_count)
{
    if(states_count <= 0 || map_count <= 0)
        return;

    Sim_State state = states[MOD(iter, states_count)];
    switch(solver->type) {
        case SOLVER_TYPE_NONE: {
            //none
        } break;

        case SOLVER_TYPE_EXPLICIT: 
        case SOLVER_TYPE_EXPLICIT_RK4:
        case SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE: {
            explicit_solver_get_maps(&solver->expli, state.expli, maps, map_count);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT: {
            semi_implicit_solver_get_maps(&solver->impli, state.impli, maps, map_count);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: {
            semi_implicit_coupled_solver_get_maps(&solver->impli_coupled, state.impli_coupled, maps, map_count);
        } break;

        default: assert(false);
    };
}


#else

extern "C" void sim_solver_reinit(Sim_Solver* solver, Solver_Type type, int nx, int ny) {}
extern "C" void sim_states_reinit(Sim_State* states, int state_count, Solver_Type type, int nx, int ny) {}
extern "C" void sim_solver_get_maps(Sim_Solver* solver, Sim_State* states, int states_count, int iter, Sim_Map* maps, int map_count) {}
extern "C" double sim_solver_step(Sim_Solver* solver, Sim_State* states, int states_count, int iter, Allen_Cahn_Params params, Allen_Cahn_Stats* stats_or_null) {return 0;}

extern "C" void sim_modify(void* device_memory, void* host_memory, size_t size, Sim_Modify modify) {}
extern "C" void sim_modify_float(Real* device_memory, float* host_memory, size_t size, Sim_Modify modify) {}
extern "C" void sim_modify_double(Real* device_memory, double* host_memory, size_t size, Sim_Modify modify) {}
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