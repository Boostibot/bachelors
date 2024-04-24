#include "kernel.h"
#include "cuda_util.cuh"

#define PI          ((Real) 3.14159265359)
#define TAU         (2*PI)
#define ECHOF(x)    printf(#x ": " REAL_FMT "\n", (x))

enum Cuda_For_Flags {
    CUDA_FOR_NONE = 0,
    CUDA_FOR_ASYNC = 1,
};

template <typename Function>
__global__ void _kernel_cuda_for_each(int from, int item_count, Function func)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < item_count; i += blockDim.x * gridDim.x) 
        func(from + i);
}

template <typename Function>
void cuda_for(int from, int to, Function func, int flags = 0)
{
    Cuda_Info info = cuda_one_time_setup();
    dim3 bs(64, 1);
    dim3 grid(info.prop.multiProcessorCount, 1);

    _kernel_cuda_for_each<<<grid, bs>>>(from, to-from, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
    if((flags & CUDA_FOR_ASYNC) == 0)
        CUDA_DEBUG_TEST(cudaDeviceSynchronize());
}


template <typename Function>
__global__ void _kernel_cuda_for_each_2D(int from_x, int x_size, int from_y, int y_size, Function func)
{
    //@TODO: Whats the optimal loop order? First x or y?
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < y_size; y += blockDim.y * gridDim.y) 
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < x_size; x += blockDim.x * gridDim.x) 
            func(x + from_x, y + from_y);
}

template <typename Function>
void cuda_for_2D(int from_x, int from_y, int to_x, int to_y, Function func, int flags = 0)
{
    Cuda_Info info = cuda_one_time_setup();
    dim3 bs(64, 1);
    dim3 grid(info.prop.multiProcessorCount, 1);
    _kernel_cuda_for_each_2D<<<grid, bs>>>(from_x, to_x-from_x, from_y, to_y-from_y, (Function&&) func);
    CUDA_DEBUG_TEST(cudaGetLastError());
    if((flags & CUDA_FOR_ASYNC) == 0)
        CUDA_DEBUG_TEST(cudaDeviceSynchronize());
}

struct Random_Map_State {
    Real* map;
    uint32_t* state;
    uint64_t seed;
    uint64_t N; 
    uint64_t iteration;
};

SHARED uint32_t pcg_hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

SHARED uint32_t pcg_rand(uint32_t* rng_state)
{
    uint32_t state = *rng_state;
    *rng_state = *rng_state * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

SHARED Real uint32_to_norm_real(uint32_t val)
{
    Real x = (Real) val / (Real) UINT32_MAX;
    return x;
}

void random_map_seed(Random_Map_State* state, uint64_t seed)
{
    uint32_t* rand = state->state;
    Real* rand_real = state->map;
    state->iteration = 0;
    cuda_for(0, (int)state->N, [=]SHARED(int i) {
        uint32_t hashed_index = pcg_hash((uint32_t) i);
        uint32_t fold = (uint32_t) seed ^ (uint32_t) (seed >> 32);

        uint32_t hashed_together = pcg_hash(hashed_index ^ fold);
        rand[i] = hashed_together;
        rand_real[i] = uint32_to_norm_real(hashed_together);
    });
}

void random_map_next(Random_Map_State* state)
{
    uint32_t* rand = state->state;
    Real* rand_real = state->map;
    state->iteration += 1;
    cuda_for(0, (int)state->N, [=]SHARED(int i) {
        uint32_t val = pcg_rand(&rand[i]); 
        rand_real[i] = uint32_to_norm_real(val);
    });
}

#if 0
//Will hand write my own version later. For now we trust in thrust *cymbal*
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>

Real vector_dot_product(const Real *a, const Real *b, int n)
{
  // wrap raw pointers to device memory with device_ptr
  thrust::device_ptr<const Real> d_a(a);
  thrust::device_ptr<const Real> d_b(b);

  // inner_product implements a mathematical dot product
  return thrust::inner_product(d_a, d_a + n, d_b, 0.0);
}

Real vector_max(const Real *a, int N)
{
    thrust::device_ptr<const Real> d_a(a);
    return *(thrust::max_element(d_a, d_a + N));
}


SHARED Real* at_mod(Real* map, int x, int y, int n, int m)
{
    #define AT_MOD_MODE 1
    #if AT_MOD_MODE == 0
        //95 ms
        int x_mod = MOD(x, m);
        int y_mod = MOD(y, n);
    #elif AT_MOD_MODE == 1
        //63 ms
        //@NOTE: this onluy works for x, y maximaly m, n respectively otuside of their proper range.
        // In our application this is enough.
        //@NOTE: this seems to be the fastest mode
        int x_mod = x;
        if(x_mod < 0)
            x_mod += m;
        else if(x_mod >= m)
            x_mod -= m;

        int y_mod = y;
        if(y_mod < 0)
            y_mod += n;
        else if(y_mod >= n)
            y_mod -= n;

    #elif AT_MOD_MODE == 2
        //85 ms
        int x_mod = (x + m) % m;
        int y_mod = (y + n) % n;
    #endif
    return &map[x_mod + y_mod*m];
}

void explicit_solver_resize(Explicit_Solver* solver, int n, int m)
{
    size_t N = (size_t)m*(size_t)n;
    size_t N_old = (size_t)solver->m*(size_t)solver->n;
    if(solver->m != m || solver->n != n)
    {
        //Big evil programming practices because we are cool and we know
        // what we are doing and dont care much about what others have to
        // say
        Real* debug_maps = (Real*) (void*) &solver->debug_maps;
        for(int i = 0; i < sizeof(solver->debug_maps) / sizeof(Real); i++)
            cuda_realloc_in_place((void**) &debug_maps[i], N*sizeof(Real), N_old*sizeof(Real), REALLOC_ZERO);

        solver->m = m;
        solver->n = n;
    }
}

void explicit_state_resize(Explicit_State* state, int n, int m)
{
    size_t N = (size_t)m*(size_t)n;
    size_t N_old = (size_t)state->m*(size_t)state->n;
    if(state->m != m || state->n != n)
    {
        cuda_realloc_in_place((void**) &state->F, N*sizeof(Real), N_old*sizeof(Real), REALLOC_ZERO);
        cuda_realloc_in_place((void**) &state->U, N*sizeof(Real), N_old*sizeof(Real), REALLOC_ZERO);

        state->m = m;
        state->n = n;
    }
}

union Explicit_Solve_Stencil {
    struct {
        Real Phi;
        Real Phi_U;
        Real Phi_D;
        Real Phi_L;
        Real Phi_R;
        Real T;
        Real T_U;
        Real T_D;
        Real T_L;
        Real T_R;
    };
    Real vals[10];
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

#if 0

SHARED Explicit_Solve_Result allen_cahn_explicit_solve(Explicit_Solve_Stencil input, Explicit_Solve_Stencil base, Allen_Cahn_Params params, Explicit_Solve_Debug* debug_or_null)
{
    Real dx = (Real) params.L0 / params.m;
    Real dy = (Real) params.L0 / params.n;

    Real a = params.a;
    Real b = params.b;
    Real alpha = params.alpha;
    Real beta = params.beta;
    Real xi = params.xi;
    Real Tm = params.Tm;
    Real L = params.L; //Latent heat, not L0 (sym size) ! 
    Real dt = params.dt;
    Real S = params.S; //anisotrophy strength
    Real m0 = params.m0; //anisotrophy frequency (?)
    Real theta0 = params.theta0;

    Real Phi = input.Phi;
    Real Phi_U = input.Phi_U;
    Real Phi_D = input.Phi_D;
    Real Phi_L = input.Phi_L;
    Real Phi_R = input.Phi_R;

    Real T = input.T;
    Real T_U = input.T_U;
    Real T_D = input.T_D;
    Real T_L = input.T_L;
    Real T_R = input.T_R;

    Real grad_Phi_x = (Phi_R - Phi_L)/(2*dx);
    Real grad_Phi_y = (Phi_U - Phi_D)/(2*dy);

    Real theta = atan2(grad_Phi_y, grad_Phi_x);;
    Real g_theta = 1.0f - S*cosf(m0*theta + theta0);

    Real laplace_T = (T_L - 2*T + T_R)/(dx*dx) + (T_D - 2*T + T_U)/(dy*dy);
    Real laplace_T_base = (base.T_L - 2*base.T + base.T_R)/(dx*dx) + (base.T_D - 2*base.T + base.T_U)/(dy*dy);
    Real laplace_Phi = (Phi_L - 2*Phi + Phi_R)/(dx*dx) + (Phi_D - 2*Phi + Phi_U)/(dy*dy);

    Real f0_tilda = g_theta*a/(xi*xi * alpha)*f0(Phi);
    Real f1_tilda = b*beta/alpha*hypotf(grad_Phi_x, grad_Phi_y);
    Real f2_tilda = g_theta/alpha;
    Real corr = 1 + f1_tilda*dt*L;

    Real dt_Phi = 0;
    if(params.do_corrector_guess)
        dt_Phi = (f2_tilda*laplace_Phi + f0_tilda - f1_tilda*(T - Tm + dt*laplace_T))/corr;
    else
        dt_Phi = f2_tilda*laplace_Phi + f0_tilda - f1_tilda*(T - Tm);
    Real dt_T = laplace_T_base + L*dt_Phi; 

    if(debug_or_null)
    {
        debug_or_null->grad_Phi = hypotf(Phi_R - Phi_L, Phi_U - Phi_D);
        debug_or_null->grad_T = hypotf(T_R - T_L, T_U - T_D);
        if(params.do_corrector_guess)
            debug_or_null->reaction_term = (f0_tilda + f1_tilda*(T - Tm + dt*laplace_T)) / corr;
        else
            debug_or_null->reaction_term = f0_tilda - f1_tilda*(T - Tm);
        debug_or_null->g_theta = g_theta;
        debug_or_null->theta = theta;
    }

    Explicit_Solve_Result out = {dt_Phi, dt_T};
    return out;
}

#else
SHARED Explicit_Solve_Result allen_cahn_explicit_solve(Explicit_Solve_Stencil input, Allen_Cahn_Params params, Explicit_Solve_Debug* debug_or_null)
{
    Real dx = (Real) params.L0 / params.m;
    Real dy = (Real) params.L0 / params.n;

    //@NOTE: dont you wish we had odin lang using in C?
    Real a = params.a;
    Real b = params.b;
    Real alpha = params.alpha;
    Real beta = params.beta;
    Real xi = params.xi;
    Real Tm = params.Tm;
    Real L = params.L; //Latent heat, not L0 (sym size) ! 
    Real dt = params.dt;
    Real S = params.S; //anisotrophy strength
    Real m0 = params.m0; //anisotrophy frequency (?)
    Real theta0 = params.theta0;

    Real Phi = input.Phi;
    Real Phi_U = input.Phi_U;
    Real Phi_D = input.Phi_D;
    Real Phi_L = input.Phi_L;
    Real Phi_R = input.Phi_R;

    Real T = input.T;
    Real T_U = input.T_U;
    Real T_D = input.T_D;
    Real T_L = input.T_L;
    Real T_R = input.T_R;

    Real grad_Phi_x = (Phi_R - Phi_L)/(2*dx);
    Real grad_Phi_y = (Phi_U - Phi_D)/(2*dy);
    Real grad_Phi_norm = hypotf(grad_Phi_x, grad_Phi_y);

    Real theta = atan2(grad_Phi_y, grad_Phi_x);;
    Real g_theta = 1.0f - S*cosf(m0*theta + theta0);

    Real laplace_T = (T_L - 2*T + T_R)/(dx*dx) + (T_D - 2*T + T_U)/(dy*dy);
    Real laplace_Phi = (Phi_L - 2*Phi + Phi_R)/(dx*dx) + (Phi_D - 2*Phi + Phi_U)/(dy*dy);

    Real k0 = g_theta*a*f0(Phi)/(xi*xi * alpha);
    Real k1 = g_theta/alpha;
    Real k2 = b*beta*grad_Phi_norm/alpha;
    Real corr = 1 + k2*dt*L;

    Real dt_Phi = 0;
    if(params.do_corrector_guess)
        dt_Phi = (k1*laplace_Phi + k0 - k2*(T - Tm + dt*laplace_T))/corr;
    else
        dt_Phi = k1*laplace_Phi + k0 - k2*(T - Tm);

    Real dt_T = laplace_T + L*dt_Phi; 
    if(debug_or_null)
    {
        debug_or_null->grad_Phi = hypotf(Phi_R - Phi_L, Phi_U - Phi_D);
        debug_or_null->grad_T = hypotf(T_R - T_L, T_U - T_D);
        if(params.do_corrector_guess)
            debug_or_null->reaction_term = (k0 + k2*(T - Tm + dt*laplace_T)) / corr;
        else
            debug_or_null->reaction_term = k0 - k2*(T - Tm);
        debug_or_null->g_theta = g_theta;
        debug_or_null->theta = theta;
    }

    Explicit_Solve_Result out = {dt_Phi, dt_T};
    return out;
}

#endif

SHARED Explicit_Solve_Stencil explicit_solve_stencil_mod(const Real* Phi, const Real* T, int x, int y, int n, int m)
{
    Explicit_Solve_Stencil solve = {0};
    solve.T = T[x + y*m];
    solve.Phi = Phi[x + y*m];

    solve.Phi_U = *at_mod((Real*) Phi, x, y + 1, n, m);
    solve.Phi_D = *at_mod((Real*) Phi, x, y - 1, n, m);
    solve.Phi_R = *at_mod((Real*) Phi, x + 1, y, n, m);
    solve.Phi_L = *at_mod((Real*) Phi, x - 1, y, n, m);

    solve.T_U = *at_mod((Real*) T, x, y + 1, n, m);
    solve.T_D = *at_mod((Real*) T, x, y - 1, n, m);
    solve.T_R = *at_mod((Real*) T, x + 1, y, n, m);
    solve.T_L = *at_mod((Real*) T, x - 1, y, n, m);

    return solve;
}

extern "C" void explicit_solver_newton_step(Explicit_Solver* solver, Explicit_State state, Explicit_State next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    Real* Phi_next = next_state.F;
    Real* Phi = state.F;
    
    Real* T_next = next_state.U;
    Real* T = state.U;

    Explicit_Solver expli = *solver;
    int m = params.m;
    int n = params.n;
    if(do_debug)
    {
        cuda_for_2D(0, 0, params.m, params.n, [=]SHARED(int x, int y){
            Explicit_Solve_Debug debug = {0};
            Explicit_Solve_Stencil input = explicit_solve_stencil_mod(Phi, T, x, y, n, m);
            Explicit_Solve_Result solved = allen_cahn_explicit_solve(input, params, &debug);

            //Newton update
            Phi_next[x + y*m] = input.Phi + solved.dt_Phi*params.dt;
            T_next[x + y*m] = input.T + solved.dt_T*params.dt;

            expli.debug_maps.grad_phi[x + y*m] = debug.grad_Phi;
            expli.debug_maps.grad_T[x + y*m] = debug.grad_T;
            expli.debug_maps.reaction[x + y*m] = debug.reaction_term;
            expli.debug_maps.aniso_factor[x + y*m] = debug.g_theta;
        });
    }
    else
    {
        cuda_for_2D(0, 0, params.m, params.n, [=]SHARED(int x, int y){
            Explicit_Solve_Stencil input = explicit_solve_stencil_mod(Phi, T, x, y, n, m);
            Explicit_Solve_Result solved = allen_cahn_explicit_solve(input, params, NULL);

            Phi_next[x + y*m] = input.Phi + solved.dt_Phi*params.dt;
            T_next[x + y*m] = input.T + solved.dt_T*params.dt;
        });
    }
}

struct Explicit_Blend_State {
    Real weight;
    Explicit_State state;
};

template<typename ... States>
void explicit_solver_solve_lin_combination(Explicit_State* out, Allen_Cahn_Params params, States... state_args)
{
    int m = params.m;
    int n = params.n;
    Real* out_F = out->F;
    Real* out_U = out->U;

    constexpr int state_count = sizeof...(state_args);
    Explicit_Blend_State states[state_count] = {state_args...};

    cuda_for_2D(0, 0, params.m, params.n, [=]SHARED(int x, int y){
        Explicit_Solve_Stencil blend = {0};
        for(int i = 0; i < state_count; i++)
        {
            Explicit_Solve_Stencil input = explicit_solve_stencil_mod(states[i].state.F, states[i].state.U, x, y, n, m);
            Real weight = states[i].weight;
            for(int i = 0; i < STATIC_ARRAY_SIZE(blend.vals); i++)
                blend.vals[i] += weight*input.vals[i];
        }

        Explicit_Solve_Result solved = allen_cahn_explicit_solve(blend, params, NULL);

        out_F[x + y*m] = solved.dt_Phi;
        out_U[x + y*m] = solved.dt_T;
    });
}
void explicit_solver_debug_step(Explicit_Solver* solver, Explicit_State state, Allen_Cahn_Params params)
{
    int m = params.m;
    int n = params.n;
    Real* F = state.F;
    Real* U = state.U;
    Real* grad_F = solver->debug_maps.grad_phi;
    Real* grad_U = solver->debug_maps.grad_T;
    Real* aniso = solver->debug_maps.aniso_factor;
    cuda_for_2D(0, 0, m, n, [=]SHARED(int x, int y){
        Real T = *at_mod(U, x, y, n, m);
        Real Phi = *at_mod(F, x, y, n, m);

        Real Phi_U = *at_mod(F, x, y + 1, n, m);
        Real Phi_D = *at_mod(F, x, y - 1, n, m);
        Real Phi_R = *at_mod(F, x + 1, y, n, m);
        Real Phi_L = *at_mod(F, x - 1, y, n, m);

        Real T_U = *at_mod(U, x, y + 1, n, m);
        Real T_D = *at_mod(U, x, y - 1, n, m);
        Real T_R = *at_mod(U, x + 1, y, n, m);
        Real T_L = *at_mod(U, x - 1, y, n, m);

        Real grad_Phi_x = (Phi_R - Phi_L);
        Real grad_Phi_y = (Phi_U - Phi_D);
        Real grad_Phi_norm = hypotf(grad_Phi_x, grad_Phi_y);

        Real grad_T_x = (T_R - T_L);
        Real grad_T_y = (T_U - T_D);
        Real grad_T_norm = hypotf(grad_T_x, grad_T_y);
        
        Real theta = atan2(grad_Phi_y, grad_Phi_x);
        Real g_theta = 1.0f - params.S*cosf(params.m0*theta + params.theta0);

        grad_F[x + y*m] = grad_Phi_norm;
        grad_U[x + y*m] = grad_T_norm;
        aniso[x + y*m] = g_theta;
    });

}

void explicit_solver_rk4_step(Explicit_Solver* solver, Explicit_State state, Explicit_State* next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    Cache_Tag tag = cache_tag_make();
    int N = params.n * params.m;

    Explicit_State steps[4] = {0};
    for(int i = 0; i < STATIC_ARRAY_SIZE(steps); i++)
    {
        steps[i].F = cache_alloc(Real, N, &tag);
        steps[i].U = cache_alloc(Real, N, &tag);
        steps[i].m = params.m;
        steps[i].n = params.n;
    }

    Explicit_State k1 = steps[0];
    Explicit_State k2 = steps[1];
    Explicit_State k3 = steps[2];
    Explicit_State k4 = steps[3];

    Real dt = params.dt;
    using W = Explicit_Blend_State;
    explicit_solver_solve_lin_combination(&k1, params, W{1, state});
    explicit_solver_solve_lin_combination(&k2, params, W{1, state}, W{dt * 0.5, k1});
    explicit_solver_solve_lin_combination(&k3, params, W{1, state}, W{dt * 0.5, k2});
    explicit_solver_solve_lin_combination(&k4, params, W{1, state}, W{dt * 1, k3});

    Real* out_F = next_state->F;
    Real* out_U = next_state->U;
    cuda_for(0, params.n*params.m, [=]SHARED(int i){
        out_F[i] =  state.F[i] + dt/6*(k1.F[i] + 2*k2.F[i] + 2*k3.F[i] + k4.F[i]);
        out_U[i] =  state.U[i] + dt/6*(k1.U[i] + 2*k2.U[i] + 2*k3.U[i] + k4.U[i]);
    }, CUDA_FOR_ASYNC);

    if(do_debug)
        explicit_solver_debug_step(solver, state, params);

    cache_free(&tag);
    CUDA_DEBUG_TEST(cudaDeviceSynchronize());
}

double explicit_solver_rk4_adaptive_step(Explicit_Solver* solver, Explicit_State state, Explicit_State* next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    Cache_Tag tag = cache_tag_make();
    int N = params.n * params.m;

    static Real _initial_step = 0;
    if(iter == 0)
        _initial_step = params.dt;

    Real tau = _initial_step;
    Explicit_State steps[5] = {0};
    for(int i = 0; i < STATIC_ARRAY_SIZE(steps); i++)
    {
        steps[i].F = cache_alloc(Real, N, &tag);
        steps[i].U = cache_alloc(Real, N, &tag);
        steps[i].m = params.m;
        steps[i].n = params.n;
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

        cuda_for(0, params.n*params.m, [=]SHARED(int i){
            Real F = 0.2*k1.F[i] - 0.9*k3.F[i] + 0.8*k4.F[i] - 0.1*k5.F[i];
            Real U = 0.2*k1.U[i] - 0.9*k3.U[i] + 0.8*k4.U[i] - 0.1*k5.U[i];

            Epsilon_F[i] = F >= 0 ? F : -F;
            Epsilon_U[i] = U >= 0 ? U : -U;
        });

        epsilon_F = vector_max(Epsilon_F, N);
        epsilon_U = vector_max(Epsilon_U, N);

        if(epsilon_F < params.Phi_tolerance && epsilon_U < params.T_tolerance)
            converged = true;

        Real epsilon = MAX(epsilon_F + epsilon_U, 1e-8);
        Real delta = MAX(params.Phi_tolerance + params.T_tolerance, 1e-8);
        used_tau = tau;
        tau = pow(delta / epsilon, 0.2)*4/5*tau;
    }

    Real* next_F = next_state->F;
    Real* next_U = next_state->U;
    cuda_for(0, params.n*params.m, [=]SHARED(int i){
        next_F[i] = state.F[i] + used_tau*(1.0/6*(k1.F[i] + k5.F[i]) + 2.0/3*k4.F[i]);
        next_U[i] = state.U[i] + used_tau*(1.0/6*(k1.U[i] + k5.U[i]) + 2.0/3*k4.U[i]);
    });

    LOG("SOLVER", converged ? LOG_DEBUG : LOG_WARN, "rk4-adaptive %s in %i iters with error F:%lf | U:%lf | tau:%le", converged ? "converged" : "diverged", i, epsilon_F, epsilon_U, used_tau);
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
        explicit_solver_newton_step(solver, state, *next_state, params, iter, params.do_debug);
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

Real vector_get_l2_dist(const Real* a, const Real* b, int N)
{
    Cache_Tag tag = cache_tag_make();
    Real* temp = cache_alloc(Real, N, &tag);
    cuda_for(0, N, [=]SHARED(int i){
        temp[i] = a[i] - b[i];
    });

    Real temp_dot_temp = vector_dot_product(temp, temp, N);
    Real error = sqrt(temp_dot_temp/N);
    cache_free(&tag);
    return error;
}

Real vector_get_max_dist(const Real* a, const Real* b, int N)
{
    Cache_Tag tag = cache_tag_make();
    Real* temp = cache_alloc(Real, N, &tag);
    cuda_for(0, N, [=]SHARED(int i){
        temp[i] = a[i] - b[i];
    });

    Real temp_dot_temp = vector_max(temp, N);
    Real error = sqrt(temp_dot_temp/N);
    cache_free(&tag);
    return error;
}


Real vector_euclid_norm(Real* vector, int N)
{
    Real dot = vector_dot_product(vector, vector, N);
    return sqrt(dot / N);
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
        int N = params.n * params.m;
        corrected_next.F = cache_alloc(Real, N, &tag);
        corrected_next.U = cache_alloc(Real, N, &tag);
        corrected_next.m = params.m;
        corrected_next.n = params.n;

        Allen_Cahn_Params changed_params = params;
        changed_params.do_debug = false;
        changed_params.do_stats = false;
        changed_params.do_stats_step_residual = false;

        explicit_solver_choose(type, solver, combined_state, &corrected_next, changed_params, iter);
        stats_or_null->L2_step_residuals[0] = vector_get_l2_dist(corrected_next.F, next_state->F, N);
        stats_or_null->Lmax_step_residuals[0] = vector_get_max_dist(corrected_next.F, next_state->F, N);
        stats_or_null->step_residuals = 1;

        LOG_DEBUG("SOLVER", "%lli step residual | avg: %e | max: %e", (long long) iter, stats_or_null->L2_step_residuals[0], stats_or_null->Lmax_step_residuals[0]);

        cache_free(&tag);
    }

    return advance_by;
}

void explicit_solver_get_maps(Explicit_Solver* solver, Explicit_State state, Sim_Map* maps, int map_count)
{
    int __map_i = 0;
    memset(maps, 0, sizeof maps[0] * map_count);

    #define ASSIGN_MAP_NAMED(var_ptr, var_name) \
        if(__map_i < map_count) \
        { \
            maps[__map_i].data = var_ptr; \
            maps[__map_i].name = var_name; \
            maps[__map_i].m = solver->m; \
            maps[__map_i].n = solver->n; \
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
}

struct Cross_Matrix_Static {
    Real C;
    Real U;
    Real D;
    Real L;
    Real R;

    int m;
    int n;
};

struct Cross_Matrix {
    Real* C;
    Real* U;
    Real* D;
    Real* L;
    Real* R;

    int m;
    int n;
};

struct Anisotrophy_Matrix {
    Real* anisotrophy;
    Real X;
    Real Y;
    Real C_minus_one;

    int m;
    int n;
};

#if 0
void cross_matrix_static_multiply(Real* out, const void* _A, const Real* x, int N)
{
    Cross_Matrix_Static A = *(Cross_Matrix_Static*)_A;
    int m = A.m;
    cuda_for(0, N, [=]SHARED(int i){
        Real val = 0;
        val += x[i]*A.C;
        if(i+1 < N)  val += x[i+1]*A.R;
        if(i-1 >= 0) val += x[i-1]*A.L;
        if(i+m < N)  val += x[i+m]*A.U;
        if(i-m >= 0) val += x[i-m]*A.D;

        out[i] = val;
    });
}
#else
void cross_matrix_static_multiply(Real* out, const void* _A, const Real* vec, int N)
{
    Cross_Matrix_Static A = *(Cross_Matrix_Static*)_A;
    int m = A.m;
    int n = A.n;
    cuda_for_2D(0, 0, m, n, [=]SHARED(int x, int y){
        int i = x + y*m;
        Real val = vec[i]*A.C;
        val += *at_mod((Real*) vec, x+1, y, n, m)*A.R;
        val += *at_mod((Real*) vec, x-1, y, n, m)*A.L;
        val += *at_mod((Real*) vec, x, y+1, n, m)*A.U;
        val += *at_mod((Real*) vec, x, y-1, n, m)*A.D;
        out[i] = val;
    });
}
#endif

void cross_matrix_multiply(Real* out, const void* _A, const Real* x, int N)
{
    Cross_Matrix A = *(Cross_Matrix*)_A;
    int m = A.m;
    cuda_for(0, N, [=]SHARED(int i){
        Real val = 0;
        val += x[i]*A.C[i];
        if(i+1 < N)  val += x[i+1]*A.R[i];
        if(i-1 >= 0) val += x[i-1]*A.L[i];
        if(i+m < N)  val += x[i+m]*A.U[i];
        if(i-m >= 0) val += x[i-m]*A.D[i];

        out[i] = val;
    });
}

#if 0
void anisotrophy_matrix_multiply(Real* out, const void* _A, const Real* x, int N)
{
    Anisotrophy_Matrix A = * (Anisotrophy_Matrix*)_A;
    int m = A.m;
    cuda_for(0, N, [=]SHARED(int i){
        Real s = A.anisotrophy[i];
        Real X = A.X*s;
        Real Y = A.Y*s;
        Real C = 1 + A.C_minus_one*s;

        Real val = 0;
        val += x[i]*C;
        if(i+1 < N)  val += x[i+1]*X;
        if(i-1 >= 0) val += x[i-1]*X;
        if(i+m < N)  val += x[i+m]*Y;
        if(i-m >= 0) val += x[i-m]*Y;

        out[i] = val;
    });
}

#else
void anisotrophy_matrix_multiply(Real* out, const void* _A, const Real* vec, int N)
{
    Anisotrophy_Matrix A = * (Anisotrophy_Matrix*)_A;
    int m = A.m;
    int n = A.n;
    cuda_for_2D(0, 0, m, n, [=]SHARED(int x, int y){
        int i = x + y*m;
        Real s = A.anisotrophy[i];
        Real X = A.X*s;
        Real Y = A.Y*s;
        Real C = 1 + A.C_minus_one*s;

        Real val = vec[i]*C;
        val += *at_mod((Real*) vec, x+1, y, n, m)*X;
        val += *at_mod((Real*) vec, x-1, y, n, m)*X;
        val += *at_mod((Real*) vec, x, y+1, n, m)*Y;
        val += *at_mod((Real*) vec, x, y-1, n, m)*Y;
        out[i] = val;
    });
}
#endif

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

    if(params.initial_value_or_null)
    {
        CUDA_DEBUG_TEST(cudaMemcpyAsync(x, params.initial_value_or_null, sizeof(Real)*N, cudaMemcpyDeviceToDevice));
        matrix_mul_func(Ap, A, params.initial_value_or_null, N);
        cuda_for(0, N, [=]SHARED(int i){
            r[i] = b[i] - Ap[i];
            p[i] = r[i];
        });

        r_dot_r = vector_dot_product(r, r, N);
    }
    else
    {
        CUDA_DEBUG_TEST(cudaMemsetAsync(x, 0, sizeof(Real)*N));
        CUDA_DEBUG_TEST(cudaMemcpyAsync(r, b, sizeof(Real)*N, cudaMemcpyDeviceToDevice));
        CUDA_DEBUG_TEST(cudaMemcpyAsync(p, b, sizeof(Real)*N, cudaMemcpyDeviceToDevice));

        r_dot_r = vector_dot_product(b, b, N);
        CUDA_DEBUG_TEST(cudaDeviceSynchronize());
    }

    if(0)
    {
        Real max_diff = vector_max(r, N);
        printf("First error MAX: " REAL_FMT " AVG: " REAL_FMT "\n", max_diff, sqrt(r_dot_r/N));
    }

    int iter = 0;
    if(r_dot_r >= scaled_squared_tolerance)
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


void semi_implicit_solver_resize(Semi_Implicit_Solver* solver, int n, int m)
{
    if(solver->m != m || solver->n != n)
    {
        //Big evil programming practices because we are cool and we know
        // what we are doing and dont care much about what others have to
        // say
        //@TODO: make this on demand load
        int N = n*m;
        int N_old = solver->n*solver->m;

        Real* maps = (Real*) (void*) &solver->maps;
        for(int i = 0; i < sizeof(solver->maps) / sizeof(Real); i++)
            cuda_realloc_in_place((void**) &maps[i], N*sizeof(Real), N_old*sizeof(Real), REALLOC_ZERO);

        Real* debug_maps = (Real*) (void*) &solver->debug_maps;
        for(int i = 0; i < sizeof(solver->debug_maps) / sizeof(Real); i++)
            cuda_realloc_in_place((void**) &debug_maps[i], N*sizeof(Real), N_old*sizeof(Real), REALLOC_ZERO);

        solver->m = m;
        solver->n = n;
    }
}

void semi_implicit_solver_step_based(Semi_Implicit_Solver* solver, Real* F, Real* U, Real* U_base, Semi_Implicit_State next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    Real dx = (Real) params.L0 / solver->m;
    Real dy = (Real) params.L0 / solver->n;

    int m = solver->m;
    int n = solver->n;
    int N = m*n;

    Real mK = dx * dy;
    Real a = params.a;
    Real b = params.b;
    Real alpha = params.alpha;
    Real beta = params.beta;
    Real xi = params.xi;
    Real Tm = params.Tm;
    Real L = params.L; 
    Real dt = params.dt;
    Real S = params.S; 
    Real m0 = params.m0; 
    Real theta0 = params.theta0;
    Real gamma = params.gamma;
    
    Real* F_next = next_state.F;
    Real* U_next = next_state.U;
    
    Real* b_F = solver->maps.b_F;
    Real* b_U = solver->maps.b_U;

    Anisotrophy_Matrix A_F = {0};
    A_F.anisotrophy = solver->maps.anisotrophy;
    A_F.C_minus_one = 2*dt/(dx*dx) + 2*dt/(dy*dy);
    A_F.X = -dt/(dx*dx);
    A_F.Y = -dt/(dy*dy);
    A_F.m = m;
    A_F.n = n;

    Cross_Matrix_Static A_U = {0};
    A_U.C = 1 + 2*dt/(dx*dx) + 2*dt/(dy*dy);
    A_U.R = -dt/(dx*dx);
    A_U.L = -dt/(dx*dx);
    A_U.U = -dt/(dy*dy);
    A_U.D = -dt/(dy*dy);
    A_U.m = m;
    A_U.n = n;

    bool do_corrector_guess = params.do_corrector_guess;
    cuda_for_2D(0, 0, m, n, [=]SHARED(int x, int y){
        Real T = U[x + y*m];
        Real Phi = F[x + y*m];
        Real Phi_U = *at_mod(F, x, y + 1, m, n);
        Real Phi_D = *at_mod(F, x, y - 1, m, n);
        Real Phi_R = *at_mod(F, x + 1, y, m, n);
        Real Phi_L = *at_mod(F, x - 1, y, m, n);

        Real grad_Phi_x = (Phi_R - Phi_L)/(2*dx);
        Real grad_Phi_y = (Phi_U - Phi_D)/(2*dy);
        Real grad_Phi_norm = hypotf(grad_Phi_x, grad_Phi_y);

        Real theta = atan2(grad_Phi_y, grad_Phi_x);;
        Real g_theta = 1.0f - S*cosf(m0*theta + theta0);

        Real laplace_Phi = (Phi_L - 2*Phi + Phi_R)/(dx*dx) + (Phi_D - 2*Phi + Phi_U)/(dy*dy);

        Real k0 = g_theta*a*f0(Phi)/(xi*xi * alpha);
        Real k2 = b*beta/alpha*grad_Phi_norm;
        Real k1 = g_theta/alpha;

        Real right = 0;
        Real factor = 0;

        if(do_corrector_guess)
        {
            Real T_U = *at_mod(U, x, y + 1, m, n);
            Real T_D = *at_mod(U, x, y - 1, m, n);
            Real T_R = *at_mod(U, x + 1, y, m, n);
            Real T_L = *at_mod(U, x - 1, y, m, n);
            Real laplace_T = (T_L - 2*T + T_R)/(dx*dx) + (T_D - 2*T + T_U)/(dy*dy);
            Real corr = 1 + k2*dt*L;

            right = Phi + dt/corr*((1-gamma)*k1*laplace_Phi + k0 - k2*(T - Tm + dt*laplace_T));
            factor = gamma/corr*k1; 
        }
        else
        {
            right = Phi + dt*((1-gamma)*k1*laplace_Phi + k0 - k2*(T - Tm));
            factor = gamma*k1; 
        }

        A_F.anisotrophy[x+y*m] = factor;
        b_F[x + y*m] = right;
    });

    Conjugate_Gardient_Params solver_params = {0};
    solver_params.epsilon = (Real) 1.0e-12;
    solver_params.tolerance = params.Phi_tolerance;
    solver_params.max_iters = params.Phi_max_iters;
    solver_params.initial_value_or_null = F;

    //Solve A_F*F_next = b_F
    Conjugate_Gardient_Convergence F_converged = conjugate_gradient_solve(&A_F, F_next, b_F, N, anisotrophy_matrix_multiply, &solver_params);
    LOG_DEBUG("SOLVER", "%lli F %s in %i iters with error %lf\n", (long long) iter, F_converged.converged ? "converged" : "diverged", F_converged.iters, F_converged.error);

    //@TODO: crank-nicolson version see NME2+TKO notebook

    //Calculate b_U
    cuda_for_2D(0, 0, m, n, [=]SHARED(int x, int y){
        Real T = *at_mod(U_base, x, y, n, m);
        Real Phi = *at_mod(F, x, y, n, m);
        Real Phi_next = *at_mod(F_next, x, y, n, m);

        b_U[x + y*m] = T + L*(Phi_next - Phi) + dt*(1-gamma)*T;
    });

    solver_params.tolerance = params.T_tolerance;
    solver_params.max_iters = params.T_max_iters;
    solver_params.initial_value_or_null = U_base;

    //Solve A_U*U_next = b_U
    Conjugate_Gardient_Convergence U_converged = conjugate_gradient_solve(&A_U, U_next, b_U, N, cross_matrix_static_multiply, &solver_params);
    LOG_DEBUG("SOLVER", "%lli U %s in %i iters with error %lf\n", (long long) iter, U_converged.converged ? "converged" : "diverged", U_converged.iters, U_converged.error);

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

            LOG_DEBUG("SOLVER", "AVG | F:" REAL_FMT " U:" REAL_FMT " Epsilon:" REAL_FMT "\n", back_error_F, back_error_U, solver_params.tolerance*2);
            LOG_DEBUG("SOLVER", "MAX | F:" REAL_FMT " U:" REAL_FMT " Epsilon:" REAL_FMT "\n", back_error_F_max, back_error_U_max, solver_params.tolerance*2);
        }

        Real* grad_F = solver->debug_maps.grad_phi;
        Real* grad_U = solver->debug_maps.grad_T;
        cuda_for_2D(0, 0, m, n, [=]SHARED(int x, int y){
            Real T = *at_mod(U, x, y, n, m);
            Real Phi = *at_mod(F, x, y, n, m);

            Real Phi_U = *at_mod(F, x, y + 1, n, m);
            Real Phi_D = *at_mod(F, x, y - 1, n, m);
            Real Phi_R = *at_mod(F, x + 1, y, n, m);
            Real Phi_L = *at_mod(F, x - 1, y, n, m);

            Real T_U = *at_mod(U, x, y + 1, n, m);
            Real T_D = *at_mod(U, x, y - 1, n, m);
            Real T_R = *at_mod(U, x + 1, y, n, m);
            Real T_L = *at_mod(U, x - 1, y, n, m);

            Real grad_Phi_x = (Phi_R - Phi_L);
            Real grad_Phi_y = (Phi_U - Phi_D);
            Real grad_Phi_norm = hypotf(grad_Phi_x, grad_Phi_y);

            Real grad_T_x = (T_R - T_L);
            Real grad_T_y = (T_U - T_D);
            Real grad_T_norm = hypotf(grad_T_x, grad_T_y);
            
            grad_F[x + y*m] = grad_Phi_norm;
            grad_U[x + y*m] = grad_T_norm;
        });
    }
}


void semi_implicit_solver_step(Semi_Implicit_Solver* solver, Semi_Implicit_State state, Semi_Implicit_State next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    semi_implicit_solver_step_based(solver, state.F, state.U, state.U, next_state, params, iter, do_debug);
}



void semi_implicit_solver_step_corrector(Semi_Implicit_Solver* solver, Semi_Implicit_State state, Semi_Implicit_State next_state, Allen_Cahn_Params params, size_t iter, Allen_Cahn_Stats* stats_or_null)
{
    Cache_Tag tag = cache_tag_make();
    int N = params.n * params.m;

    Explicit_State temp_state = {0};
    temp_state.F = cache_alloc(Real, N, &tag);
    temp_state.U = cache_alloc(Real, N, &tag);
    temp_state.m = params.m;
    temp_state.n = params.n;

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
    
    int k = 0;
    int max_iters = params.corrector_max_iters;
    if(params.do_corrector_loop == false)
        max_iters = 0;

    if(max_iters == 0 && params.do_stats_step_residual)
        max_iters = 1;

    bool do_debug = false;
    if(max_iters == 0)
        do_debug = params.do_debug;
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
        if(k < STATIC_ARRAY_SIZE(solver->debug_maps.step_residuals))
            CUDA_DEBUG_TEST(cudaMemcpyAsync(solver->debug_maps.step_residuals[k], step_resiudal, N*sizeof(Real), cudaMemcpyDeviceToDevice));

        step_residual_avg_error = vector_euclid_norm(step_resiudal, N);
        step_residual_max_error = vector_max(step_resiudal, N);
        if(k < STATIC_ARRAY_SIZE(stats_or_null->L2_step_residuals) && stats_or_null)
        {
            stats_or_null->L2_step_residuals[k] = step_residual_avg_error;
            stats_or_null->Lmax_step_residuals[k] = step_residual_max_error;
            stats_or_null->step_residuals = k + 1;
        }

        LOG_DEBUG("SOLVER", "step residual loop: %i | avg: %e | max: %e | tolerance: %lf", k, step_residual_avg_error, step_residual_max_error, params.corrector_tolerance);
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
            // converged ? "converged" : "diverged", k + 1, step_residual_avg_error, step_residual_max_error, params.corrector_tolerance);

    //If the ended on step is already next_state dont copy anything
    Explicit_State final_step = steps[MOD(k, 2)];
    if(final_step.F != next_state.F)
    {
        CUDA_DEBUG_TEST(cudaMemcpyAsync(next_state.F, final_step.F, N*sizeof(Real), cudaMemcpyDeviceToDevice));
        CUDA_DEBUG_TEST(cudaMemcpyAsync(next_state.U, final_step.U, N*sizeof(Real), cudaMemcpyDeviceToDevice));
    }

    cache_free(&tag);
}

void semi_implicit_solver_get_maps(Semi_Implicit_Solver* solver, Semi_Implicit_State state, Sim_Map* maps, int map_count)
{
    int __map_i = 0;
    memset(maps, 0, sizeof maps[0] * map_count);
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

    int m;
    int n;
};

void semi_implicit_coupled_solver_resize(Semi_Implicit_Coupled_Solver* solver, int n, int m)
{
    if(solver->m != m || solver->n != n)
    {
        int N = n*m;
        int N_old = solver->n*solver->m;
        cuda_realloc_in_place((void**) &solver->b_C, 2*N*sizeof(Real), 2*N_old*sizeof(Real), REALLOC_ZERO);
        cuda_realloc_in_place((void**) &solver->aniso, N*sizeof(Real), N_old*sizeof(Real), REALLOC_ZERO);
        cuda_realloc_in_place((void**) &solver->B_U, N*sizeof(Real), N_old*sizeof(Real), REALLOC_ZERO);

        solver->m = m;
        solver->n = n;
    }
}

void semi_implicit_coupled_state_resize(Semi_Implicit_Coupled_State* state, int n, int m)
{
    if(state->m != m || state->n != n)
    {
        int N = n*m;
        int N_old = state->n*state->m;
        cuda_realloc_in_place((void**) &state->C, 2*N*sizeof(Real), 2*N_old*sizeof(Real), REALLOC_ZERO);
        state->m = m;
        state->n = n;
    }
}

void semi_implicit_coupled_matrix_multiply(Real* out, const void* A_, const Real* x, int vec_size)
{
    Semi_Implicit_Coupled_Cross_Matrix A = *(Semi_Implicit_Coupled_Cross_Matrix*)A_;

    int m = A.m;
    int n = A.n;
    int N = m*n;

    Real* F = (Real*) x;
    Real* U = (Real*) x + N; 

    Real* out_F = out;
    Real* out_U = out + N;

    //F equation
    cuda_for_2D(0, 0, m, n, [=]SHARED(int x, int y){
        int i = x + y*m;
        Real s = A.A_F.anisotrophy[i];
        Real X = A.A_F.X*s;
        Real Y = A.A_F.Y*s;
        Real C = 1 + A.A_F.C_minus_one*s;

        Real F_val = F[i]*C;
        F_val += *at_mod(F, x + 1, y, n, m)*X;
        F_val += *at_mod(F, x - 1, y, n, m)*X;
        F_val += *at_mod(F, x , y + 1, n, m)*Y;
        F_val += *at_mod(F, x , y - 1, n, m)*Y;

        Real U_val = A.B_U[i]*U[i]; 

        out_F[i] = F_val + U_val;
    });

    //U equation
    cuda_for_2D(0, 0, m, n, [=]SHARED(int x, int y){
        int i = x + y*m;

        Real U_val = U[i]*A.A_U.C;
        U_val += *at_mod(U, x + 1, y, n, m)*A.A_U.R;
        U_val += *at_mod(U, x - 1, y, n, m)*A.A_U.L;
        U_val += *at_mod(U, x , y + 1, n, m)*A.A_U.U;
        U_val += *at_mod(U, x , y - 1, n, m)*A.A_U.D;

        Real F_val = F[i]*A.B_F;
        out_U[i] = F_val + U_val;
    });
}

void semi_implicit_coupled_solver_step(Semi_Implicit_Coupled_Solver* solver, Semi_Implicit_Coupled_State state, Semi_Implicit_Coupled_State next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    Real dx = (Real) params.L0 / solver->m;
    Real dy = (Real) params.L0 / solver->n;

    int m = solver->m;
    int n = solver->n;
    int N = m*n;

    Real mK = dx * dy;
    Real a = params.a;
    Real b = params.b;
    Real alpha = params.alpha;
    Real beta = params.beta;
    Real xi = params.xi;
    Real Tm = params.Tm;
    Real L = params.L; 
    Real dt = params.dt;
    Real S = params.S; 
    Real m0 = params.m0; 
    Real theta0 = params.theta0;

    Real* F = state.C;
    Real* U = state.C + N;
    
    Real* b_F = solver->b_C;
    Real* b_U = solver->b_C + N;

    Real* aniso = solver->aniso;
    Real* B_U = solver->B_U;

    //Prepare dynamic data
    cuda_for_2D(0, 0, m, n, [=]SHARED(int x, int y){
        Real T = U[x + y*m];
        Real Phi = F[x + y*m];

        Real Phi_U = *at_mod(F, x, y + 1, m, n);
        Real Phi_D = *at_mod(F, x, y - 1, m, n);
        Real Phi_R = *at_mod(F, x + 1, y, m, n);
        Real Phi_L = *at_mod(F, x - 1, y, m, n);

        Real grad_Phi_x = (Phi_R - Phi_L)/(2*dx);
        Real grad_Phi_y = (Phi_U - Phi_D)/(2*dy);
        Real grad_Phi_norm = hypotf(grad_Phi_x, grad_Phi_y);
 
        Real g_theta = 1;
        {
            Real theta = atan2(grad_Phi_y, grad_Phi_x);
            g_theta = 1.0f - S*cosf(m0*theta + theta0);
        }

        // g_theta = 1;
        Real f_tilda = b*xi*xi*beta*grad_Phi_norm;
        Real f = g_theta*a*f0(Phi) - b*xi*xi*beta*(T - Tm)*grad_Phi_norm;
        Real T_factor = dt*f_tilda/(xi*xi*alpha);
        B_U[x+y*m] = -T_factor;
        aniso[x+y*m] = g_theta;
        b_F[x + y*m] = Phi + dt*g_theta/(xi*xi*alpha)*f0(Phi) - T_factor*Tm;
        b_U[x + y*m] = T + L*Phi;
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
    A_C.m = m;
    A_C.n = n;

    Conjugate_Gardient_Params solver_params = {0};
    solver_params.epsilon = (Real) 1.0e-10;
    solver_params.tolerance = (Real) 1.0e-7;
    solver_params.max_iters = 200;
    solver_params.initial_value_or_null = state.C;

    Conjugate_Gardient_Convergence conv = conjugate_gradient_solve(&A_C, next_state.C, solver->b_C, 2*N, semi_implicit_coupled_matrix_multiply, &solver_params);
    printf("%lli C %s in %i iters with error %lf\n", (long long) iter, conv.converged ? "converged" : "diverged", conv.iters, conv.error);
}

void semi_implicit_coupled_solver_get_maps(Semi_Implicit_Coupled_Solver* solver, Semi_Implicit_Coupled_State state, Sim_Map* maps, int map_count)
{
    int N = solver->m*solver->n;
    int __map_i = 0;
    memset(maps, 0, sizeof maps[0] * map_count);
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


extern "C" void sim_solver_reinit(Sim_Solver* solver, Solver_Type type, int n, int m)
{
    if(solver->type != type && solver->type != SOLVER_TYPE_NONE)
        sim_solver_reinit(solver, solver->type, 0, 0);

    switch(type) {
        case SOLVER_TYPE_NONE: {
            n = 0;
            m = 0;
        } break;

        case SOLVER_TYPE_EXPLICIT: 
        case SOLVER_TYPE_EXPLICIT_RK4:
        case SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE: {
            explicit_solver_resize(&solver->expli, n, m);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT: {
            semi_implicit_solver_resize(&solver->impli, n, m);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: {
            semi_implicit_coupled_solver_resize(&solver->impli_coupled, n, m);
        } break;

        default: {
            assert(false);
        }
    };

    solver->type = type;
    solver->m = m;
    solver->n = n;
}

void sim_state_reinit(Sim_State* states, Solver_Type type, int n, int m)
{
    if(states->type != type && states->type != SOLVER_TYPE_NONE)
        sim_state_reinit(states, states->type, 0, 0);

    switch(type) {
        case SOLVER_TYPE_NONE: {
            n = 0;
            m = 0;
        } break;

        case SOLVER_TYPE_EXPLICIT: 
        case SOLVER_TYPE_EXPLICIT_RK4:
        case SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE: {
            explicit_state_resize(&states->expli, n, m);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT: {
            //For the moemnt these are the same
            explicit_state_resize(&states->impli, n, m);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: {
            semi_implicit_coupled_state_resize(&states->impli_coupled, n, m);
        } break;

        default: {
            assert(false);
        }
    };

    states->type = type;
    states->m = m;
    states->n = n;
}

extern "C" void sim_states_reinit(Sim_State* states, int state_count, Solver_Type type, int n, int m)
{
    for(int i = 0; i < state_count; i++)
        sim_state_reinit(&states[i], type, n, m);
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
                step_by = explicit_solver_choose_and_copute_step_residual(solver->type, &solver->expli, state.expli, &next_state.expli, params, iter, stats_or_null);
            } break;

            case SOLVER_TYPE_SEMI_IMPLICIT: {
                semi_implicit_solver_step_corrector(&solver->impli, state.impli, next_state.impli, params, iter, stats_or_null);
            } break;

            case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: {
                semi_implicit_coupled_solver_step(&solver->impli_coupled, state.impli_coupled, next_state.impli_coupled, params, iter, params.do_debug);
            } break;

            default: assert(false);
        };
    }

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


extern "C" bool benchmark_reduce_kernels(int N)
{
    (void) N;
    return false;
}
#else

extern "C" void sim_solver_reinit(Sim_Solver* solver, Solver_Type type, int n, int m) {}
extern "C" void sim_states_reinit(Sim_State* states, int state_count, Solver_Type type, int n, int m) {}
extern "C" void sim_solver_get_maps(Sim_Solver* solver, Sim_State* states, int states_count, int iter, Sim_Map* maps, int map_count) {}
extern "C" double sim_solver_step(Sim_Solver* solver, Sim_State* states, int states_count, int iter, Allen_Cahn_Params params, Allen_Cahn_Stats* stats_or_null) {return 0;}

extern "C" void sim_modify(void* device_memory, void* host_memory, size_t size, Sim_Modify modify) {}
extern "C" void sim_modify_float(Real* device_memory, float* host_memory, size_t size, Sim_Modify modify) {}
extern "C" void sim_modify_double(Real* device_memory, double* host_memory, size_t size, Sim_Modify modify) {}


enum {
    WARP_SIZE = 32, 
    // We use a constant. If this is not the case we will 'need' a different algorhimt anyway. 
    // The PTX codegen treats warpSize as 'runtime immediate constant' which from my undertanding
    // is a special constat accesible through its name 'mov.u32 %r6, WARP_SZ;'. 
    // In many contexts having it be a immediate constant is not enough. 
    // For example when doing 'x % warpSize == 0' the code will emit 'rem.s32' instruction which is
    // EXTREMELY costly (on GPUs even more than CPUs!) compared to the single binary 'and.b32' emmited 
    // in case of 'x % WARP_SIZE == 0'.

    // If you would need to make the below code a bit more "future proof" you could make WARP_SIZE into a 
    // template argument. However we really dont fear this to cahnge anytime soon since many functions
    // such as __activemask(), __ballot_sync() or anything else producing lane mask, operates on u32 which 
    // assumes WARP_SIZE == 32.
};

enum {
    REDUCE_WARP_GATHER,
    REDUCE_SHARED_ONLY_FOLD,
    REDUCE_WARP_ONLY_FOLD,
    REDUCE_SAMPLES_VERSION,
};

template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        mySum += __shfl_down_sync(mask, mySum, offset);
    }
    return mySum;
}

template <typename T>
__global__ void reduce7(const T *__restrict__ input, T *__restrict__ output, unsigned int n) 
{
    assert(blockDim.x > WARP_SIZE && blockDim.x % WARP_SIZE == 0 && "we expect the block dim to be chosen as expected");

    uint shared_size = blockDim.x / WARP_SIZE;
    extern __shared__ T shared[];

    uint tid = threadIdx.x;
    uint grids_size = blockDim.x * gridDim.x;
    uint mask = 0xffffffff;

    T sum = 0;
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n) 
    {
        sum += input[i];
        i += grids_size;
    }

    sum = warpReduceSum<T>(mask, sum);
    if (tid % WARP_SIZE == 0) 
        shared[tid / WARP_SIZE] = sum;
    
    __syncthreads();
    uint ballot_mask = __ballot_sync(mask, tid < shared_size);
    if (tid < shared_size) 
    {
        sum = shared[tid];
        sum = warpReduceSum<T>(ballot_mask, sum);
    }

    // write result for this block to global mem
    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}

struct Reduce {
    struct Add {};
    struct Mul {};
    struct Min {};
    struct Max {};
    struct Or {};
    struct And {};
    struct Xor {};

    static Add ADD;
    static Mul MUL;
    static Min MIN;
    static Max MAX;
    static Or  OR;
    static And AND;
    static Xor XOR;
};

template <class Reduction, typename T>
__host__ __device__ __forceinline__ T _reduce(T a, T b)
{
    if constexpr      (std::is_same_v<Reduction, Reduce::Add>)
        return a + b;
    else if constexpr (std::is_same_v<Reduction, Reduce::Mul>)
        return a * b;
    else if constexpr (std::is_same_v<Reduction, Reduce::Min>)
        return MIN(a, b);
    else if constexpr (std::is_same_v<Reduction, Reduce::Max>)
        return MAX(a, b);
    else if constexpr (std::is_same_v<Reduction, Reduce::And>)
        return a & b;
    else if constexpr (std::is_same_v<Reduction, Reduce::Or>)
        return a | b;
    else if constexpr (std::is_same_v<Reduction, Reduce::Xor>)
        return a ^ b;
    else
        return Reduction::reduce(a, b);
}

template <class Reduction, typename T>
__host__ __device__ __forceinline__ T _identity()
{
    if constexpr      (std::is_same_v<Reduction, Reduce::Add>)
        return (T) 0;
    else if constexpr (std::is_same_v<Reduction, Reduce::Mul>)
        return (T) 1;
    else if constexpr (std::is_same_v<Reduction, Reduce::Min>)
        return (T) MAX(std::numeric_limits<T>::max(), std::numeric_limits<T>::infinity());
    else if constexpr (std::is_same_v<Reduction, Reduce::Max>)
        return (T) MIN(std::numeric_limits<T>::min(), -std::numeric_limits<T>::infinity());
    else if constexpr (std::is_same_v<Reduction, Reduce::And>)
        return (T) 0xFFFFFFFFFFFFULL;
    else if constexpr (std::is_same_v<Reduction, Reduce::Or>)
        return (T) 0;
    else if constexpr (std::is_same_v<Reduction, Reduce::Xor>)
        return (T) 0;
    else
        return Reduction::identity();
}

template <class Reduction, typename T>
__device__ __forceinline__ T warp_reduce(unsigned int mask, T value) 
{
    //For integral (up to 32 bit) types we can use builtins for super fast reduction.
    constexpr bool is_okay_int = std::is_integral_v<T> && sizeof(T) <= sizeof(int); (void) is_okay_int;

    if constexpr(0) {}
    #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    else if constexpr (is_okay_int && std::is_same_v<Reduction, Reduce::Add>)
        return (T) __reduce_add_sync(mask, value);
    else if constexpr (is_okay_int && std::is_same_v<Reduction, Reduce::Min>)
        return (T) __reduce_min_sync(mask, value);
    else if constexpr (is_okay_int && std::is_same_v<Reduction, Reduce::Max>)
        return (T) __reduce_max_sync(mask, value);
    else if constexpr (is_okay_int && std::is_same_v<Reduction, Reduce::And>)
        return (T) __reduce_and_sync(mask, (unsigned) value);
    else if constexpr (is_okay_int && std::is_same_v<Reduction, Reduce::Or>)
        return (T) __reduce_or_sync(mask, (unsigned) value);
    else if constexpr (is_okay_int && std::is_same_v<Reduction, Reduce::Xor>)
        return (T) __reduce_xor_sync(mask, (unsigned) value);
    #endif
    else
    {
        //@NOTE: T must be one of: (unsigned) short, (unsigned) int, (unsigned) long long, hafl, float, double.
        //Otherwise when sizeof(T) < 8 we will need to write custom casting code
        //Otherwiese when sizeof(T) > 8 we will need to change to algorhimt to do only shared memory reduction
        #if 1
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) 
            value = _reduce<Reduction, T>(__shfl_down_sync(mask, value, offset), value);
        #else
            value = _reduce<Reduction, T>(__shfl_down_sync(mask, value, 16), value);
            value = _reduce<Reduction, T>(__shfl_down_sync(mask, value, 8), value);
            value = _reduce<Reduction, T>(__shfl_down_sync(mask, value, 4), value);
            value = _reduce<Reduction, T>(__shfl_down_sync(mask, value, 2), value);
            value = _reduce<Reduction, T>(__shfl_down_sync(mask, value, 1), value);
        #endif

        return value;
    }
}

//@TODO: finish & test reduction
//@TODO: benchmark improvemnt with semi implcit and adaptive rk4 method

//produce_reduce(N, out, (int)->Real, Reduction::Sum)
//reduce<Reduction::Sum>(N, out, in)
// -> compare for equality if types equal inline. Has to be a template argument
// -> template arguments must have their 'subtype' present. this is ugly.
// -> If we do the tag approach we get rid of a certain template argument explosion

//reduce<Reduction::Sum>
//reduce<MyReducer>

// Convolution?

//-> L1 diff
//-> L2 diff
//-> Lmax diff
//-> dot product
//-> sum
//-> max

template <class Reduction, typename T, typename Producer, bool is_trivial_producer>
__global__ void produce_reduce_kernel(T* __restrict__ output, Producer produce, uint N) 
{
    assert(blockDim.x > WARP_SIZE && blockDim.x % WARP_SIZE == 0 && "we expect the block dim to be chosen sainly");
    uint shared_size = blockDim.x / WARP_SIZE;

    extern __shared__ max_align_t shared_backing[];
    T* shared = (T*) (void*) shared_backing;

    T reduced = _identity<Reduction, T>();
    for(uint i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x)
    {
        T produced;
        if constexpr(is_trivial_producer)
            produced = produce[i];
        else 
            produced = produce(i);

        reduced = _reduce<Reduction, T>(produced, reduced);
    }

    reduced = warp_reduce<Reduction, T>(0xffffffffU, reduced);
    uint ti = threadIdx.x;
    if (ti % WARP_SIZE == 0) 
        shared[ti / WARP_SIZE] = reduced;
    
    __syncthreads();
    uint ballot_mask = __ballot_sync(0xffffffffU, ti < shared_size);
    if (ti < shared_size) 
    {
        reduced = shared[ti];
        reduced = warp_reduce<Reduction, T>(ballot_mask, reduced);
    }

    if (ti == 0) 
        output[blockIdx.x] = reduced;
}

#if 0
template<uint SHARED_ITEMS, uint GATHER_MULT, int FLAGS, typename Map_Func, typename Reduce_Func>
__global__ void map_reduce_kernel(Real* __restrict__ reduced, Map_Func map, Reduce_Func reduce, Real identity_element, uint N)
{
    if(FLAGS & REDUCE_WARP_GATHER)
        assert(SHARED_ITEMS*REDUCE_WARP_GATHER == blockDim.x);
    else
        assert(SHARED_ITEMS == blockDim.x);

    assert(WARP_SIZE == warpSize);

    //GATHER_MULT times because the first reduction is while loading to shared memory by doing GATHER_MULT iterations
    uint segment_size = GATHER_MULT*SHARED_ITEMS;
    uint ti = threadIdx.x;

    __shared__ Real shared[SHARED_ITEMS];

    for (uint segment = segment_size*blockIdx.x; ; segment += segment_size*gridDim.x) 
    {
        uint top_level_mask = __ballot_sync(0xffffffffU, segment + ti < N);
        if(segment + ti >= N)
        {
            shared[ti] = identity_element;
            break;
        }

        //Taken from: 

        //We reduce each segment (2*SHARED_ITEMS = 256) into a single
        // value. These single values must then be finally reduced 
        // a) on the cpu
        // b) by relaunching this kernel
        // c) using atomics.
        //
        // We do either (a) or (b) depending on the count of the values. 
        // If there are more then SHARED_ITEMS*X values we relaunch.

        // 1 [--------------segment-----------------]|[--------------segment-----------------]
        // 2 [--------shared------]                  |[--------shared------]                  
        // 3 [---shared---]                          |[---shared---]                          
        //   ...                                     |...                                       
        // n [--warp--]                              |[--warp--]                              
        //   ..                                      |...                                        
        // m x                                       |x    

        //Operations in shared memory index by threadIdx.x since shared memory is seperate for each block
        // & threadIdx.x is the index within each block.
        
        Real sum = map(segment + ti);
        for(uint k = 1; k < GATHER_MULT; k++)
        {
            if(segment + ti + k*SHARED_ITEMS < N)
            {
                Real other = map(segment + ti + k*SHARED_ITEMS);
                sum = reduce(sum, other);
            }
        }

        if(FLAGS & REDUCE_WARP_GATHER)
        {
            __syncwarp();
            for(uint stride = WARP_SIZE/2; stride > 0; stride /= 2)
                sum = reduce(sum, __shfl_down_sync(top_level_mask, sum, stride));

            if(ti % WARP_SIZE == 0)
                shared[ti / WARP_SIZE] = sum;
        }
        else
        {
            shared[ti] = sum;
        }
        __syncthreads();

        if(FLAGS & REDUCE_SHARED_ONLY_FOLD)
        {
            for(uint stride = SHARED_ITEMS/2; stride > 0; stride /= 2)
            {
                if(ti < stride) 
                    shared[ti] = reduce(shared[ti], shared[ti + stride]);
                
                __syncthreads();
            }

            if(ti == 0)
                reduced[segment / segment_size] = shared[0];
        }
        else if(FLAGS & REDUCE_WARP_ONLY_FOLD)
        {
            assert(SHARED_ITEMS % WARP_SIZE == 0);

            int iters = 0;
            // for(uint big_stride = SHARED_ITEMS; big_stride > 1; big_stride = 1)
            for(uint big_stride = MIN(SHARED_ITEMS, N); big_stride > 1; big_stride = DIV_CEIL(big_stride, WARP_SIZE))
            {
                assert(iters++ < 2);

                if(ti < big_stride)
                    sum = shared[ti];
                else
                    sum = identity_element;

                // uint mask = __ballot_sync(top_level_mask, ti < big_stride);
                // __syncwarp(mask); //@TODO: is this necessary here? probably not!
                
                __syncwarp(top_level_mask);
                for(uint stride = WARP_SIZE/2; stride > 0; stride /= 2) 
                    sum = reduce(sum, __shfl_down_sync(top_level_mask, sum, stride));

                __syncthreads();
                if(ti % WARP_SIZE == 0)
                    shared[ti / WARP_SIZE] = sum;

                __syncthreads();
            }

            if(ti == 0)
                reduced[segment / segment_size] = shared[0];
        }
        else
        {
            assert(SHARED_ITEMS > WARP_SIZE);
            for(uint stride = SHARED_ITEMS/2; stride > WARP_SIZE; stride /= 2)
            {
                if(ti < stride) 
                    shared[ti] = reduce(shared[ti], shared[ti + stride]);
                
                __syncthreads();
            }
            
            if(ti < WARP_SIZE) {
                sum = reduce(shared[ti], shared[ti + WARP_SIZE]);
                
                for(uint stride = WARP_SIZE/2; stride > 0; stride /= 2)
                    sum = reduce(sum, __shfl_down_sync(0xffffffffU, sum, stride));
            }

            //Store the partial result
            if(ti == 0)
                reduced[segment / segment_size] = sum;
        }
    }
}
#else
__global__ void map_reduce_kernel(Real* __restrict__ reduced, Real* __restrict__ input, uint N)
{
    //GATHER_MULT times because the first reduction is while loading to shared memory by doing GATHER_MULT iterations
    uint ti = threadIdx.x;

    extern __shared__ Real shared[];

    Real sum = 0;
    uint i = blockDim.x*blockIdx.x + ti;
    while(i < N)
    {
        sum += input[i];
        i += blockDim.x*gridDim.x;
    }

    // __syncwarp();
    for(uint stride = WARP_SIZE/2; stride > 0; stride /= 2)
        sum = sum + __shfl_down_sync(0xffffffffU, sum, stride);

    if(ti % WARP_SIZE == 0)
        shared[ti / WARP_SIZE] = sum;
    
    __syncthreads();

    uint mask = __ballot_sync(0xffffffffU, ti < blockDim.x);
    if(ti < blockDim.x) {
        sum = shared[ti];

        for(uint stride = WARP_SIZE/2; stride > 0; stride /= 2)
            sum = sum + __shfl_down_sync(mask, sum, stride);

        if(ti % WARP_SIZE == 0)
            shared[ti / WARP_SIZE] = sum;
    }

    __syncthreads();
    uint extent = DIV_CEIL(blockDim.x, WARP_SIZE);
    mask = __ballot_sync(0xffffffffU, ti < extent);
    if(ti < extent) {
        sum = shared[ti*WARP_SIZE]; //uncoalesced

        for(uint stride = WARP_SIZE/2; stride > 0; stride /= 2)
            sum = sum + __shfl_down_sync(mask, sum, stride);
    }

    //Store the partial result
    if(ti == 0)
        reduced[blockIdx.x] = sum;
}
#endif

template <typename T, class Reduction, typename Producer, bool has_trivial_producer = false>
T produce_reduce(uint N, Producer produce, Reduction reduce_dummy = Reduction(), uint cpu_reduce = 128)
{
    (void) reduce_dummy;
    enum {
        CPU_REDUCE_MAX = 16,
        CPU_REDUCE_MIN = 16,
    };

    T reduced = _identity<Reduction, T>();
    if(N > 0)
    {
        Cache_Tag tag = cache_tag_make();
        Cuda_Info info = cuda_one_time_setup();
        cpu_reduce = CLAMP(cpu_reduce, CPU_REDUCE_MIN, CPU_REDUCE_MAX);
        uint N_curr = N;

        uint max_shared_items = info.prop.sharedMemPerBlock / sizeof(T);
        uint block_size_hw_upper_bound = info.prop.maxThreadsPerBlock;
        uint block_size_hw_lower_bound = DIV_CEIL(info.prop.maxThreadsPerMultiProcessor + 1, info.prop.maxBlocksPerMultiProcessor + 1);

        uint block_size_upper_bound = MIN(block_size_hw_upper_bound, max_shared_items*WARP_SIZE);
        uint block_size_lower_bound = block_size_hw_lower_bound;

        uint block_size_max = (block_size_upper_bound/WARP_SIZE)*WARP_SIZE;
        uint block_size_min = DIV_CEIL(block_size_lower_bound,WARP_SIZE)*WARP_SIZE;
        if(block_size_min > block_size_max || WARP_SIZE != info.prop.warpSize)
        {
            LOG_ERROR("kernel", "this device has very strange hardware and as such we cannot launch the redutcion kernel. This shouldnt happen.");
            return reduced;
        }
        
        //Find the optimal block_size. It must
        // 1) be twithin range [block_size_min, block_size_max] (to be feasible)
        // 2) be divisible by warpSize (W) (to have good block utlization)
        // 3) be as big as possible (to obtain maximum paralelism)
        // 4) divide prop.maxThreadsPerMultiProcessor (to have good streaming multiprocessor utilization - requiring less SMs)
        uint block_size = 0;
        for(uint curr_size = block_size_max; curr_size >= block_size_min; curr_size -= WARP_SIZE)
        {
            if(info.prop.maxThreadsPerMultiProcessor % curr_size == 0)
            {
                block_size = curr_size; 
                break;
            }
        }

        //If couldnt find one that would divide cleanly just use the biggest ammount
        if(block_size == 0)
            block_size = block_size_max;

        uint max_concurent_blocks = info.prop.multiProcessorCount*(info.prop.maxThreadsPerMultiProcessor / block_size);
        uint shared_items = DIV_CEIL(block_size, WARP_SIZE);
        ASSERT(block_size_lower_bound <= block_size && block_size <= block_size_upper_bound && block_size % WARP_SIZE == 0);

        T* partials[2] = {NULL, NULL};
        uint iter = 0;
        for(; (iter == 0 && has_trivial_producer == false) || N_curr > cpu_reduce; iter++)
        {
            uint i_curr = iter%2;
            uint i_next = (iter + 1)%2;
            if(iter < 2)
                partials[i_next] = cache_alloc(T, N, &tag);

            uint block_dim = block_size;
            uint grid_dim = MIN(DIV_CEIL(N_curr, block_size), max_concurent_blocks); //Only fire the blocks we actually need!
            uint shared_memory = shared_items*sizeof(T);
            ASSERT(shared_memory <= info.prop.sharedMemPerBlock, "too much shared mem requested!");

            T* __restrict__ curr_input = partials[i_curr];
            T* __restrict__ curr_output = partials[i_next];
            if(iter ==  0)
            {
                produce_reduce_kernel<Reduction, T, Producer, has_trivial_producer>
                    <<<grid_dim, block_dim, shared_memory>>>(curr_output, (Producer&&) produce, N_curr);
            }
            else
            {
                produce_reduce_kernel<Reduction, T, const T* __restrict__, true>
                    <<<grid_dim, block_dim, shared_memory>>>(curr_output, curr_input, N_curr);
            }

            CUDA_DEBUG_TEST(cudaGetLastError());
            CUDA_DEBUG_TEST(cudaDeviceSynchronize());

            uint G = MIN(N_curr, block_size*grid_dim);
            uint N_next = DIV_CEIL(G, WARP_SIZE*WARP_SIZE); 
            N_curr = N_next;
        }   

        //in case N <= CPU_REDUCE and has_trivial_producer 
        // we entirely skipped the whole loop => the partials array is
        // still null. We have to init it to the produce array.
        const T* last_input = partials[iter%2];
        if constexpr(has_trivial_producer)
            if(N <= cpu_reduce)
                last_input = produce;

        T cpu[CPU_REDUCE_MAX];
        cudaMemcpy(cpu, last_input, sizeof(T)*N_curr, cudaMemcpyDeviceToHost);
        for(uint i = 0; i < N_curr; i++)
            reduced = _reduce<Reduction, T>(reduced, cpu[i]);

        cache_free(&tag);
    }

    return reduced;
}

void test_identity()
{
    ASSERT((_identity<Reduce::Add, double>() == 0));
    ASSERT((_identity<Reduce::Min, double>() == INFINITY));
    ASSERT((_identity<Reduce::Max, double>() == -INFINITY));

    ASSERT((_identity<Reduce::Add, int>() == 0));
    ASSERT((_identity<Reduce::Min, int>() == INT_MAX));
    ASSERT((_identity<Reduce::Max, int>() == INT_MIN));

    ASSERT((_identity<Reduce::Add, unsigned>() == 0));
    ASSERT((_identity<Reduce::Min, unsigned>() == UINT_MAX));
    ASSERT((_identity<Reduce::Max, unsigned>() == 0));

    ASSERT((_identity<Reduce::Or, unsigned>() == 0));
    ASSERT((_identity<Reduce::And, unsigned>() == 0xFFFFFFFFU));
    ASSERT((_identity<Reduce::Xor, unsigned>() == 0));
}

template<class Reduction, typename T>
T reduce(const T *input, uint N, Reduction reduce_tag = Reduction())
{
    T output = produce_reduce<T, Reduction, const T* __restrict__, true>(N, input);
    return output;
}

#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

template<class Reduction, typename T>
T thrust_reduce(const T *input, int n, Reduction reduce_tag = Reduction())
{
    // wrap raw pointers to device memory with device_ptr
    thrust::device_ptr<const T> d_input(input);

    T id = _identity<Reduction, T>();
    if constexpr(std::is_same_v<Reduction, Reduce::Add>)
        return thrust::reduce(d_input, d_input+n, id, thrust::plus<T>());
    else if constexpr(std::is_same_v<Reduction, Reduce::Min>)
        return thrust::reduce(d_input, d_input+n, id, thrust::minimum<T>());
    else if constexpr(std::is_same_v<Reduction, Reduce::Max>)
        return thrust::reduce(d_input, d_input+n, id, thrust::maximum<T>());
    else if constexpr(std::is_same_v<Reduction, Reduce::And>)
        return thrust::reduce(d_input, d_input+n, id, thrust::bit_and<T>());
    else if constexpr(std::is_same_v<Reduction, Reduce::Or>)
        return thrust::reduce(d_input, d_input+n, id, thrust::bit_or<T>());
    else if constexpr(std::is_same_v<Reduction, Reduce::Xor>)
        return thrust::reduce(d_input, d_input+n, id, thrust::bit_xor<T>());
    else
        ASSERT(false, "Bad reduce type for thrust!");
    return 0;
}

template<class Reduction, typename T>
T cpu_reduce(const T *input, int n, Reduction reduce_tag = Reduction())
{
    enum {COPY_AT_ONCE = 256};
    T cpu[COPY_AT_ONCE] = {0};
    T sum = _identity<Reduction, T>();

    for(int k = 0; k < n; k += COPY_AT_ONCE)
    {
        int from = k;
        int to = MIN(k + COPY_AT_ONCE, n);
        cudaMemcpy(cpu, input + from, sizeof(T)*(to - from), cudaMemcpyDeviceToHost);
        
        for(int i = 0; i < to - from; i++)
            sum = _reduce<Reduction, T>(sum, cpu[i]);
    }

    return sum;
}

template<class Reduction, typename T>
T cpu_fold_reduce(const T *input, int n, Reduction reduce_tag = Reduction())
{
    T sum = _identity<Reduction, T>();
    if(n > 0)
    {
        static T* copy = 0;
        static int local_capacity = 0;
        if(local_capacity < n)
        {
            copy = (T*) realloc(copy, n*sizeof(T));
            local_capacity = n;
        }

        cudaMemcpy(copy, input, sizeof(T)*n, cudaMemcpyDeviceToHost);
        for(int range = n; range > 1; range /= 2)
        {
            // if(range == n)
            // {
            //     printf("%i > ", n);
            //     for(int i = 0; i < range; i ++)
            //         printf("%.2lf ", copy[i]);
            //     printf("\n");
            // }

            for(int i = 0; i < range/2 ; i ++)
                copy[i] = _reduce<Reduction, T>(copy[2*i], copy[2*i + 1]);

            if(range%2)
                copy[range/2 - 1] = _reduce<Reduction, T>(copy[range/2 - 1], copy[range - 1]);
        }

        sum  = copy[0];
    }

    return sum;
}


#include <chrono>
static int64_t clock_ns()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

static double clock_s()
{
    static int64_t init_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int64_t now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    double unit = (double) std::chrono::high_resolution_clock::period::den;
    double clock = (double) (now - init_time) / unit;
    return clock;
}


template <typename Func>
double benchmark(double time, double warmup, Func func)
{
    int64_t sec_to_ns = 1000000000;
    int64_t start_time = clock_ns();
    int64_t time_ns = (int64_t)(time * sec_to_ns);
    int64_t warmup_ns = (int64_t)(warmup * sec_to_ns);

    int64_t sum = 0;
    int64_t iters = 0;
    int64_t start = clock_ns();
    int64_t before = 0;
    int64_t after = start;
    for(; after < start + time_ns; iters ++)
    {
        before = clock_ns();
        func();
        after = clock_ns();

        if(after >= start + warmup_ns)
            sum += after - before;
    }

    double avg = (double) sum / (double) iters / (double) sec_to_ns; 
    return avg;
}

template <typename Func>
double benchmark(double time, Func func)
{
    return benchmark(time, time / 10.0, func);
}

bool is_near(double a, double b, double epsilon = 1e-8)
{
    //this form guarantees that is_nearf(NAN, NAN, 1) == true
    return !(fabs(a - b) > epsilon);
}

//Returns true if x and y are within epsilon distance of each other.
//If |x| and |y| are less than 1 uses epsilon directly
//else scales epsilon to account for growing floating point inaccuracy
bool is_near_scaled(double x, double y, double epsilon = 1e-8)
{
    //This is the form that produces the best assembly
    double calced_factor = fabs(x) + fabs(y);
    double factor = 2 > calced_factor ? 2 : calced_factor;
    return is_near(x, y, factor * epsilon / 2);
}

template<typename T>
void test_reduce(uint64_t seed = (uint64_t) clock_ns())
{
    test_identity();
    uint Ns[] = {0, 1, 5, 31, 32, 33, 64, 65, 256, 257, 512, 513, 1023, 1024, 1025, 256*256, 1024*1024 - 1, 1024*1024};

    //Find max size 
    uint N = 0;
    for(int i = 0; i < STATIC_ARRAY_SIZE(Ns); i++)
        if(N < Ns[i])
            N = Ns[i];

    //generate max sized map of radonom data.
    Cache_Tag tag = cache_tag_make();
    uint32_t* rand_state = cache_alloc(uint, N, &tag);
    T*        rand = cache_alloc(T, N, &tag);
    cuda_for(0, (int)N, [=]SHARED(int i) {
        uint32_t hashed_index = pcg_hash((uint32_t) i);
        uint32_t fold = (uint32_t) seed ^ (uint32_t) (seed >> 32);

        uint32_t hashed_together = pcg_hash(hashed_index ^ fold);
        rand_state[i] = hashed_together;
        rand[i] = (T) 0;
        if constexpr(std::is_floating_point_v<T>)
            rand[i] = (T) hashed_together / (T) UINT32_MAX;
        else if constexpr(std::is_integral_v<T>)
            rand[i] = (T) hashed_together;
    });

    //test each size on radom data.
    for(int i = 0; i < STATIC_ARRAY_SIZE(Ns); i++)
    {
        uint n = Ns[i];
        T sum0 = cpu_reduce(rand, n, Reduce::ADD);
        T sum1 = cpu_fold_reduce(rand, n, Reduce::ADD);
        T sum2 = thrust_reduce(rand, n, Reduce::ADD);
        T sum3 = reduce(rand, n, Reduce::ADD);

        //naive cpu sum reduce diverges from the true values for large n due to
        // floating point rounding
        if(n < 1024) TEST(is_near(sum1, sum0));

        // TEST(is_near(sum1, sum2)); //thrust gives inaccuarte results...
        TEST(is_near(sum1, sum3));

        T min0 = cpu_reduce(rand, n, Reduce::MIN);
        T min1 = cpu_fold_reduce(rand, n, Reduce::MIN);
        T min2 = thrust_reduce(rand, n, Reduce::MIN);
        T min3 = reduce(rand, n, Reduce::MIN);

        TEST(is_near(min1, min0));
        TEST(is_near(min1, min2));
        TEST(is_near(min1, min3));

        T max0 = cpu_reduce(rand, n, Reduce::MAX);
        T max1 = cpu_fold_reduce(rand, n, Reduce::MAX);
        T max2 = thrust_reduce(rand, n, Reduce::MAX);
        T max3 = reduce(rand, n, Reduce::MAX);

        TEST(is_near(max1, max0));
        TEST(is_near(max1, max2));
        TEST(is_near(max1, max3));

        if constexpr(std::is_integral_v<T>)
        {
            T and0 = cpu_reduce(rand, n, Reduce::AND);
            T and1 = cpu_fold_reduce(rand, n, Reduce::AND);
            T and2 = thrust_reduce(rand, n, Reduce::AND);
            T and3 = reduce(rand, n, Reduce::AND);

            TEST(and1 == and0);
            TEST(and1 == and2);
            TEST(and1 == and3);

            T or0 = cpu_reduce(rand, n, Reduce::OR);
            T or1 = cpu_fold_reduce(rand, n, Reduce::OR);
            T or2 = thrust_reduce(rand, n, Reduce::OR);
            T or3 = reduce(rand, n, Reduce::OR);

            TEST(or1 == or0);
            TEST(or1 == or2);
            TEST(or1 == or3);

            T xor0 = cpu_reduce(rand, n, Reduce::XOR);
            T xor1 = cpu_fold_reduce(rand, n, Reduce::XOR);
            T xor2 = thrust_reduce(rand, n, Reduce::XOR);
            T xor3 = reduce(rand, n, Reduce::XOR);

            TEST(xor1 == xor0);
            TEST(xor1 == xor2);
            TEST(xor1 == xor3);
        }
    }

    cache_free(&tag);
}

void cache_prepare(int count, int item_size, int N)
{
    Cache_Tag tag = cache_tag_make();
    for(int i = 0; i < count; i++)
        _cache_alloc(item_size*N, &tag, SOURCE_INFO());
    cache_free(&tag);
}

extern "C" bool benchmark_reduce_kernels(int N)
{
    if(1)
    {
        // test_reduce<char>();
        // test_reduce<unsigned char>();
        // test_reduce<short>();
        // test_reduce<ushort>();
        test_reduce<int>();
        // test_reduce<uint>();
        // test_reduce<float>();
        // test_reduce<double>();
    }

    cache_prepare(3, N, sizeof(Real));
    cache_prepare(3, N, sizeof(uint));

    Cache_Tag tag = cache_tag_make();
    Random_Map_State rand = {0};
    rand.N = N;
    rand.state = cache_alloc(uint, N, &tag);
    rand.map = cache_alloc(Real, N, &tag);

    int GB = 1024*1024*1024;
    {
        double cpu_time = benchmark(3, [=]{ cpu_fold_reduce(rand.map, N, Reduce::ADD); });
        double thrust_time = benchmark(3, [=]{ thrust_reduce(rand.map, N, Reduce::ADD); });
        double custom_time = benchmark(3, [=]{ reduce(rand.map, N, Reduce::ADD); });
        double total_gb = (double) N / GB * sizeof(Real);
        LOG_INFO("BENCH", "Benchmark results for %i items (%lfGB)", N, total_gb);
        LOG_INFO("BENCH", "double (gb/s): cpu %5.2lf | thrust: %5.2lf | custom: %5.2lf", total_gb/cpu_time, total_gb/thrust_time, total_gb/custom_time);
        LOG_INFO("BENCH", "double (time): cpu: %e | thrust: %e | custom: %e | custom uint: %e", N, cpu_time, thrust_time, custom_time);
    }

    {
        double cpu_time = benchmark(3, [=]{ cpu_fold_reduce(rand.state, N, Reduce::ADD); });
        double thrust_time = benchmark(3, [=]{ thrust_reduce(rand.state, N, Reduce::ADD); });
        double custom_time = benchmark(3, [=]{ reduce(rand.state, N, Reduce::ADD); });
        double total_gb = (double) N / GB * sizeof(uint);
        LOG_INFO("BENCH", "Benchmark results for %i items (%lfGB)", N, total_gb);
        LOG_INFO("BENCH", "uint (gb/s): cpu %5.2lf | thrust: %5.2lf | custom: %5.2lf", total_gb/cpu_time, total_gb/thrust_time, total_gb/custom_time);
        LOG_INFO("BENCH", "uint (time): cpu: %e | thrust: %e | custom: %e | custom uint: %e", N, cpu_time, thrust_time, custom_time);
    }

    cache_free(&tag);
    return true;
}
#endif