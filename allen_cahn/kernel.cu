#define SHARED __host__ __device__

#include <device_launch_parameters.h>
#include "kernel.h"
#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <cuda_runtime.h>


#define MOD(x, mod) (((x) % (mod) + (mod)) % (mod))
#define MAX(a, b)   ((a) > (b) ? (a) : (b))
#define MIN(a, b)   ((a) < (b) ? (a) : (b))
#define PI          ((Real) 3.14159265359)
#define TAU         (2*PI)
#define CUDA_ERR_AND(err) (err) != cudaSuccess ? (err) :
#define ECHOF(x)    printf(#x ": " REAL_FMT "\n", (x))

#include <cuda_runtime.h>
#include <stdarg.h>
static bool _test_cuda_(cudaError_t error, const char* expression, int line, const char* file, const char* format, ...)
{
    if(error != cudaSuccess)
    {
        printf("CUDA_TEST(%s) failed with %s! %s:%i\n", expression, cudaGetErrorString(error), file, line);
        if(format != NULL && strlen(format) != 0)
        {
            va_list args;
            va_start(args, format);
            vprintf(format, args);
            va_end(args);
            printf("\n");
        }
    }
    fflush(stdout);
    return error == cudaSuccess;
}

static void _test(const char* expression, int line, const char* file, const char* format, ...)
{
    if(format != NULL && strlen(format) != 0)
    {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        printf("\n");
    }
    else
    {
        printf("TEST(%s) failed! %s:%i\n", expression, file, line);
    }

    fflush(stdout);
}

#define CUDA_TEST(status, ...) (_test_cuda_((status), #status,  __LINE__, __FILE__, "" __VA_ARGS__) ? (void) 0 : abort())
#define TEST(x, ...)           ((x) ? (void) 0 : (_test(#x,  __LINE__, __FILE__, "" __VA_ARGS__), abort()))

#undef NDEBUG

#ifdef NDEBUG
    #define CUDA_DEBUG_TEST(status, ...) (0 ? printf("" __VA_ARGS__) : (status))
#else
    #define CUDA_DEBUG_TEST(status, ...) CUDA_TEST(status, __VA_ARGS__)
#endif

struct Cuda_Info {
    int device_id;
    cudaDeviceProp prop;
};

Cuda_Info cuda_one_time_setup()
{
    static bool was_setup = false;
    static Cuda_Info info = {0};

    if(was_setup == false)
    {
        enum {MAX_DEVICES = 100};
        cudaDeviceProp devices[MAX_DEVICES] = {0};
        double scores[MAX_DEVICES] = {0};
        double peak_memory[MAX_DEVICES] = {0};
        
        int nDevices = 0;
        CUDA_TEST(cudaGetDeviceCount(&nDevices));
        if(nDevices > MAX_DEVICES)
        {
            assert(false && "wow this should probably not happen!");
            nDevices = MAX_DEVICES;
        }
        TEST(nDevices > 0, "Didnt find any CUDA capable devices. Stopping.");

        for (int i = 0; i < nDevices; i++) 
            CUDA_DEBUG_TEST(cudaGetDeviceProperties(&devices[i], i));

        //compute maximum in each tracked category to
        // be able to properly select the best device for
        // the job!
        cudaDeviceProp max_prop = {0};
        double max_peak_memory = 0;
        for (int i = 0; i < nDevices; i++) 
        {
            cudaDeviceProp prop = devices[i];
            max_prop.warpSize = MAX(max_prop.warpSize, prop.warpSize);
            max_prop.multiProcessorCount = MAX(max_prop.multiProcessorCount, prop.multiProcessorCount);
            max_prop.concurrentKernels = MAX(max_prop.concurrentKernels, prop.concurrentKernels);
            max_prop.memoryClockRate = MAX(max_prop.memoryClockRate, prop.memoryClockRate);
            max_prop.memoryBusWidth = MAX(max_prop.memoryBusWidth, prop.memoryBusWidth);
            max_prop.totalGlobalMem = MAX(max_prop.totalGlobalMem, prop.totalGlobalMem);
            max_prop.sharedMemPerBlock = MAX(max_prop.sharedMemPerBlock, prop.sharedMemPerBlock);
            peak_memory[i] = 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6;

            max_peak_memory = MAX(max_peak_memory, peak_memory[i]);
        }

        double max_score = 0;
        int max_score_i = 0;
        for (int i = 0; i < nDevices; i++) 
        {
            cudaDeviceProp prop = devices[i];
            double score = 0
                + 0.40 * prop.warpSize/max_prop.warpSize
                + 0.40 * prop.multiProcessorCount/max_prop.multiProcessorCount
                + 0.05 * prop.concurrentKernels/max_prop.concurrentKernels
                + 0.05 * peak_memory[i]/max_peak_memory
                + 0.05 * prop.totalGlobalMem/max_prop.totalGlobalMem
                + 0.05 * prop.sharedMemPerBlock/max_prop.sharedMemPerBlock
                ;

            scores[i] = score;
            if(max_score < score)
            {
                max_score = score;
                max_score_i = i;
            }
        }
        cudaDeviceProp selected = devices[max_score_i];
        info.prop = selected;
        info.device_id = max_score_i;
        was_setup = true;
        CUDA_TEST(cudaSetDevice(info.device_id));

        printf("Listing devices below (%d):\n", nDevices);
        for (int i = 0; i < nDevices; i++)
            printf("%i > %s (score: %lf) %s\n", i, devices[i].name, scores[i], i == max_score_i ? "[selected]" : "");

        printf("Selected %s:\n", selected.name);
        printf("  Multi Processor count: %i\n", selected.multiProcessorCount);
        printf("  Warp-size: %d\n", selected.warpSize);
        printf("  Memory Clock Rate (MHz): %d\n", selected.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n", selected.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n", peak_memory[max_score_i]);
        printf("  Total global memory (Gbytes) %.1f\n",(float)(selected.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n",(float)(selected.sharedMemPerBlock)/1024.0);
        printf("  minor-major: %d-%d\n", selected.minor, selected.major);
        printf("  Concurrent kernels: %s\n", selected.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n\n",selected.deviceOverlap ? "yes" : "no");
    }

    return info;
}

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

enum {
    REALLOC_COPY = 1,
    REALLOC_ZERO = 2,
};

typedef struct Memory_Format {
    const char* unit;
    size_t unit_value;
    double fraction;

    int whole;
    int remainder;
} Memory_Format;

Memory_Format get_memory_format(size_t bytes)
{
    size_t B  = (size_t) 1;
    size_t KB = (size_t) 1024;
    size_t MB = (size_t) 1024*1024;
    size_t GB = (size_t) 1024*1024*1024;
    size_t TB = (size_t) 1024*1024*1024*1024;

    Memory_Format out = {0};
    out.unit = "";
    out.unit_value = 1;
    if(bytes >= TB)
    {
        out.unit = "TB";
        out.unit_value = TB;
    }
    else if(bytes >= GB)
    {
        out.unit = "GB";
        out.unit_value = GB;
    }
    else if(bytes >= MB)
    {
        out.unit = "MB";
        out.unit_value = MB;
    }
    else if(bytes >= KB)
    {
        out.unit = "KB";
        out.unit_value = KB;
    }
    else
    {
        out.unit = "B";
        out.unit_value = B;
    }

    out.fraction = (double) bytes / (double) out.unit_value;
    out.whole = (int) (bytes / out.unit_value);
    out.remainder = (int) (bytes / out.unit_value);

    return out;
}

#define MEMORY_FMT "%.2lf%s"
#define MEMORY_PRINT(bytes) get_memory_format((bytes)).fraction, get_memory_format((bytes)).unit

void* _cuda_realloc(void* old_ptr, size_t new_size, size_t old_size, int flags, const char* file, int line)
{
    printf("CUDA realloc " MEMORY_FMT "-> " MEMORY_FMT " %s:%i\n",
            MEMORY_PRINT(old_size), 
            MEMORY_PRINT(new_size),
            file, line);

    static int64_t used_bytes = 0;
    void* new_ptr = NULL;
    if(new_size != 0)
    {
        Cuda_Info info = cuda_one_time_setup();
        CUDA_TEST(cudaMalloc(&new_ptr, new_size), 
            "Out of CUDA memory! Requested " MEMORY_FMT ". Using " MEMORY_FMT " / " MEMORY_FMT ". %s:%i", 
            MEMORY_PRINT(new_size), 
            MEMORY_PRINT(used_bytes), 
            MEMORY_PRINT(info.prop.totalGlobalMem),
            file, line);

        size_t min_size = MIN(old_size, new_size);
        if((flags & REALLOC_ZERO) && !(flags & REALLOC_COPY))
            CUDA_DEBUG_TEST(cudaMemset(new_ptr, 0, new_size));
        else
        {
            if(flags & REALLOC_COPY)
                CUDA_DEBUG_TEST(cudaMemcpy(new_ptr, old_ptr, min_size, cudaMemcpyDeviceToDevice));
            if(flags & REALLOC_ZERO)
                CUDA_DEBUG_TEST(cudaMemset((uint8_t*) new_ptr + min_size, 0, new_size - min_size));
        }
    }


    CUDA_DEBUG_TEST(cudaFree(old_ptr), 
        "Invalid pointer passed to cuda_realloc! %s:%i", file, line);

    used_bytes += (int64_t) new_size - (int64_t) old_size;
    assert(used_bytes >= 0);
    return new_ptr;
}

void _cuda_realloc_in_place(void** ptr_ptr, size_t new_size, size_t old_size, int flags, const char* file, int line)
{
    *ptr_ptr = _cuda_realloc(*ptr_ptr, new_size, old_size, flags, file, line);
}

#define cuda_realloc(old_ptr, new_size, old_size, flags)          _cuda_realloc(old_ptr, new_size, old_size, flags, __FILE__, __LINE__)
#define cuda_realloc_in_place(ptr_ptr, new_size, old_size, flags) _cuda_realloc_in_place(ptr_ptr, new_size, old_size, flags, __FILE__, __LINE__)

SHARED Real* at_mod(Real* map, int x, int y, int n, int m)
{
    int x_mod = MOD(x, m);
    int y_mod = MOD(y, n);

    return &map[x_mod + y_mod*m];
}

SHARED Real f0(Real phi)
{
	return phi*(1 - phi)*(phi - 1.0f/2);
}


int explicit_solver_resize(Explicit_Solver* solver, int n, int m)
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

    return EXPLICIT_SOLVER_REQUIRED_HISTORY;
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

union Explicit_Solve_In {
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

SHARED Explicit_Solve_Result allen_cahn_explicit_solve(Explicit_Solve_In input, Allen_Cahn_Params params, Explicit_Solve_Debug* debug_or_null)
{
    Real dx = (Real) params.L0 / params.m;
    Real dy = (Real) params.L0 / params.n;
    Real mK = dx * dy;

    //@NOTE: dont you wish we had odin lang using in C?
    Real a = params.a;
    Real b = params.b;
    Real alpha = params.alpha;
    Real beta = params.beta;
    Real xi = params.xi;
    Real Tm = params.Tm;
    Real L = params.L; //Latent heat, not L0 (sym size) ! 
    // Real dt = params.dt;
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

    Real grad_T_x = dy*(T_R - T_L);
    Real grad_T_y = dx*(T_U - T_D);

    Real grad_Phi_x = dy*(Phi_R - Phi_L);
    Real grad_Phi_y = dx*(Phi_U - Phi_D);

    Real grad_T_norm = hypotf(grad_T_x, grad_T_y);
    Real grad_Phi_norm = hypotf(grad_Phi_x, grad_Phi_y);

    Real g_theta = 1;
    Real theta = 0;
    //prevent nans
    // if(grad_Phi_norm > 0.0001)
    {

        theta = atan2(grad_Phi_y, grad_Phi_x);
        g_theta = 1.0f - S*cosf(m0*theta + theta0);
    }

    Real int_K_laplace_T   = dy/dx*(T_L - 2*T + T_R)       + dx/dy*(T_D - 2*T + T_U);
    Real int_K_laplace_Phi = dy/dx*(Phi_L - 2*Phi + Phi_R) + dx/dy*(Phi_D - 2*Phi + Phi_U);
    Real int_K_f = g_theta*a*mK*f0(Phi) - b*xi*xi*beta*(T - Tm)*grad_Phi_norm/2;

    Real int_K_dt_Phi = g_theta/alpha*int_K_laplace_Phi + 1/(xi*xi * alpha)*int_K_f;
    Real int_K_dt_T = int_K_laplace_T + L*int_K_dt_Phi;

    Real dt_Phi = 1/mK*int_K_dt_Phi;
    Real dt_T = 1/mK*int_K_dt_T;

    if(debug_or_null)
    {
        debug_or_null->grad_Phi = hypotf(Phi_R - Phi_L, Phi_U - Phi_D);
        debug_or_null->grad_T = hypotf(T_R - T_L, T_U - T_D);
        debug_or_null->reaction_term = int_K_f / mK;
        debug_or_null->g_theta = g_theta;
        debug_or_null->theta = g_theta;
    }

    Explicit_Solve_Result out = {dt_Phi, dt_T};
    return out;
}

SHARED Explicit_Solve_In explicit_solve_in_mod(const Real* Phi, const Real* T, int x, int y, int n, int m)
{
    Explicit_Solve_In solve = {0};
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
            Explicit_Solve_In input = explicit_solve_in_mod(Phi, T, x, y, n, m);
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
            Explicit_Solve_In input = explicit_solve_in_mod(Phi, T, x, y, n, m);
            Explicit_Solve_Result solved = allen_cahn_explicit_solve(input, params, NULL);

            Phi_next[x + y*m] = input.Phi + solved.dt_Phi*params.dt;
            T_next[x + y*m] = input.T + solved.dt_T*params.dt;
        });
    }
}


void explicit_solver_rk4_dt(Explicit_State* out, Explicit_State base_state, Real blend_factor, Explicit_State half_state, Allen_Cahn_Params params)
{
    int m = params.m;
    int n = params.n;
    Real* out_F = out->F;
    Real* out_U = out->U;

    //If blend fatcor == 0 no point in loading extra memory...
    if(blend_factor == 0)
    {
        cuda_for_2D(0, 0, params.m, params.n, [=]SHARED(int x, int y){
            Explicit_Solve_In input_base = explicit_solve_in_mod(base_state.F, base_state.U, x, y, n, m);
            Explicit_Solve_Result solved = allen_cahn_explicit_solve(input_base, params, NULL);

            out_F[x + y*m] = solved.dt_Phi;
            out_U[x + y*m] = solved.dt_T;
        });
    }
    else
    {
        cuda_for_2D(0, 0, params.m, params.n, [=]SHARED(int x, int y){
            Explicit_Solve_In input_base = explicit_solve_in_mod(base_state.F, base_state.U, x, y, n, m);
            Explicit_Solve_In input_half = explicit_solve_in_mod(half_state.F, half_state.U, x, y, n, m);
            Explicit_Solve_In input_blend = {0};

            for(int i = 0; i < sizeof(input_blend.vals) / sizeof(input_blend.vals[0]); i++)
                input_blend.vals[i] = input_base.vals[i] + blend_factor*input_half.vals[i];

            Explicit_Solve_Result solved = allen_cahn_explicit_solve(input_blend, params, NULL);

            out_F[x + y*m] = solved.dt_Phi;
            out_U[x + y*m] = solved.dt_T;
        });
    }
}

#if 0
static double runge_kutta4(double t_ini, double x_ini, double T, double tau, RK4_Func f, void* context)
{
    double x = x_ini;
    for (double t = t_ini; t <= T; t += tau)
    {
        double k1 = tau*f(t, x, context);
        double k2 = tau*f(t + 0.5f*tau, x + 0.5f*k1, context);
        double k3 = tau*f(t + 0.5f*tau, x + 0.5f*k2, context);
        double k4 = tau*f(t + tau, x + k3, context);
  
        double x_next = x + (k1 + 2*k2 + 2*k3 + k4)/6;
        x = x_next;  
    }
  
    return x;
}
#endif

extern "C" void explicit_solver_rk4_step(Explicit_Solver* solver, Explicit_State state, Explicit_State* next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
{
    Explicit_State empty = {0};
    static Explicit_State steps[4] = {0};
    if(steps[0].m != params.m || steps[0].n != params.n)
    {
        for(int i = 0; i < 4; i++)
            explicit_state_resize(&steps[i], params.n, params.n);
    }

    Explicit_State k1 = steps[0];
    Explicit_State k2 = steps[1];
    Explicit_State k3 = steps[2];
    Explicit_State k4 = steps[3];

    Real dt = params.dt;
    explicit_solver_rk4_dt(&k1, state, dt * 0, empty, params);
    explicit_solver_rk4_dt(&k2, state, dt * 0.5, k1, params);
    explicit_solver_rk4_dt(&k3, state, dt * 0.5, k2, params);
    explicit_solver_rk4_dt(&k4, state, dt * 1, k3, params);

    //@TODO: async and explicit sync.
    Real* out_F = next_state->F;
    Real* out_U = next_state->U;
    cuda_for(0, params.n*params.m, [=]SHARED(int i){
        // out_F[i] =  state.F[i] + k1.F[i]*dt;
        // out_U[i] =  state.U[i] + k1.F[i]*dt;
        out_F[i] =  state.F[i] + dt/6*(k1.F[i] + 2*k2.F[i] + 2*k3.F[i] + k4.F[i]);
        out_U[i] =  state.U[i] + dt/6*(k1.U[i] + 2*k2.U[i] + 2*k3.U[i] + k4.U[i]);
    });
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
    Real* scale;
    Real X;
    Real Y;
    Real C_minus_one;

    int m;
    int n;
};

void* cross_matrix_vector_alloced(Real* vector, int n, int m)
{
    if(vector == NULL)
        return NULL;
    (void) n;
    return vector - m;
}

Real* cross_matrix_vector_padded(void* alloced, int n, int m)
{
    if(alloced == NULL)
        return (Real*) NULL;
    (void) n;
    return (Real*) alloced + m;
}

void cross_matrix_vector_pad(Real* vector, int n, int m)
{
    CUDA_DEBUG_TEST(cudaMemset(vector - m, 0, sizeof(Real)*m));
    CUDA_DEBUG_TEST(cudaMemset(vector + n*m, 0, sizeof(Real)*m));
}

Real* cross_matrix_vector_realloc(Real* vector, int n, int m, int old_n, int old_m)
{
    int new_size = 2*m + n*m;
    int old_size = 2*old_m + old_n*old_m;
    void* old = cross_matrix_vector_alloced(vector, n, m);
    void* new_ = cuda_realloc(old, (size_t) new_size * sizeof(Real), (size_t) old_size * sizeof(Real), REALLOC_ZERO);
    return cross_matrix_vector_padded(new_, n, m);
}


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

void anisotrophy_matrix_multiply(Real* out, const void* _A, const Real* x, int N)
{
    Anisotrophy_Matrix A = * (Anisotrophy_Matrix*)_A;
    int m = A.m;
    cuda_for(0, N, [=]SHARED(int i){
        Real s = A.scale[i];
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

    //@NOTE: Evil programmer doing evil programming practices
    static int static_N = 0;
    static Real* _r = NULL;
    static Real* _p = NULL;
    static Real* _Ap = NULL;
    if(static_N < N)
    {
        cuda_realloc_in_place((void**) &_r, N*sizeof(Real), static_N*sizeof(Real), 0);
        cuda_realloc_in_place((void**) &_p, N*sizeof(Real), static_N*sizeof(Real), 0);
        cuda_realloc_in_place((void**) &_Ap, N*sizeof(Real), static_N*sizeof(Real), 0);
        static_N = N;
    }
   
    //NVCC seams to struggle with statics in device code
    // (is probably passing them by global reference or something)
    Real* r = _r;
    Real* p = _p;
    Real* Ap = _Ap;

    //@TODO: async + explicit sync
    //@TODO: This is broken since r assumes x to be zero the first iter. Make it not.
    // if(params.initial_value_or_null)
        // CUDA_DEBUG_TEST(cudaMemcpy(x, params.initial_value_or_null, sizeof(Real)*N, cudaMemcpyDeviceToDevice));
    // else
        CUDA_DEBUG_TEST(cudaMemset(x, 0, sizeof(Real)*N));
    CUDA_DEBUG_TEST(cudaMemcpy(r, b, sizeof(Real)*N, cudaMemcpyDeviceToDevice));
    CUDA_DEBUG_TEST(cudaMemcpy(p, b, sizeof(Real)*N, cudaMemcpyDeviceToDevice));

    Real r_dot_r = vector_dot_product(r, r, N);
    int iter = 0;
    for(; iter < params.max_iters; iter++)
    {
        matrix_mul_func(Ap, A, p, N);
        
        Real p_dot_Ap = vector_dot_product(p, Ap, N);
        Real alpha = r_dot_r / MAX(p_dot_Ap, params.epsilon);
        
        //@TODO: split into two separate kernels and do async
        //@TODO: add option for capping the values for x in a given range.
        //       this can potentially increase convergence and make the reuslts more
        //       accurate (if the range is correct) 
        cuda_for(0, N, [=]SHARED(int i){
            x[i] = x[i] + alpha*p[i];
            r[i] = r[i] - alpha*Ap[i];
        });

        Real r_dot_r_new = vector_dot_product(r, r, N);
        if(r_dot_r_new/N < params.tolerance*params.tolerance)
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

    Conjugate_Gardient_Convergence out = {0};
    out.iters = iter;
    out.converged = iter != params.max_iters;
    out.error = sqrt(r_dot_r/N);
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

Real vector_get_dist_norm(const Real* a, const Real* b, int N)
{
    static Real* temp = NULL;
    static int temp_size = 0;
    if(temp_size < N)
    {
        cuda_realloc_in_place((void**) &temp, N*sizeof(Real), temp_size*sizeof(Real), 0);
        temp_size = N;
    }

    Real* t = temp; //Needed for lambda to work (they capture statics differently)
    cuda_for(0, N, [=]SHARED(int i){
        t[i] = a[i] - b[i];
    });

    Real temp_dot_temp = vector_dot_product(temp, temp, N);
    Real error = sqrt(temp_dot_temp/N);
    return error;
}

bool vector_is_near(const Real* a, const Real* b, Real epsilon, int N)
{
    return vector_get_dist_norm(a, b, N) < epsilon;
}

int semi_implicit_solver_resize(Semi_Implicit_Solver* solver, int n, int m)
{
    if(solver->m != m || solver->n != n)
    {
        //Big evil programming practices because we are cool and we know
        // what we are doing and dont care much about what others have to
        // say
        Real** debug_maps = (Real**) (void*) &solver->debug_maps;
        for(int i = 0; i < sizeof(solver->debug_maps) / sizeof(Real); i++)
            debug_maps[i] = cross_matrix_vector_realloc(debug_maps[i], n, m, solver->n, solver->m);

        Real** maps = (Real**) (void*) &solver->maps;
        for(int i = 0; i < sizeof(solver->maps) / sizeof(Real); i++)
            maps[i] = cross_matrix_vector_realloc(maps[i], n, m, solver->n, solver->m);

        solver->m = m;
        solver->n = n;
    }

    return SEMI_IMPLICIT_SOLVER_REQUIRED_HISTORY;
}

void semi_implicit_state_resize(Semi_Implicit_State* state, int n, int m)
{
    if(state->m != m || state->n != n)
    {
        state->F = cross_matrix_vector_realloc(state->F, n, m, state->n, state->m);
        state->U = cross_matrix_vector_realloc(state->U, n, m, state->n, state->m);

        state->m = m;
        state->n = n;
    }
}

extern "C" void semi_implicit_solver_step(Semi_Implicit_Solver* solver, Semi_Implicit_State state, Semi_Implicit_State next_state, Allen_Cahn_Params params, size_t iter, bool do_debug)
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
    
    Real* F_next = next_state.F;
    Real* U_next = next_state.U;

    Real* F = state.F;
    Real* U = state.U;
    
    Real* b_F = solver->maps.b_F;
    Real* b_U = solver->maps.b_U;

    Anisotrophy_Matrix A_F = {0};
    A_F.scale = solver->maps.scale;
    A_F.C_minus_one = 2*dt/(alpha*dx*dx) + 2*dt/(alpha*dy*dy);
    A_F.X = -dt/(alpha*dx*dx);
    A_F.Y = -dt/(alpha*dy*dy);
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

    cuda_for_2D(0, 0, m, n, [=]SHARED(int x, int y){
        Real T = U[x + y*m];
        Real Phi = F[x + y*m];

        Real Phi_U = *at_mod(F, x, y + 1, m, n);
        Real Phi_D = *at_mod(F, x, y - 1, m, n);
        Real Phi_R = *at_mod(F, x + 1, y, m, n);
        Real Phi_L = *at_mod(F, x - 1, y, m, n);

        Real grad_Phi_x = dy*(Phi_R - Phi_L);
        Real grad_Phi_y = dx*(Phi_U - Phi_D);
        Real grad_Phi_norm = hypotf(grad_Phi_x, grad_Phi_y);
 
        Real g_theta = 1;
        {
            Real theta = atan2(grad_Phi_y, grad_Phi_x);
            g_theta = 1.0f - S*cosf(m0*theta + theta0);
        }

        // g_theta = 1;

        Real f = g_theta*a*f0(Phi) - b*xi*xi*beta*(T - Tm)*grad_Phi_norm/(2*mK);
        A_F.scale[x+y*m] = g_theta;
        b_F[x + y*m] = Phi + dt/(xi*xi*alpha)*f;
    });

    Conjugate_Gardient_Params solver_params = {0};
    solver_params.epsilon = (Real) 1.0e-12;
    solver_params.tolerance = (Real) 1.0e-9;
    solver_params.max_iters = 100;
    solver_params.initial_value_or_null = F;

    //Solve A_F*F_next = b_F
    Conjugate_Gardient_Convergence F_converged = conjugate_gradient_solve(&A_F, F_next, b_F, N, anisotrophy_matrix_multiply, &solver_params);
    printf("%lli F %s in %i iters with error %lf\n", (long long) iter, F_converged.converged ? "converged" : "diverged", F_converged.iters, F_converged.error);

    //Clamp in valid range. This should reduce the error even more
    // cuda_for(0, N, [=]SHARED(int i){
    //     F_next[i] = MAX(MIN(F_next[i], 1), 0);
    // });

    //Calculate b_U
    cuda_for_2D(0, 0, m, n, [=]SHARED(int x, int y){
        Real T = *at_mod(U, x, y, n, m);
        Real Phi = *at_mod(F, x, y, n, m);
        Real Phi_next = *at_mod(F_next, x, y, n, m);

        b_U[x + y*m] = T + L*(Phi_next - Phi);
    });

    solver_params.initial_value_or_null = U;

    //Solve A_U*U_next = b_U
    Conjugate_Gardient_Convergence U_converged = conjugate_gradient_solve(&A_U, U_next, b_U, N, cross_matrix_static_multiply, &solver_params);
    printf("%lli U %s in %i iters with error %lf\n", (long long) iter, U_converged.converged ? "converged" : "diverged", U_converged.iters, U_converged.error);

    if(do_debug)
    {
        Real* AfF = solver->debug_maps.AfF;
        Real* AuU = solver->debug_maps.AuU;
        //Back test
        if(1)
        {
            anisotrophy_matrix_multiply(AfF, &A_F, F_next, N);
            cross_matrix_static_multiply(AuU, &A_U, U_next, N);

            Real back_error_F = vector_get_dist_norm(AfF, b_F, N);
            Real back_error_U = vector_get_dist_norm(AuU, b_U, N);
            printf("F:" REAL_FMT " U:" REAL_FMT " Epsilon:" REAL_FMT "\n", back_error_F, back_error_U, solver_params.tolerance*2);
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

void semi_implicit_solver_get_maps(Semi_Implicit_Solver* solver, Semi_Implicit_State state, Sim_Map* maps, int map_count)
{
    int __map_i = 0;
    memset(maps, 0, sizeof maps[0] * map_count);
    ASSIGN_MAP_NAMED(state.F, "Phi");            
    ASSIGN_MAP_NAMED(state.U, "T");            
    ASSIGN_MAP_NAMED(solver->maps.b_F, "b_F");           
    ASSIGN_MAP_NAMED(solver->debug_maps.AfF, "AfF");           
    ASSIGN_MAP_NAMED(solver->maps.b_U, "b_U");           
    ASSIGN_MAP_NAMED(solver->debug_maps.AuU, "AuU");           
    ASSIGN_MAP_NAMED(solver->debug_maps.grad_phi, "grad_phi");           
    ASSIGN_MAP_NAMED(solver->debug_maps.grad_T, "grad_T");           
    ASSIGN_MAP_NAMED(solver->maps.scale, "Anisotrophy");  
}

struct Semi_Implicit_Coupled_Cross_Matrix {
    Anisotrophy_Matrix A_F; //A anisotrophy scaled cross matrix
    Real* B_U; //A changing diagonal 

    Cross_Matrix_Static A_U; //Static cross matrix
    Real B_F; //A single value diagonal

    int m;
    int n;
};

int semi_implicit_coupled_solver_resize(Semi_Implicit_Coupled_Solver* solver, int n, int m)
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

    return SEMI_IMPLICIT_SOLVER_REQUIRED_HISTORY;
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

    //@TODO: async + explicit sync

    //F equation
    cuda_for_2D(0, 0, m, n, [=]SHARED(int x, int y){
        int i = x + y*m;
        Real s = A.A_F.scale[i];
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
    A_F.scale = aniso;
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


extern "C" int  sim_solver_reinit(Sim_Solver* solver, Solver_Type type, int n, int m)
{
    if(solver->type != type && solver->type != SOLVER_TYPE_NONE)
        sim_solver_reinit(solver, solver->type, 0, 0);

    int out = 0;
    switch(type) {
        case SOLVER_TYPE_NONE: {
            n = 0;
            m = 0;
        } break;

        case SOLVER_TYPE_EXPLICIT: {
            out = explicit_solver_resize(&solver->expli, n, m);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT: {
            out = semi_implicit_solver_resize(&solver->impli, n, m);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: {
            out = semi_implicit_coupled_solver_resize(&solver->impli_coupled, n, m);
        } break;

        default: {
            assert(false);
        }
    };

    solver->type = type;
    solver->m = m;
    solver->n = n;
    return out;
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

        case SOLVER_TYPE_EXPLICIT: {
            explicit_state_resize(&states->expli, n, m);
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT: {
            semi_implicit_state_resize(&states->impli, n, m);
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


void _switch(Solver_Type type)
{
    switch(type) {
        case SOLVER_TYPE_NONE: {

        } break;

        case SOLVER_TYPE_EXPLICIT: {
            
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT: {

        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: {

        } break;

        default: assert(false);
    };
}


extern "C" void sim_solver_step(Sim_Solver* solver, Sim_State* states, int states_count, int iter, Allen_Cahn_Params params, bool do_debug)
{

    switch(solver->type) {
        case SOLVER_TYPE_NONE: {
            // nothing
        } break;

        case SOLVER_TYPE_EXPLICIT: {
            if(states_count < EXPLICIT_SOLVER_REQUIRED_HISTORY)
                printf("explicit solver requires bigger history\n");
            else
            {
                Sim_State state = states[iter % states_count];
                Sim_State next_state = states[(iter + 1) % states_count];
                if(state.type != SOLVER_TYPE_EXPLICIT || next_state.type != SOLVER_TYPE_EXPLICIT)
                    printf("not matching state provided to explicit solver\n");
                else
                {
                    explicit_solver_rk4_step(&solver->expli, state.expli, &next_state.expli, params, iter, do_debug);
                    // explicit_solver_newton_step(&solver->expli, state.expli, next_state.expli, params, iter, do_debug);
                }
            }
        } break;

        case SOLVER_TYPE_SEMI_IMPLICIT: {
            if(states_count < EXPLICIT_SOLVER_REQUIRED_HISTORY)
                printf("explicit solver requires bigger history");
            else
            {
                Sim_State state = states[iter % states_count];
                Sim_State next_state = states[(iter + 1) % states_count];
                if(state.type != SOLVER_TYPE_SEMI_IMPLICIT || next_state.type != SOLVER_TYPE_SEMI_IMPLICIT)
                    printf("not matching state provided to semi implicit solver\n");
                else
                    semi_implicit_solver_step(&solver->impli, state.impli, next_state.impli, params, iter, do_debug);
            }
        } break;


        case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: {
            if(states_count < EXPLICIT_SOLVER_REQUIRED_HISTORY)
                printf("explicit solver requires bigger history");
            else
            {
                Sim_State state = states[iter % states_count];
                Sim_State next_state = states[(iter + 1) % states_count];
                if(state.type != SOLVER_TYPE_SEMI_IMPLICIT_COUPLED || next_state.type != SOLVER_TYPE_SEMI_IMPLICIT_COUPLED)
                    printf("not matching state provided to semi implicit solver\n");
                else
                    semi_implicit_coupled_solver_step(&solver->impli_coupled, state.impli_coupled, next_state.impli_coupled, params, iter, do_debug);
            }
        } break;

        default: assert(false);
    };
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

        case SOLVER_TYPE_EXPLICIT: {
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