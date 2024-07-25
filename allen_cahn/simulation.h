#pragma once

// This file defines a C interface between the cuda implementation and outside world.
//
// We try to be as transparent as possible while still allowing the user application to
//   not care about the solver type. We use tagged unions instead of inheritence like approaches.
//
// We try to decouple the concept of state (T, Phi, ... inside Sim_State) from the auxiliary data 
//   needed to perform a sngle simulation step  (Matrices, vectors of right hands side ... 
//   inside Sim_Solver). This provides us with a lot extra flexibility we wouldnt be able to get with
//   the usual Solver calls that manages everything. We are free to create as much history state as we 
//   want, replay a particular step in the simulation, change solvers mid way through algorhitm... .
//   
// The retrieval of all maps used is done through the Sim_Map interface which returns a raw pointer
//   int the cuda DEVICE memory along with name identifying it. This makes it extremely easy to iterate
//   throught the maps and display/serialize only the desired one in solver agnostic way. We use this
//   to quickly define new debug maps without having to change almost anything.

#include <stddef.h>

#define COMPILE_GRAPHICS  
// #define COMPILE_NETCDF  
#ifndef CUSTOM_SETTINGS
// #define COMPILE_BENCHMARKS
// #define COMPILE_TESTS
#define COMPILE_SIMULATION
// #define COMPILE_THRUST
#define COMPILE_NOISE

#define USE_CUSTOM_REDUCE
// #define USE_TILED_FOR 
// #define USE_FLOATS
#endif

#ifdef USE_FLOATS
    typedef float Real;
#else
    typedef double Real;
#endif

enum Sim_Boundary_Type {
    BOUNDARY_PERIODIC = 0, 
    BOUNDARY_DIRICHLET_ZERO, 
    BOUNDARY_NEUMANN_ZERO, 
};

typedef struct Sim_Params{
    int nx;
    int ny;

    double L0; 

    double dt;
    double L; 
    double xi;
    double a;  
    double b;
    double alpha;
    double beta;
    double gamma; 
    double Tm; 
    double exact_R_ini; //TODO: remove

    double S; 
    double m0; 
    double theta0;

    double T_tolerance;
    double Phi_tolerance;
    double corrector_tolerance;
    
    int T_max_iters;
    int Phi_max_iters;
    int corrector_max_iters;

    bool do_corrector_loop;
    bool do_corrector_guess;

    bool do_debug;
    bool do_stats;
    bool do_stats_step_residual;
    bool do_exact;

    Sim_Boundary_Type T_boundary;
    Sim_Boundary_Type Phi_boundary;
} Sim_Params;

enum {MAX_STEP_RESIDUALS = 20};

typedef struct Float_Array {
    double* data;
    size_t len;
    size_t capacity;
} Float_Array;

typedef struct Sim_Stats_Vectors{
    Float_Array time;
    Float_Array iter;
    
    Float_Array T_delta_L1;
    Float_Array T_delta_L2;
    Float_Array T_delta_max;
    Float_Array T_delta_min;

    Float_Array phi_delta_L1;
    Float_Array phi_delta_L2;
    Float_Array phi_delta_max;
    Float_Array phi_delta_min;

    Float_Array phi_iters;
    Float_Array T_iters;
    Float_Array phi_ellapsed_time; //TODO
    Float_Array T_ellapsed_time;

    Float_Array step_res_L1[MAX_STEP_RESIDUALS];
    Float_Array step_res_L2[MAX_STEP_RESIDUALS];
    Float_Array step_res_max[MAX_STEP_RESIDUALS];
    Float_Array step_res_min[MAX_STEP_RESIDUALS];
} Sim_Stats_Vectors;

typedef struct Sim_Stats {
    Sim_Stats_Vectors vectors;
    size_t step_res_count;
} Sim_Stats;

typedef struct Sim_Step_Info {
    size_t iter;
    double sim_time;
} Sim_Step_Info;

#include <stdlib.h>
static void float_array_reserve(Float_Array* array, size_t to_capaciy)
{
    if(array->capacity < to_capaciy)
    {
        array->capacity = array->capacity*3/2 + 8;
        array->data = (double*) realloc(array->data, array->capacity*sizeof(double));
    }
} 

static void float_array_push(Float_Array* array, double val)
{
    float_array_reserve(array, array->len + 1);
    array->data[array->len ++] = val;
}

static void float_array_clear(Float_Array* array)
{
    array->len = 0;
}

typedef enum Sim_Solver_Type{
    SOLVER_TYPE_NONE = 0,
    SOLVER_TYPE_EXPLICIT_EULER,
    SOLVER_TYPE_EXPLICIT_RK4,
    SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE,
    SOLVER_TYPE_SEMI_IMPLICIT,
    SOLVER_TYPE_SEMI_IMPLICIT_COUPLED,
    SOLVER_TYPE_EXACT,

    SOLVER_TYPE_ENUM_COUNT,
} Sim_Solver_Type;

static const char* solver_type_to_cstring(Sim_Solver_Type type)
{
    switch(type)
    {
        default: return "unknown";
        case SOLVER_TYPE_NONE: return "none";
        case SOLVER_TYPE_EXPLICIT_EULER: return "explicit";
        case SOLVER_TYPE_EXPLICIT_RK4: return "explicit-rk4";
        case SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE: return "explicit-rk4-adaptive";
        case SOLVER_TYPE_SEMI_IMPLICIT: return "semi-implicit";
        case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: return "semi-implicit-coupled"; 
        case SOLVER_TYPE_EXACT: return "exact"; 
    }
}

static int solver_type_required_history(Sim_Solver_Type type) 
{
    switch(type)
    {
        default:
        case SOLVER_TYPE_NONE: return 0;
        case SOLVER_TYPE_EXPLICIT_EULER: return 2;
        case SOLVER_TYPE_EXPLICIT_RK4: return 2;
        case SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE: return 2;
        case SOLVER_TYPE_SEMI_IMPLICIT: return 2;
        case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: return 2;
        case SOLVER_TYPE_EXACT: return 1;
    }
}

#define SIM_HISTORY_MAX 32
#define SIM_MAPS_MAX    64

//Explicit
typedef struct Explicit_Solver {
    struct {
        Real* grad_phi;
        Real* grad_T;
        Real* reaction;
        Real* aniso_factor;
        Real* step_residual;
        Real* perlin;
        Real* simplex;
    } debug_maps;

    void* pad;
    int nx;
    int ny;
} Explicit_Solver;

typedef struct Explicit_State{
    Real* F;
    Real* U;

    int nx;
    int ny;
} Explicit_State;

typedef Explicit_State Explicit_RK4_Solver;
typedef Explicit_State Explicit_RK4_Adaptive_Solver;
typedef Explicit_State Semi_Implicit_State;
typedef Explicit_State Exact_State;

typedef struct Semi_Implicit_Solver {
    struct {
        Real* b_F;
        Real* b_U;
        Real* anisotrophy;
    } maps;

    struct {
        Real* grad_phi;
        Real* grad_T;
        Real* aniso_factor;
        Real* AfF;
        Real* AuU;
        Real* step_residuals[3];
    } debug_maps;

    int nx;
    int ny;
} Semi_Implicit_Solver;

//Semi implicit coupled
typedef struct Semi_Implicit_Coupled_State{
    Real* C;

    int nx;
    int ny;
} Semi_Implicit_Coupled_State;

typedef struct Semi_Implicit_Coupled_Solver {
    Real* b_C; //size 2N

    Real* aniso; //size N
    Real* B_U; //size N

    int nx;
    int ny;
} Semi_Implicit_Coupled_Solver;

//Polymorphic
typedef struct Sim_State {
    int nx;
    int ny;
    Sim_Solver_Type type;

    union {
        Exact_State exact;
        Explicit_State expli;
        Semi_Implicit_State impli;
        Semi_Implicit_Coupled_State impli_coupled;
    };
} Sim_State;

typedef struct Sim_Solver  {
    int nx;
    int ny;
    Sim_Solver_Type type;

    union {
        int exact;
        Explicit_Solver expli;
        Semi_Implicit_Solver impli;
        Semi_Implicit_Coupled_Solver impli_coupled;
    };
} Sim_Solver;

typedef struct Sim_Map {
    const char* name;
    Real* data;

    int nx;
    int ny;
} Sim_Map;

extern "C" void sim_solver_reinit(Sim_Solver* solver, Sim_Solver_Type type, int nx, int ny);
extern "C" void sim_states_reinit(Sim_State* states, int state_count, Sim_Solver_Type type, int nx, int ny);
extern "C" void sim_solver_get_maps(Sim_Solver* solver, Sim_State* states, int states_count, int iter, Sim_Map* maps, int map_count);
extern "C" double sim_solver_step(Sim_Solver* solver, Sim_State* states, int states_count, Sim_Step_Info info, Sim_Params params, Sim_Stats* stats_or_null);

typedef enum {
    MODIFY_UPLOAD,
    MODIFY_DOWNLOAD,
} Sim_Modify;

extern "C" void sim_modify(void* device_memory, void* host_memory, size_t size, Sim_Modify modify);
extern "C" void sim_modify_float(Real* device_memory, float* host_memory, size_t count, Sim_Modify modify);
extern "C" void sim_modify_double(Real* device_memory, double* host_memory, size_t count, Sim_Modify modify);

extern "C" bool run_tests();
extern "C" bool run_benchmarks(int N);

// solver -> init to some concrete solver
// solver has to have internal state for caching CACHING! CACHING! its all about caching! We dont even have to have a solver
// since we can just cache the needed data inside the function or better yet recapture them every time.
// But we have to have states and have to have views into the solver debug info. Idk man. Maybe I am really just attempting to combine
// things that shouldnt be combined and the best would be to leave the user code coupled to a particular solver since I cannot 
// figure out a seemless fitting interface.

#if 0
#include <string.h>
static void sim_example()
{
    //Construct by zero init
    Sim_Params params = {0};
    Sim_Solver solver = {0};
    Sim_State states[SIM_HISTORY_MAX] = {0};
    
    int nx = 20;
    int ny = 20;
    Sim_Solver_Type type = SOLVER_TYPE_EXPLICIT_EULER; 
    (void) type; //for some reason we get unreferenced variable warning here even though it clearly is used (?)
    
    //Init solver. It returns the MINIMAL number of states it requires
    // to work properly.
    sim_solver_reinit(&solver, type, nx, ny);
    int state_count = solver_type_required_history(type);
    //Init the states 
    sim_states_reinit(states, state_count, type, nx, ny);

    //Loop simulation
    for(int i = 0; i < 1000; i++)
    {
        //Do a step in the simulation
        Sim_Map maps[SIM_MAPS_MAX] = {0};
        sim_solver_step(&solver, states, state_count, i, params, NULL);

        //Get debug maps
        sim_solver_get_maps(&solver, states, state_count, i, maps, SIM_MAPS_MAX);
        for(int map_i = 0; map_i < SIM_MAPS_MAX; i++)
        {
            if(strcmp(maps[map_i].name, "T")) {
                float* visualisation_storage = NULL; //pretend this points to something
                sim_modify_float(maps[map_i].data, visualisation_storage, (size_t) (ny*nx), MODIFY_DOWNLOAD);

                //... print map inside visualisation_storage ...
            }
        }
    }

    //deinit solver and states by initing them to SOLVER_TYPE_NONE (nx, ny is ignored)
    sim_solver_reinit(&solver, SOLVER_TYPE_NONE, 0, 0);
    sim_states_reinit(states, state_count, SOLVER_TYPE_NONE, 0, 0);
}

#endif
