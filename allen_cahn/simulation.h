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

// #define DO_FLOATS 

#include <stddef.h>

#ifdef DO_FLOATS
    typedef float Real;
#else
    typedef double Real;
#endif

typedef struct Sim_Real_Vector {
    Real* data;
    int nx;
    int ny;
} Sim_Real_Vector; 

typedef struct Allen_Cahn_Params{
    int nx;
    int ny;

    double L0; //simulation region size in real units

    double dt; //time step
    double L;  //latent heat
    double xi; //boundary thickness
    double a;  //
    double b;
    double alpha;
    double beta;
    double Tm; //melting point
    double Tinit; //currenlty unsused
    double gamma; //Crank-Nicolson factor: 

    double S; //anisotrophy strength
    double m0; //anisotrophy frequency (?)
    double theta0; //anisotrophy orientation
    bool do_anisotropy;

    double T_tolerance;
    double Phi_tolerance;
    double corrector_tolerance;
    
    int T_max_iters;
    int Phi_max_iters;
    int corrector_max_iters;

    bool do_corrector_loop;
    bool do_corrector_guess;

    double noise_T_strength;
    double noise_T_min;      
    double noise_T_max;
    double noise_T_timestep; //Regenerates a new noise map ever 

    double noise_Phi_strength;
    double noise_Phi_min;      
    double noise_Phi_max;

    double noise_der_Phi_strength;
    double noise_der_Phi_min;      
    double noise_der_Phi_max;

    double noise_Phi_timestep; //Regenerates a new noise map ever 
    bool do_debug;
    bool do_noise;
    bool do_stats;
    bool do_stats_step_residual;
} Allen_Cahn_Params;

#define STEP_RESIDUALS_MAX 10
typedef struct Allen_Cahn_Stats{
    Real L2_step_residuals[10];
    Real Lmax_step_residuals[10];
    int step_residuals;

    int T_iters;
    int Phi_iters;

    Real T_error;
    Real Phi_errror;
} Allen_Cahn_Stats;

typedef enum Solver_Type{
    SOLVER_TYPE_NONE = 0,
    SOLVER_TYPE_EXPLICIT,
    SOLVER_TYPE_EXPLICIT_RK4,
    SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE,
    SOLVER_TYPE_SEMI_IMPLICIT,
    SOLVER_TYPE_SEMI_IMPLICIT_COUPLED,

    SOLVER_TYPE_ENUM_COUNT,
} Solver_Type;

static const char* solver_type_to_cstring(Solver_Type type)
{
    switch(type)
    {
        default: return "unknown";
        case SOLVER_TYPE_NONE: return "none";
        case SOLVER_TYPE_EXPLICIT: return "explicit";
        case SOLVER_TYPE_EXPLICIT_RK4: return "explicit-rk4";
        case SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE: return "explicit-rk4-adaptive";
        case SOLVER_TYPE_SEMI_IMPLICIT: return "semi-implicit";
        case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: return "semi-implicit-coupled"; 
    }
}

static int solver_type_required_history(Solver_Type type) 
{
    switch(type)
    {
        default:
        case SOLVER_TYPE_NONE: return 0;
        case SOLVER_TYPE_EXPLICIT: return 2;
        case SOLVER_TYPE_EXPLICIT_RK4: return 2;
        case SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE: return 2;
        case SOLVER_TYPE_SEMI_IMPLICIT: return 2;
        case SOLVER_TYPE_SEMI_IMPLICIT_COUPLED: return 2;
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
    } debug_maps;

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

//Semi implicit
typedef Explicit_State Semi_Implicit_State;

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
    Solver_Type type;

    union {
        Explicit_State expli;
        Semi_Implicit_State impli;
        Semi_Implicit_Coupled_State impli_coupled;
    };
} Sim_State;

typedef struct Sim_Solver  {
    int nx;
    int ny;
    Solver_Type type;

    union {
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

extern "C" void sim_solver_reinit(Sim_Solver* solver, Solver_Type type, int nx, int ny);
extern "C" void sim_states_reinit(Sim_State* states, int state_count, Solver_Type type, int nx, int ny);
extern "C" void sim_solver_get_maps(Sim_Solver* solver, Sim_State* states, int states_count, int iter, Sim_Map* maps, int map_count);
extern "C" double sim_solver_step(Sim_Solver* solver, Sim_State* states, int states_count, int iter, Allen_Cahn_Params params, Allen_Cahn_Stats* stats_or_null);

typedef enum {
    MODIFY_UPLOAD,
    MODIFY_DOWNLOAD,
} Sim_Modify;

extern "C" void sim_modify(void* device_memory, void* host_memory, size_t size, Sim_Modify modify);
extern "C" void sim_modify_float(Real* device_memory, float* host_memory, size_t size, Sim_Modify modify);
extern "C" void sim_modify_double(Real* device_memory, double* host_memory, size_t size, Sim_Modify modify);

extern "C" bool run_tests();
extern "C" bool run_benchmarks(int N);

// solver -> init to some concrete solver
// solver has to have internal state for caching CACHING! CACHING! its all about caching! We dont even have to have a solver
// since we can just cache the needed data inside the function or better yet recapture them every time.
// But we have to have states and have to have views into the solver debug info. Idk man. Maybe I am really just attempting to combine
// things that shouldnt be combined and the best would be to leave the user code coupled to a particular solver since I cannot 
// figure out a seemless fitting interface.

#if 1
#include <string.h>
static void sim_example()
{
    //Construct by zero init
    Allen_Cahn_Params params = {0};
    Sim_Solver solver = {0};
    Sim_State states[SIM_HISTORY_MAX] = {0};
    
    int nx = 20;
    int ny = 20;
    Solver_Type type = SOLVER_TYPE_EXPLICIT; 
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
