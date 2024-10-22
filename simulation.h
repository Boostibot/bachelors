#pragma once

// This file defines a C interface between the cuda implementation and outside world.

#include <stddef.h>
#include <stdlib.h>
#include "defines.h"

#define COMPILE_GRAPHICS  
    // #define COMPILE_NETCDF  
    #ifndef CUSTOM_SETTINGS
    #define COMPILE_BENCHMARKS
    #define COMPILE_TESTS
    #define COMPILE_SIMULATION
    #define COMPILE_THRUST
    #define COMPILE_NOISE
    // #define COMPILE_EXACT //do not use slightly broken!
    // #define USE_FLOATS
#endif

#ifdef USE_FLOATS
    typedef float Real;
#else
    typedef double Real;
#endif

typedef enum Sim_Boundary_Type {
    BOUNDARY_PERIODIC = 0, 
    BOUNDARY_DIRICHLET_ZERO, 
    BOUNDARY_NEUMANN_ZERO, 
    BOUNDARY_ENUM_COUNT, 
} Sim_Boundary_Type;

typedef enum Sim_Solver_Type{
    SOLVER_TYPE_NONE = 0,
    SOLVER_TYPE_EXPLICIT_EULER,
    SOLVER_TYPE_EXPLICIT_RK4,
    SOLVER_TYPE_EXPLICIT_RK4_ADAPTIVE,
    SOLVER_TYPE_SEMI_IMPLICIT,
    SOLVER_TYPE_EXACT,
    SOLVER_TYPE_ENUM_COUNT,
} Sim_Solver_Type;

typedef int64_t i64;

typedef struct Sim_Map {
    Real* data;
    int nx;
    int ny;
    char name[32];
    i64 iter; //iteration at which this mapped was touched
    double time;
    bool is_debug;
} Sim_Map;

enum {MAX_STEP_RESIDUALS = 20};
typedef struct Sim_Stats {
    double time;
    i64 iter;
    
    int Phi_iters;
    int Phi_ellapsed_time; //TODO
    float T_iters;
    float T_ellapsed_time;

    float T_delta_L1;
    float T_delta_L2;
    float T_delta_max;
    float T_delta_min;

    float Phi_delta_L1;
    float Phi_delta_L2;
    float Phi_delta_max;
    float Phi_delta_min;

    float step_res_L1[MAX_STEP_RESIDUALS];
    float step_res_L2[MAX_STEP_RESIDUALS];
    float step_res_max[MAX_STEP_RESIDUALS];
    float step_res_min[MAX_STEP_RESIDUALS];
    int step_res_count;
} Sim_Stats;

typedef struct Sim_Params{
    Sim_Solver_Type solver;
    int nx;
    int ny;

    double time;
    i64 iter;

    Sim_Boundary_Type T_boundary;
    Sim_Boundary_Type Phi_boundary;

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
    double min_dt;

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
    bool do_prints;

    Sim_Stats* stats;
    Sim_Map* temp_maps;
    int temp_map_count;
} Sim_Params;

void sim_realloc(Sim_Map* map, const char* name, int nx, int ny, double time, i64 iter);
double sim_step(Sim_Map F, Sim_Map U, Sim_Map* next_F, Sim_Map* next_U, Sim_Params params);

typedef enum {
    MODIFY_UPLOAD,
    MODIFY_DOWNLOAD,
} Sim_Modify;

extern "C" void sim_modify(void* device_memory, void* host_memory, size_t size, Sim_Modify modify);
extern "C" void sim_modify_float(Real* device_memory, float* host_memory, size_t count, Sim_Modify modify);
extern "C" void sim_modify_double(Real* device_memory, double* host_memory, size_t count, Sim_Modify modify);

extern "C" bool run_tests();
extern "C" bool run_benchmarks(int N);

static const char* boundary_type_to_cstring(Sim_Boundary_Type type)
{
    switch(type)
    {
        default: return "unknown";
        case BOUNDARY_PERIODIC: return "periodic";
        case BOUNDARY_DIRICHLET_ZERO: return "dirichlet";
        case BOUNDARY_NEUMANN_ZERO: return "neumann";
    }
}

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
        case SOLVER_TYPE_EXACT: return "exact"; 
    }
}