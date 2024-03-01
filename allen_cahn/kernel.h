#pragma once
#define DO_FLOATS 

#include <stddef.h>

#ifdef DO_FLOATS
    typedef float Real;
    #define REAL_FMT "%f"
    #define REAL_FMT_LOW_PREC "%.3f"
#else
    typedef double Real;
    #define REAL_FMT "%lf"
    #define REAL_FMT_LOW_PREC "%.3lf"
#endif

typedef struct Allen_Cahn_Params{
    int mesh_size_x;
    int mesh_size_y;
    Real L0; //simulation region size in real units

    Real dt; //time step
    Real L;  //latent heat
    Real xi; //boundary thickness
    Real a;  //
    Real b;
    Real alpha;
    Real beta;
    Real Tm; //melting point
    Real Tinit; //currenlty unsused

    Real S; //anisotrophy strength
    Real m; //anisotrophy frequency (?)
    Real theta0; //anisotrophy orientation
    bool do_anisotropy;
} Allen_Cahn_Params;

#define ALLEN_CAHN_HISTORY 4
#define ALLEN_CAHN_DEBUG_MAPS 8
#define ALLEN_CAHN_DEBUG_MAP_NAME_LEN 32

typedef struct Debug_State {
    Real* maps[ALLEN_CAHN_DEBUG_MAPS];
    char names[ALLEN_CAHN_DEBUG_MAPS][ALLEN_CAHN_DEBUG_MAP_NAME_LEN + 1];

    int m;
    int n;
} Debug_State;

typedef struct Explicit_State{
    Real* F[ALLEN_CAHN_HISTORY];
    Real* U[ALLEN_CAHN_HISTORY];

    int m;
    int n;
} Explicit_State;

#define SEMI_AUX 4

struct Semi_Implicit_State {
    Real* F[ALLEN_CAHN_HISTORY];
    Real* U[ALLEN_CAHN_HISTORY];

    Real* b_F[ALLEN_CAHN_HISTORY];
    Real* b_U[ALLEN_CAHN_HISTORY];

    Real* aux[SEMI_AUX];

    int m;
    int n;
};

typedef enum {
    MODIFY_UPLOAD,
    MODIFY_DOWNLOAD,
} Solver_Modify;

extern "C" void explicit_solver_step(Explicit_State* state, Debug_State* debug_state_or_null, Allen_Cahn_Params params, size_t iter);
extern "C" void semi_implicit_solver_step(Semi_Implicit_State* state, Debug_State* debug_state_or_null, Allen_Cahn_Params params, size_t iter);

extern "C" void explicit_state_resize(Explicit_State* state, int n, int m);
extern "C" void semi_implicit_state_resize(Semi_Implicit_State* state, int n, int m);
extern "C" void debug_state_resize(Debug_State* state, int n, int m);

//helper conversion functions for opengl compatibility (opengl cannot work with double values)
extern "C" void device_float_modify(Real* device_memory, float* host_memory, size_t size, Solver_Modify modify);
extern "C" void device_modify(void* device_memory, void* host_memory, size_t size, Solver_Modify modify);