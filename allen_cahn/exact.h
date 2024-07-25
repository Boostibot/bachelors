#include "simulation.h"
#include <math.h>

struct Exact_Params 
{
    Real lambda;
    Real R_ini;
    Real epsilon;
    Real xi;
};

#ifdef __CUDACC__
    #define EXACT_API static __device__ __host__
#else
    #define EXACT_API static
#endif

EXACT_API Real exact_R(Real t, Exact_Params params)
{
    return sqrt(params.R_ini*params.R_ini + 2*params.lambda*t);
}

EXACT_API Real exact_U(Real t, Real Rt, Exact_Params params)
{    
    return -params.epsilon*(params.lambda + 2)/Rt;
}

EXACT_API Real exact_T(Real s, Exact_Params params)
{
    Real l = params.lambda;
    Real pi = (Real) 3.14159265359;
    Real sqrtl2 = sqrt(l/2);
    Real integral = exp(-l/2) - 1/s*exp(-l/2*s*s) 
        + sqrtl2*pi*(erf(sqrtl2) - erf(s*sqrtl2));
    return -l*exp(l/2)*integral;
}

EXACT_API Real exact_fu(Real t, Exact_Params params)
{
    Real Rt = exact_R(t, params);
    Real l = params.lambda;
    return params.epsilon*l*(l + 2)/(Rt*Rt*Rt);
}

EXACT_API Real exact_u(Real t, Real r, Exact_Params params)
{
    Real Rt = exact_R(t, params);
    Real Ut = exact_U(t, Rt, params);
    Real ut = Ut;
    if(r > Rt)
        ut += exact_T(r/Rt, params);

    return ut;
}

EXACT_API Real exact_phi(Real t, Real r, Exact_Params params)
{
    return r <= exact_R(t, params) ? (Real) 1 : (Real) 0; 
}

EXACT_API Real smootherstep(Real edge0, Real edge1, Real x) {
  // Scale, and clamp x to 0..1 range
    x = (x - edge0) / (edge1 - edge0);
    if(x > 1)
        x = 1;
    if(x < 0)
        x = 0;
    return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
}

EXACT_API Real exact_corresponing_phi_ini(Real r, Exact_Params params)
{
    Real p_ini = 0;
    // Real lo = params.R_ini - params.xi/2; 
    // Real hi = params.R_ini + params.xi/2; 
    Real fade = 14; //TODO!
    Real lo = params.R_ini - params.xi*fade/2; 
    Real hi = params.R_ini + params.xi*fade/2; 

    if(r < lo)
        p_ini = 1;
    else if(r > hi)
        p_ini = 0;
    else 
        // p_ini = (params.R_ini - r)/params.xi + 0.5;
        // p_ini = 1 - (r - lo)/(hi - lo);
        p_ini = 1 - smootherstep(lo, hi, r);

    return p_ini;
}

EXACT_API Exact_Params get_static_exact_params(Sim_Params params)
{
    Exact_Params exact_params = {0};
    exact_params.xi = params.xi;
    exact_params.epsilon = 0.001;
    exact_params.lambda = 0.5;
    exact_params.R_ini = params.exact_R_ini;
    return exact_params;
}