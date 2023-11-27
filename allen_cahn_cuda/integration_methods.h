#pragma once
#include <math.h>
#include "lib/defines.h"
#include "lib/log.h"

typedef f32 (*RK4_Func)(f32 t, const f32 x, void* context);

f32 runge_kutta4_var(f32 t_ini, f32 x_ini, f32 T, f32 tau_ini, f32 delta, RK4_Func f, void* context)
{
    (void) delta;
    f32 t = t_ini;
    f32 tau = tau_ini;
    f32 x = x_ini;

    f32 n = 1;
    for(;;)
    {
        bool last = false;
        if(fabsf(T - t) < fabs(tau))
        {
            tau = T - t;
            last = true;
        }
        else
        {
            last = false;
        }

        f32 k1 = f(t, x, context);
        f32 k2 = f(t + tau/3, x + tau/3*k1, context);
        f32 k3 = f(t + tau/3, x + tau/6*(k1 + k2), context);
        f32 k4 = f(t + tau/2, x + tau/8*(k1 + 3*k3), context);
        f32 k5 = f(t + tau/1, x + tau*(0.5f*k1 - 1.5f*k3 + 2*k4), context);
        
        f32 epsilon = 0;
        for(f32 i = 0; i < n; i++)
        {
            f32 curr = fabsf(0.2f*k1 - 0.9f*k3 + 0.8f*k4 - 0.1f*k5);
            epsilon = MAX(epsilon, curr);
        }

        //if(epsilon < delta)
        {
            x = x + tau*(1.0f/6*(k1 + k5) + 2.0f/3*k4);
            t = t + tau;

            if(last)
                break;

            if(epsilon == 0)
                continue;
        }

        //tau = powf(delta / epsilon, 0.2f)* 4.0f/5*tau;
    }

    return x;
}

f32 runge_kutta4(f32 t_ini, f32 x_ini, f32 T, f32 tau, RK4_Func f, void* context)
{
    f32 x = x_ini;
    for (f32 t = t_ini; t <= T; t += tau)
    {
        f32 k1 = tau*f(t, x, context);
        f32 k2 = tau*f(t + 0.5f*tau, x + 0.5f*k1, context);
        f32 k3 = tau*f(t + 0.5f*tau, x + 0.5f*k2, context);
        f32 k4 = tau*f(t + tau, x + k3, context);
  
        f32 x_next = x + (k1 + 2*k2 + 2*k3 + k4)/6;
        x = x_next;  
    }
  
    return x;
}

f32 euler(f32 t_ini, f32 x_ini, f32 T, f32 tau, RK4_Func f, void* context)
{
    f32 t = t_ini;
    f32 x = x_ini;

    for(; t <= T; )
    {
        f32 derx = f(t, x, context);
        f32 x_next = x + tau*derx;

        x = x_next;
        t += tau;
    }

    return x;
}


f32 semi_euler(f32 t_ini, f32 x_ini, f32 T, f32 tau, RK4_Func f, void* context)
{
    f32 t = t_ini;
    f32 x = x_ini;

    for(; t <= T; )
    {
        f32 derx = f(t + tau, x, context);
        f32 x_next = x + tau*derx;

        x = x_next;
        t += tau;
    }

    return x;
}

f32 rk4_func_x(f32 t, const f32 x, void* context)
{
    (void) context;
    (void) x;
    return t;
}

void compare_rk4()
{
    f32 t_ini = 0;
    f32 x_ini = 0;
    f32 tau = 0.001f;

    for(f32 T = 0; T < 5; T += 0.5f)
    {
        f32 euler_r = euler(t_ini, x_ini, T, tau, rk4_func_x, NULL);
        f32 semi_euler_r = semi_euler(t_ini, x_ini, T, tau, rk4_func_x, NULL);
        f32 rk4_var_r = runge_kutta4_var(t_ini, x_ini, T, tau, 0.01f, rk4_func_x, NULL);
        f32 rk4_r = runge_kutta4(t_ini, x_ini, T, tau, rk4_func_x, NULL);
        f32 exact_r = T*T/2;

        LOG_INFO("RK4", "T = %f", T);
        log_group_push();
            LOG_INFO("RK4", "exact:         %f", exact_r);
            LOG_INFO("RK4", "euler:         %f", euler_r);
            LOG_INFO("RK4", "semi_euler:    %f", semi_euler_r);
            LOG_INFO("RK4", "rk4_var_r:     %f", rk4_var_r);
            LOG_INFO("RK4", "rk4_r:         %f", rk4_r);
        log_group_pop();
    }
}