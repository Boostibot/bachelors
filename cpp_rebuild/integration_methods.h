#pragma once
#include <cmath>
#include "log.h"

typedef double (*RK4_Func)(double t, const double x, void* context);

static double runge_kutta4_var(double t_ini, double x_ini, double T, double tau_ini, double delta, RK4_Func f, void* context)
{
    (void) delta;
    double t = t_ini;
    double tau = tau_ini;
    double x = x_ini;

    double n = 1;
    for(;;)
    {
        bool last = false;
        if(fabs(T - t) < fabs(tau))
        {
            tau = T - t;
            last = true;
        }
        else
        {
            last = false;
        }

        double k1 = f(t, x, context);
        double k2 = f(t + tau/3, x + tau/3*k1, context);
        double k3 = f(t + tau/3, x + tau/6*(k1 + k2), context);
        double k4 = f(t + tau/2, x + tau/8*(k1 + 3*k3), context);
        double k5 = f(t + tau/1, x + tau*(0.5f*k1 - 1.5f*k3 + 2*k4), context);
        
        double epsilon = 0;
        for(double i = 0; i < n; i++)
        {
            double curr = fabs(0.2f*k1 - 0.9f*k3 + 0.8f*k4 - 0.1f*k5);
            epsilon = std::max(epsilon, curr);
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

static double euler(double t_ini, double x_ini, double T, double tau, RK4_Func f, void* context)
{
    double t = t_ini;
    double x = x_ini;

    for(; t <= T; )
    {
        double derx = f(t, x, context);
        double x_next = x + tau*derx;

        x = x_next;
        t += tau;
    }

    return x;
}


static double semi_euler(double t_ini, double x_ini, double T, double tau, RK4_Func f, void* context)
{
    double t = t_ini;
    double x = x_ini;

    for(; t <= T; )
    {
        double derx = f(t + tau, x, context);
        double x_next = x + tau*derx;

        x = x_next;
        t += tau;
    }

    return x;
}

static double rk4_func_x(double t, const double x, void* context)
{
    (void) context;
    (void) x;
    return t;
}

static void compare_rk4()
{
    double t_ini = 0;
    double x_ini = 0;
    double tau = 0.001f;

    for(double T = 0; T < 5; T += 0.5f)
    {
        double euler_r = euler(t_ini, x_ini, T, tau, rk4_func_x, NULL);
        double semi_euler_r = semi_euler(t_ini, x_ini, T, tau, rk4_func_x, NULL);
        double rk4_var_r = runge_kutta4_var(t_ini, x_ini, T, tau, 0.01f, rk4_func_x, NULL);
        double rk4_r = runge_kutta4(t_ini, x_ini, T, tau, rk4_func_x, NULL);
        double exact_r = T*T/2;

        LOG_INFO("RK4", "T = %f", T);
        LOG_INFO(">RK4", "exact:         %f", exact_r);
        LOG_INFO(">RK4", "euler:         %f", euler_r);
        LOG_INFO(">RK4", "semi_euler:    %f", semi_euler_r);
        LOG_INFO(">RK4", "rk4_var_r:     %f", rk4_var_r);
        LOG_INFO(">RK4", "rk4_r:         %f", rk4_r);
    }
}