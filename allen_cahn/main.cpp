#define JOT_ALL_IMPL
#include "config.h"
#include "integration_methods.h"
#include "simulation.h"
#include "log.h"
#include "assert.h"

#include "assert.h"
#include <cmath>
#include <stddef.h>
#include <cuda_runtime.h> 
#include <vector>
#include "exact.h"

#define WINDOW_TITLE                "sim"
#define DEF_WINDOW_WIDTH	        1024 
#define DEF_WINDOW_HEIGHT	        1024

#define FPS_DISPLAY_PERIOD          0.0
#define SCREEN_UPDATE_PERIOD        0.03
#define SCREEN_UPDATE_IDLE_PERIOD   0.05
#define POLL_EVENTS_PERIOD          0.03
#define FREE_RUN_PERIOD             0.001

static double clock_s();
static void wait_s(double seconds);

struct App_Stats {
    std::vector<double> time;
    std::vector<i64>   iter;
    
    std::vector<int>   Phi_iters;
    std::vector<int>   Phi_ellapsed_time; //TODO
    std::vector<float> T_iters;
    std::vector<float> T_ellapsed_time;

    std::vector<float> T_delta_L1;
    std::vector<float> T_delta_L2;
    std::vector<float> T_delta_max;
    std::vector<float> T_delta_min;

    std::vector<float> Phi_delta_L1;
    std::vector<float> Phi_delta_L2;
    std::vector<float> Phi_delta_max;
    std::vector<float> Phi_delta_min;

    std::vector<float> step_res_L1[MAX_STEP_RESIDUALS];
    std::vector<float> step_res_L2[MAX_STEP_RESIDUALS];
    std::vector<float> step_res_max[MAX_STEP_RESIDUALS];
    std::vector<float> step_res_min[MAX_STEP_RESIDUALS];
    int step_res_count;
};

enum {
    SIM_MAP_F = 0,
    SIM_MAP_U = 1,
    SIM_MAP_NEXT_F = 2,
    SIM_MAP_NEXT_U = 3,
    SIM_MAP_TEMP = 4,
    SIM_MAP_COUNT = 32,
};
struct App_State {
    time_t init_time;
    bool is_in_step_mode;
    bool is_in_debug_mode;
    double remaining_steps;
    double step_by;
    double sim_time;
    double dt;
    double last_stats_save;
    i64 iter;
    int render_target;

    int count_written_snapshots;
    int count_written_stats;
    int count_written_configs;
    Sim_Stats stats;
    Sim_Config config;

    union {
        struct {
            Sim_Map F;
            Sim_Map U;
            Sim_Map next_F;
            Sim_Map next_U;
            Sim_Map temp[28];
        };
        Sim_Map all[32];
    } maps;
    App_Stats stat_vectors;
};

void sim_make_initial_conditions(Real* F, Real* U, const Sim_Config& config)
{
    const Sim_Params& params = config.params;
    Exact_Params exact_params = get_static_exact_params();
    for(size_t y = 0; y < (size_t) params.ny; y++)
    {
        for(size_t x = 0; x < (size_t) params.nx; x++)
        {
            Vec2 pos = Vec2{((x+0.5)) / params.nx * params.L0, ((y+0.5)) / params.ny * params.L0}; 
            size_t i = x + y*(size_t) params.nx;
            
            if(config.params.do_exact)
            {
                Real r = hypot(pos.x - params.L0/2, pos.y - params.L0/2); 

                F[i] = exact_corresponing_phi_ini(r, exact_params, params.xi);
                U[i] = exact_u(0, r, exact_params);
            }
            else
            {
                Real lo = config.init_circle_radius - params.xi*config.init_circle_fade/2; 
                Real hi = config.init_circle_radius + params.xi*config.init_circle_fade/2; 

                double r = hypot(config.init_circle_center.x - pos.x, config.init_circle_center.y - pos.y);

                bool is_within_cube = (config.init_square_from.x <= pos.x && pos.x < config.init_square_to.x) && 
                    (config.init_square_from.y <= pos.y && pos.y < config.init_square_to.y);

                double circle_sdf = 1 - (r - lo)/(hi - lo);
                if(circle_sdf > 1)
                    circle_sdf = 1;
                if(circle_sdf < 0)
                    circle_sdf = 0;

                double cube_sdf = is_within_cube ? 1 : 0;
                double factor = cube_sdf > circle_sdf ? cube_sdf : circle_sdf;

                F[i] = (Real) (factor*config.init_inside_phi + (1 - factor)*config.init_outside_phi);
                U[i] = (Real) (factor*config.init_inside_T + (1 - factor)*config.init_outside_T);
            }

        }
    }
}

int maps_find(const Sim_Map* maps, int map_count, const char* name)
{
    for(int i = 0; i < map_count; i++)
    {
        if(maps[i].name && name)
        {
            if(strcmp(maps[i].name, name) == 0)
                return i;
        }
    }
    return -1;
}

void simulation_state_reload(App_State* app, Sim_Config config)
{
    cudaGetErrorString(cudaSuccess);
    int ny = config.params.ny;
    int nx = config.params.nx;

    size_t bytes_size = (size_t) (ny * nx) * sizeof(Real);
    Real* initial_F = (Real*) malloc(bytes_size);
    Real* initial_U = (Real*) malloc(bytes_size);
    sim_make_initial_conditions(initial_F, initial_U, config);
    
    double t = 0;
    i64 iter = 0;
    sim_realloc(&app->maps.F, "F", nx, ny, t, iter);
    sim_realloc(&app->maps.U, "U", nx, ny, t, iter);

    sim_modify(app->maps.F.data, initial_F, bytes_size, MODIFY_UPLOAD);
    sim_modify(app->maps.U.data, initial_U, bytes_size, MODIFY_UPLOAD);
    app->config = config;
    app->init_time = time(NULL); 
    free(initial_F);
    free(initial_U);
}

#ifdef COMPILE_GRAPHICS
#include "gl.h"
#include <GLFW/glfw3.h>

void glfw_resize_func(GLFWwindow* window, int width, int heigth);
void glfw_key_func(GLFWwindow* window, int key, int scancode, int action, int mods);
#endif

enum {
    SAVE_NETCDF = 1,
    SAVE_BIN = 4,
    SAVE_STATS = 8,
    SAVE_CONFIG = 16,
    SAVE_ALL = 31
};
bool save_state(App_State* app, int flags, int snapshot_index);

int main()
{
    static App_State app_data = {};
    static Sim_Config config = {};
    if(allen_cahn_read_config("config.ini", &config) == false)
        return 1;

    if(config.app_run_tests)
        run_tests();
    if(config.app_run_benchmarks)
        run_benchmarks(config.params.nx * config.params.ny);
    if(config.app_run_simulation == false)
        return 0;

    App_State* app = (App_State*) &app_data;
    app->is_in_step_mode = true;
    app->remaining_steps = 0;
    app->step_by = 1;
    
    simulation_state_reload(app, config);
    
    if(config.snapshot_initial_conditions)
        save_state(app, SAVE_NETCDF | SAVE_BIN | SAVE_CONFIG, 0);

    #define LOG_INFO_CONFIG_FLOAT(var) LOG_INFO("config", #var " = %.2lf", (double) config.params.var);
    #define LOG_INFO_CONFIG_INT(var) LOG_INFO("config", #var " = %i", (int) config.params.var);

    LOG_INFO("config", "solver = %s", solver_type_to_cstring(config.params.solver));

    LOG_INFO_CONFIG_INT(nx);
    LOG_INFO_CONFIG_INT(ny);
    LOG_INFO_CONFIG_FLOAT(L0);

    LOG_INFO_CONFIG_FLOAT(dt); 
    LOG_INFO_CONFIG_FLOAT(L);  
    LOG_INFO_CONFIG_FLOAT(xi); 
    LOG_INFO_CONFIG_FLOAT(a);  
    LOG_INFO_CONFIG_FLOAT(b);
    LOG_INFO_CONFIG_FLOAT(alpha);
    LOG_INFO_CONFIG_FLOAT(beta);
    LOG_INFO_CONFIG_FLOAT(Tm);

    LOG_INFO_CONFIG_FLOAT(S);
    LOG_INFO_CONFIG_FLOAT(theta0);

    #undef LOG_INFO_CONFIG_FLOAT
    #undef LOG_INFO_CONFIG_INT

    #ifndef COMPILE_GRAPHICS
    config.interactive_mode = false;
    #endif // COMPILE_GRAPHICS

    //OPENGL setup
    if(config.app_interactive_mode)
    {   
        #ifdef COMPILE_GRAPHICS
        TEST(glfwInit(), "Failed to init glfw");

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);  
 
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        ASSERT(monitor && mode);
        if(monitor != NULL && mode != NULL)
        {
            glfwWindowHint(GLFW_RED_BITS, mode->redBits);
            glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
            glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
            glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
        }
 
        GLFWwindow* window = glfwCreateWindow(DEF_WINDOW_WIDTH, DEF_WINDOW_HEIGHT, WINDOW_TITLE, NULL, NULL);
        TEST(window != NULL, "Failed to make glfw window");

        //glfwSetWindowUserPointer(window, &app);
        glfwMakeContextCurrent(window);
        glfwSetWindowUserPointer(window, app);
        glfwSetFramebufferSizeCallback(window, glfw_resize_func);
        glfwSetKeyCallback(window, glfw_key_func);
        glfwSwapInterval(0);
        gl_init((void*) glfwGetProcAddress);
    
        double time_display_last_time = 0;
        double render_last_time = 0;
        double simulated_last_time = 0;
        double poll_last_time = 0;

        double processing_time = 0;

        int snapshot_every_i = 0;
        int snapshot_times_i = 0;
        bool end_reached = false;

        double start_time = clock_s();
	    while (!glfwWindowShouldClose(window))
        {
            bool save_this_iter = false;
            double frame_start_time = clock_s();

            bool update_screen = frame_start_time - render_last_time > SCREEN_UPDATE_PERIOD;
            bool update_frame_time_display = frame_start_time - time_display_last_time > FPS_DISPLAY_PERIOD;
            bool poll_events = frame_start_time - poll_last_time > POLL_EVENTS_PERIOD;

            double next_snapshot_every = (double) (snapshot_every_i + 1) * config.snapshot_every;
            double next_snapshot_times = (double) (snapshot_times_i + 1) * config.simul_stop_time / config.snapshot_times;

            if(app->sim_time >= next_snapshot_every)
            {
                snapshot_every_i += 1;
                save_this_iter = true;
            }

            if(app->sim_time >= next_snapshot_times && end_reached == false)
            {
                snapshot_times_i += 1;
                save_this_iter = true;
            }

            if(config.simul_stop_time - app->sim_time < 1e-16 && end_reached == false)
            {
                LOG_INFO("app", "reached stop time %lfs. Took %lf seconds. Simulation paused.", config.simul_stop_time, clock_s() - start_time);
                app->is_in_step_mode = true;
                end_reached = true;
                save_this_iter = true;
            }

            if(save_this_iter)
                save_state(app, SAVE_NETCDF | SAVE_BIN | SAVE_CONFIG | SAVE_STATS, ++app->count_written_snapshots);

            if(update_screen)
            {
                render_last_time = frame_start_time;
                Sim_Map render_target = app->maps.all[app->render_target];
                draw_sci_cuda_memory("main", render_target.nx, render_target.ny, (float) app->config.app_display_min, (float) app->config.app_display_max, config.app_linear_filtering, render_target.data);
                glfwSwapBuffers(window);
            }

            if(update_frame_time_display)
            {
                glfwSetWindowTitle(window, format_string("%s step: %3.3lfms | real: %8.6lfms", solver_type_to_cstring(app->config.params.solver), processing_time * 1000, app->sim_time*1000).c_str());
                time_display_last_time = frame_start_time;
            }

            bool step_sym = false;
            if(app->is_in_step_mode)
                step_sym = app->remaining_steps > 0.5;
            else
                step_sym = frame_start_time - simulated_last_time > FREE_RUN_PERIOD/app->step_by;

            if(step_sym)
            {
                simulated_last_time = frame_start_time;
                app->remaining_steps -= 1;

                Sim_Params params = app->config.params;
                params.iter = app->iter;
                params.time = app->sim_time;
                params.do_debug = app->is_in_debug_mode;
                params.do_stats = app->config.app_collect_stats;
                params.do_stats_step_residual = app->config.app_collect_step_residuals;
                params.stats = &app->stats;
                params.temp_maps = app->maps.temp;
                params.temp_map_count = sizeof app->maps.temp / sizeof *app->maps.temp;

                double solver_start_time = clock_s();
                app->sim_time += sim_step(app->maps.F, app->maps.U, &app->maps.next_F, &app->maps.next_U, params);
                double solver_end_time = clock_s();

                if(params.do_stats && app->sim_time >= app->last_stats_save + app->config.app_collect_stats_every)
                {
                    if(app->stat_vectors.step_res_count < app->stats.step_res_count)
                    {
                        size_t first_size = app->stat_vectors.step_res_L1[0].size();
                        for(int i = app->stat_vectors.step_res_count; i < app->stats.step_res_count; i++)
                        {
                            app->stat_vectors.step_res_L1[i].resize(first_size);
                            app->stat_vectors.step_res_L2[i].resize(first_size);
                            app->stat_vectors.step_res_max[i].resize(first_size);
                            app->stat_vectors.step_res_min[i].resize(first_size);
                        }
                        app->stat_vectors.step_res_count = app->stats.step_res_count;
                    }

                    app->stat_vectors.step_res_count = std::max(app->stat_vectors.step_res_count, app->stats.step_res_count);
                    app->stat_vectors.time.push_back(app->stats.time);
                    app->stat_vectors.iter.push_back(app->stats.iter);
                    
                    app->stat_vectors.Phi_iters.push_back(app->stats.Phi_iters);
                    app->stat_vectors.Phi_ellapsed_time.push_back(app->stats.Phi_ellapsed_time); //TODO
                    app->stat_vectors.T_iters.push_back(app->stats.T_iters);
                    app->stat_vectors.T_ellapsed_time.push_back(app->stats.T_ellapsed_time);

                    app->stat_vectors.T_delta_L1.push_back(app->stats.T_delta_L1);
                    app->stat_vectors.T_delta_L2.push_back(app->stats.T_delta_L2);
                    app->stat_vectors.T_delta_max.push_back(app->stats.T_delta_max);
                    app->stat_vectors.T_delta_min.push_back(app->stats.T_delta_min);

                    app->stat_vectors.Phi_delta_L1.push_back(app->stats.Phi_delta_L1);
                    app->stat_vectors.Phi_delta_L2.push_back(app->stats.Phi_delta_L2);
                    app->stat_vectors.Phi_delta_max.push_back(app->stats.Phi_delta_max);
                    app->stat_vectors.Phi_delta_min.push_back(app->stats.Phi_delta_min);

                    for(int i = 0; i < app->stat_vectors.step_res_count; i++)
                    {
                        app->stat_vectors.step_res_L1[i].push_back(app->stats.step_res_L1[i]);
                        app->stat_vectors.step_res_L2[i].push_back(app->stats.step_res_L2[i]);
                        app->stat_vectors.step_res_max[i].push_back(app->stats.step_res_max[i]);
                        app->stat_vectors.step_res_min[i].push_back(app->stats.step_res_min[i]);
                    }
                    app->last_stats_save = app->sim_time;
                }

                processing_time = solver_end_time - solver_start_time;
                app->iter += 1;

                std::swap(app->maps.F, app->maps.next_F);
                std::swap(app->maps.U, app->maps.next_U);
            }

            if(poll_events)
            {
                poll_last_time = frame_start_time;
                glfwPollEvents();
            }

            double end_frame_time = clock_s();
            app->dt = end_frame_time - frame_start_time;

            //if is idle for the last 0.5 seconds limit the framerate to IDLE_RENDER_FREQ
            bool do_frame_limiting = simulated_last_time + 0.5 < frame_start_time;
            do_frame_limiting = true;
            if(do_frame_limiting && app->dt < SCREEN_UPDATE_IDLE_PERIOD)
                wait_s(SCREEN_UPDATE_IDLE_PERIOD - app->dt);
        }
    
        glfwDestroyWindow(window);
        glfwTerminate();
        #endif
    }
    else
    {
        LOG_ERROR("APP", "Nom interactive mode is right now not maintained");
        app->is_in_debug_mode = false;
        i64 iters = (i64) ceil(config.simul_stop_time / config.params.dt);
        size_t snapshot_every_i = 0;
        size_t snapshot_times_i = 0;
        bool end_reached = false;
        double start_time = clock_s();
        double last_notif_time = 0;

        for(; app->iter <= iters; app->iter++)
        {
            double now = clock_s();
            
            double next_snapshot_every = (double) (snapshot_every_i + 1) * config.snapshot_every;
            double next_snapshot_times = (double) (snapshot_times_i + 1) * config.simul_stop_time / config.snapshot_times;

            if(app->sim_time >= next_snapshot_every)
            {
                snapshot_every_i += 1;
                save_state(app, SAVE_NETCDF | SAVE_BIN | SAVE_CONFIG | SAVE_STATS, ++app->count_written_snapshots);
            }

            if(app->sim_time >= next_snapshot_times && end_reached == false)
            {
                snapshot_times_i += 1;
                save_state(app, SAVE_NETCDF | SAVE_BIN | SAVE_CONFIG | SAVE_STATS, ++app->count_written_snapshots);
            }

            if(now - last_notif_time > 1 || app->iter == iters || app->iter == 0)
            {
                last_notif_time = now;
                LOG_INFO("app", "... completed %2lf%%", (double) app->iter * 100 / iters);
                if(app->iter == iters)
                    break;
            }

            // Sim_Step_Info step_info = {app->iter, app->sim_time};
            // app->sim_time += sim_solver_step(&app->solver, app->states, app->used_states, step_info, app->config.params, &app->stats);
        }
        double end_time = clock_s();
        double runtime = end_time - start_time;

        LOG_INFO("app", "Finished!");
        LOG_INFO("app", "runtime: %.2lfs | iters: %lli | average step time: %.2lf ms", runtime, (long long) iters, runtime / (double) iters * 1000);
    }

    return 0;    
}

#ifdef COMPILE_GRAPHICS

void glfw_resize_func(GLFWwindow* window, int width, int heigth)
{
    (void) window;
	glViewport(0, 0, width, heigth);
}

void glfw_key_func(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void) mods;
    (void) scancode;
    (void) window;
    App_State* app = (App_State*) glfwGetWindowUserPointer(window); (void) app;

    if(action == GLFW_RELEASE)
    {
        if(key == GLFW_KEY_ENTER)
            app->remaining_steps = app->step_by;
        if(key == GLFW_KEY_SPACE)
        {
            app->is_in_step_mode = !app->is_in_step_mode;
            LOG_INFO("APP", "Simulation %s", app->is_in_step_mode ? "paused" : "running");
        }
        if(key == GLFW_KEY_D)
        {
            app->is_in_debug_mode = !app->is_in_debug_mode;
            LOG_INFO("APP", "Debug %s", app->is_in_debug_mode ? "true" : "false");
        }
        if(key == GLFW_KEY_L)
        {
            app->config.app_linear_filtering = !app->config.app_linear_filtering;
            LOG_INFO("APP", "Linear FIltering %s", app->config.app_linear_filtering ? "true" : "false");
        }
        if(key == GLFW_KEY_C)
        {
            app->config.params.do_corrector_loop = !app->config.params.do_corrector_loop;
            LOG_INFO("APP", "Corrector loop %s", app->config.params.do_corrector_loop ? "true" : "false");
        }
        if(key == GLFW_KEY_S)
        {
            LOG_INFO("APP", "On demand snapshot triggered");
            save_state(app, SAVE_NETCDF | SAVE_BIN | SAVE_CONFIG | SAVE_STATS, ++app->count_written_snapshots);
        }
        if(key == GLFW_KEY_R)
        {
            LOG_INFO("APP", "Input range to display in form 'MIN space MAX'");

            double new_display_max = app->config.app_display_max;
            double new_display_min = app->config.app_display_min;
            if(scanf("%lf %lf", &new_display_min, &new_display_max) != 2)
            {
                LOG_INFO("APP", "Bad range syntax!");
            }
            else
            {
                LOG_INFO("APP", "displaying range [%.2lf, %.2lf]", new_display_min, new_display_max);
                app->config.app_display_max = (Real) new_display_max;
                app->config.app_display_min = (Real) new_display_min;
            }
        }

        if(key == GLFW_KEY_P)
        {
            LOG_INFO("APP", "Input simulation speed modifier in form 'NUM'");
            double new_step_by = app->step_by;
            if(scanf("%lf", &new_step_by) != 1)
            {
                LOG_INFO("APP", "Bad speed syntax!");
            }
            else
            {
                LOG_INFO("APP", "using simulation speed %.2lf", new_step_by);
                app->step_by = new_step_by;
            }
        }

        int new_render_target = -1;
        if(key == GLFW_KEY_F1) new_render_target = 0;
        if(key == GLFW_KEY_F2) new_render_target = 1;
        if(key == GLFW_KEY_F3) new_render_target = 2;
        if(key == GLFW_KEY_F4) new_render_target = 3;
        if(key == GLFW_KEY_F5) new_render_target = 4;
        if(key == GLFW_KEY_F6) new_render_target = 5;
        if(key == GLFW_KEY_F7) new_render_target = 6;
        if(key == GLFW_KEY_F8) new_render_target = 7;

        if(key == GLFW_KEY_F9) new_render_target = MOD(app->render_target-1, SIM_MAP_COUNT);
        if(key == GLFW_KEY_F10) new_render_target = MOD(app->render_target+1, SIM_MAP_COUNT);
        
        if(new_render_target != -1)
        {
            const char* map_name = app->maps.all[new_render_target].name;
            if(strlen(map_name) == 0)
                map_name = "<EMPTY>";

            LOG_INFO("APP", "redering %s", map_name);
            app->render_target = new_render_target;
        }
    }
}
#endif

#include <chrono>
static double clock_s()
{
    static int64_t init_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int64_t now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    double unit = (double) std::chrono::high_resolution_clock::period::den;
    double clock = (double) (now - init_time) / unit;
    return clock;
}

#include <thread>
static void wait_s(double seconds)
{
    auto now = std::chrono::high_resolution_clock::now();
    auto sleep_til = now + std::chrono::microseconds((int64_t) (seconds * 1000*1000));
    std::this_thread::sleep_until(sleep_til);
}

bool save_bin_map_file(const char* filename, int nx, int ny, double dx, double dy, i64 iter, double dt, const Sim_Map* maps, int map_count);
bool save_csv_stat_file(const char* filename, int nx, int ny, double dt, size_t from, size_t to, const App_Stats& stats, bool append);
bool save_netcfd_file(const char* filename, App_State* app);

#include <cstdio>
#define BIN_FILE_MAGIC 0x11223344

bool save_bin_map_file(const char* filename, int nx, int ny, double dx, double dy, i64 iter, double time, const Sim_Map* maps, int map_count)
{
    bool state = false;
    FILE* file = fopen(filename, "wb");
    if(file != NULL)
    {
        size_t N = (size_t) (nx*ny);
        int magic = BIN_FILE_MAGIC;
        fwrite(&magic, sizeof magic, 1, file);
        fwrite(&map_count, sizeof map_count, 1, file);
        fwrite(&nx, sizeof nx, 1, file);
        fwrite(&ny, sizeof ny, 1, file);
        fwrite(&dx, sizeof dx, 1, file);
        fwrite(&dy, sizeof dy, 1, file);
        fwrite(&time, sizeof time, 1, file);
        fwrite(&iter, sizeof iter, 1, file);
        for(int i = 0; i < map_count; i++)
            if(maps[i].iter == iter && maps[i].data)
                fwrite(maps[i].name, sizeof maps[i].name, 1, file);

        for(int i = 0; i < map_count; i++)
            if(maps[i].iter == iter && maps[i].data)
                fwrite(maps[i].data, sizeof(double), N, file);

        state = ferror(file) == 0;
        fclose(file);
    }
    if(state == false)
        LOG_ERROR("APP", "Error saving bin file '%s'", filename);
    return state;
}

struct Small_String {
    char data[16];
};

template<typename T>
Small_String float_vec_string_at(const std::vector<T>& arr, size_t at)
{
    Small_String out = {0};
    if(at < arr.size())
        snprintf(out.data, sizeof out.data, "%f", (float) arr[at]);
    return out;
}

template<typename T>
Small_String int_vec_string_at(const std::vector<T>& arr, size_t at)
{
    Small_String out = {0};
    if(at < arr.size())
        snprintf(out.data, sizeof out.data, "%lli", (long long) arr[at]);
    return out;
}

bool save_csv_stat_file(const char* filename, int nx, int ny, double dt, size_t from, size_t to, const App_Stats& stats, bool append)
{
    bool state = false;
    FILE* file = fopen(filename, append ? "ab" : "wb");
    if(file != NULL)
    {
        if(append == false)
        {
            fprintf(file, "%i,%i,%lf\n", nx, ny, dt);
            fprintf(file, 
                "\"time\",\"iter\",\"Phi_iters\",\"T_iters\",\"T_delta_L1\",\"T_delta_L2\",\"T_delta_max\",\"T_delta_min\","
                "\"Phi_delta_L1\",\"Phi_delta_L2\",\"Phi_delta_max\",\"Phi_delta_min\"");
                
            for(int s = 0; s < (int) stats.step_res_count; s++)
                fprintf(file, ",\"step_res_L1[%i]\",\"step_res_L2[%i]\",\"step_res_max[%i]\",\"step_res_min[%i]\"", s, s, s, s);

            fprintf(file, "\n");
        }

        for(size_t y = from; y < to; y++)
        {
            #define F(name) float_vec_string_at(stats.name, y).data
            #define I(name) int_vec_string_at(stats.name, y).data
            fprintf(file, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", 
                F(time),I(iter),I(Phi_iters),I(T_iters),
                F(T_delta_L1),F(T_delta_L2),F(T_delta_max),F(T_delta_min),
                F(Phi_delta_L1),F(Phi_delta_L2),F(Phi_delta_max),F(Phi_delta_min)
            );
                
            for(int s = 0; s < (int) stats.step_res_count; s++)
                fprintf(file, ",%s,%s,%s,%s", F(step_res_L1[s]),F(step_res_L2[s]),F(step_res_max[s]),F(step_res_min[s]));

            fprintf(file, "\n");
            #undef F
            #undef I
        }

        state = ferror(file) == 0;
        fclose(file);
    }
    return state;
}

#include <filesystem>
bool make_save_folder(const Sim_Config& config, time_t init_time, std::string* out, bool create_in_filesystem)
{
    tm* t = localtime(&init_time);
    std::string folder = format_string("%s%s%s%04i-%02i-%02i__%02i-%02i-%02i__%s%s", 
        config.snapshot_folder.data(),
        config.snapshot_folder.size() > 0 ? "/" : "",
        config.snapshot_prefix.data(), 
        t->tm_year + 1900, t->tm_mon, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec,
        solver_type_to_cstring(config.params.solver),
        config.snapshot_postfix.data()
    );
    std::error_code error = {};
    if(folder.empty() == false && create_in_filesystem)
        std::filesystem::create_directory(folder, error);

    if(error)
        LOG_ERROR("APP", "Error creating save folder '%s': %s", folder.data(), error.message().data());

    *out = folder;
    return !error;
}

bool save_state(App_State* app, int flags, int snapshot_index)
{
    std::string save_folder;
    bool state = make_save_folder(app->config, app->init_time, &save_folder, true);

    if(flags & SAVE_NETCDF)    
    {
        std::string file = format_string("%s/%s_nc_%04i.nc", save_folder.data(), solver_type_to_cstring(app->config.params.solver), snapshot_index);
        state = save_netcfd_file(file.data(), app) && state;
    }

    if(flags & SAVE_BIN)    
    {
        Sim_Map maps[SIM_MAP_COUNT] = {0};

        int nx = app->config.params.nx;
        int ny = app->config.params.ny;
        double dx = app->config.params.L0 / nx;
        double dy = app->config.params.L0 / ny;
        double time = app->sim_time;
        size_t N = (size_t) (nx*ny);

        int maps_count = 0;
        for(int i = 0; i < SIM_MAP_COUNT; i++)
        {
            Sim_Map* from = &app->maps.all[i];

            if(from->iter == app->iter && from->nx > 0 && from->ny > 0)
            {
                Sim_Map* into = &maps[maps_count++];
                *into = *from;
                into->data = (double*) malloc(N*sizeof(double));
                sim_modify_double(from->data, into->data, N, MODIFY_DOWNLOAD);
            }
        }

        std::string file = format_string("%s/maps_%04i.bin", save_folder.data(), snapshot_index);
        state = save_bin_map_file(file.data(), nx, ny, dx, dy, app->iter, time, maps, maps_count) && state;
        for(int i = 0; i < SIM_MAP_COUNT; i++)
            free((double*) maps[i].data);
    }

    if((flags & SAVE_STATS))    
    {
        std::string file = format_string("%s/stats.csv", save_folder.data());
        state = save_csv_stat_file(file.data(), app->config.params.nx, app->config.params.ny, app->config.params.dt, 0, app->stat_vectors.time.size(), app->stat_vectors, app->count_written_stats != 0) && state;
        app->stat_vectors.time.clear();
        app->stat_vectors.iter.clear();
        app->stat_vectors.Phi_iters.clear();
        app->stat_vectors.Phi_ellapsed_time.clear();
        app->stat_vectors.T_iters.clear();
        app->stat_vectors.T_ellapsed_time.clear();
        app->stat_vectors.T_delta_L1.clear();
        app->stat_vectors.T_delta_L2.clear();
        app->stat_vectors.T_delta_max.clear();
        app->stat_vectors.T_delta_min.clear();
        app->stat_vectors.Phi_delta_L1.clear();
        app->stat_vectors.Phi_delta_L2.clear();
        app->stat_vectors.Phi_delta_max.clear();
        app->stat_vectors.Phi_delta_min.clear();
        for(size_t i = 0; i < MAX_STEP_RESIDUALS; i++)
        {
            app->stat_vectors.step_res_L1[i].clear();
            app->stat_vectors.step_res_L2[i].clear();
            app->stat_vectors.step_res_max[i].clear();
            app->stat_vectors.step_res_min[i].clear();
        }
        app->count_written_stats += 1;
    } 

    if((flags & SAVE_CONFIG) && app->count_written_configs == 0)    
    {
        std::string file = format_string("%s/config.ini", save_folder.data());
        file_write_entire(file.data(), app->config.entire_config_file);
        app->count_written_configs += 1;
    } 

    return state;
}


#ifdef COMPILE_NETCDF
#include <netcdf.h>
bool save_netcfd_file(const char* filename, App_State* app)
{
    enum {NDIMS = 2};
    int dataset_ID = 0;
    int dim_ids[NDIMS] = {0};
    const Sim_Params& params = app->config.params;

    int iter = (int) app->iter;
    double real_time = (Real) app->iter * app->config.params.dt;

    #define NC_ERROR_AND(code) (code) != NC_NOERR ? (code) :

    int nc_error = nc_create(filename, NC_NETCDF4, &dataset_ID);
    LOG_INFO("APP", "saving NetCDF file '%s'", filename);
    if(nc_error != NC_NOERR) {
        LOG_INFO("APP", "NetCDF create error on file '%s': %s.", filename, nc_strerror(nc_error));
        return false;
    }

    nc_error = NC_ERROR_AND(nc_error) nc_def_dim (dataset_ID, "x", (size_t) params.nx, dim_ids + 0);
    nc_error = NC_ERROR_AND(nc_error) nc_def_dim (dataset_ID, "y", (size_t) params.ny, dim_ids + 1);
    if(nc_error != NC_NOERR) {
        LOG_INFO("APP", "NetCDF define dim error: %s.", nc_strerror(nc_error));
        return false;
    }

    int real_type = sizeof(Real) == sizeof(double) ? NC_DOUBLE : NC_FLOAT;
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "mesh_size_x", NC_INT, 1, &params.nx);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "mesh_size_y", NC_INT, 1, &params.ny);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "L0", NC_INT, 1, &params.L0);

    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "iter", NC_INT, 1, &iter);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "time", NC_INT, 1, &real_time);

    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "dt", real_type, 1, &params.dt);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "L", real_type, 1, &params.L);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "xi", real_type, 1, &params.xi);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "a", real_type, 1, &params.a);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "b", real_type, 1, &params.b);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "alpha", real_type, 1, &params.alpha);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "beta", real_type, 1, &params.beta);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "Tm", real_type, 1, &params.Tm);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "S", real_type, 1, &params.S);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "nx", real_type, 1, &params.nx);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "theta0", real_type, 1, &params.theta0);

    if(nc_error != NC_NOERR) {
        LOG_INFO("APP", "NetCDF error while outputing params: %s.", nc_strerror(nc_error));
        return false;
    }

    // nc_enddef(dataset_ID);
    int Phi_ID = 0;
    int T_ID = 0;

    nc_error = NC_ERROR_AND(nc_error) nc_def_var(dataset_ID, "Phi", real_type, NDIMS, dim_ids, &Phi_ID);
    nc_error = NC_ERROR_AND(nc_error) nc_def_var(dataset_ID, "T", real_type, NDIMS, dim_ids, &T_ID);

    size_t copy_size = (size_t) params.nx * (size_t) params.ny*sizeof(Real);
    Real* F = (Real*) malloc(copy_size);
    Real* U = (Real*) malloc(copy_size);

    Sim_Map maps[SIM_MAPS_MAX] = {0};
    sim_solver_get_maps(&app->solver, app->states, app->used_states, app->iter, maps, SIM_MAPS_MAX);
    int Phi_i = maps_find(maps, SIM_MAPS_MAX, "Phi");
    int T_i = maps_find(maps, SIM_MAPS_MAX, "T");

    if(Phi_i != -1)
        sim_modify(maps[Phi_i].data, F, copy_size, MODIFY_DOWNLOAD);

    if(T_i != -1)
        sim_modify(maps[T_i].data, U, copy_size, MODIFY_DOWNLOAD);

    nc_error = NC_ERROR_AND(nc_error) nc_put_var(dataset_ID, Phi_ID, F);
    nc_error = NC_ERROR_AND(nc_error) nc_put_var(dataset_ID, T_ID, U);

    nc_error = NC_ERROR_AND(nc_error) nc_close(dataset_ID);

    free(F);
    free(U);
    if(nc_error != NC_NOERR) {
        LOG_INFO("APP", "NetCDF error while outputing data: %s.", nc_strerror(nc_error));
        return false;
    }

    return true;
}
#else
bool save_netcfd_file(const char* filename, App_State* app)
{
    (void) filename;
    (void) app;
    return true;
}
#endif