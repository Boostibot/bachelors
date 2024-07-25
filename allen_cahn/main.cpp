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

typedef struct App_State {
    time_t init_time;
    bool is_in_step_mode;
    bool is_in_debug_mode;
    double remaining_steps;
    double step_by;
    double sim_time;
    double dt;
    size_t iter;
    int render_target; //0:phi 1:T rest: debug_option at index - 2; 

    int count_written_snapshots;
    int count_written_stats;
    int count_written_configs;
    Sim_Stats stats;
    Sim_Config config;

    Sim_Solver solver;
    Sim_State  states[SIM_HISTORY_MAX];
    int        used_states;
} App_State;

void sim_make_initial_conditions(Real* initial_phi_map, Real* initial_T_map, const Sim_Config& config)
{
    const Sim_Params& params = config.params;
    Exact_Params exact_params = get_static_exact_params(params);
    for(size_t y = 0; y < (size_t) params.ny; y++)
    {
        for(size_t x = 0; x < (size_t) params.nx; x++)
        {
            Vec2 pos = Vec2{((x+0.5)) / params.nx * params.L0, ((y+0.5)) / params.ny * params.L0}; 
            size_t i = x + y*(size_t) params.nx;
            
            if(config.params.do_exact)
            {
                Real r = hypot(pos.x - params.L0/2, pos.y - params.L0/2); 

                initial_phi_map[i] = exact_corresponing_phi_ini(r, exact_params);
                initial_T_map[i] = exact_u(0, r, exact_params);
            }
            else
            {
                double center_dist = hypot(config.init_circle_center.x - pos.x, config.init_circle_center.y - pos.y);

                bool is_within_cube = (config.init_square_from.x <= pos.x && pos.x < config.init_square_to.x) && 
                    (config.init_square_from.y <= pos.y && pos.y < config.init_square_to.y);

                double circle_normed_sdf = (config.init_circle_outer_radius - center_dist) / (config.init_circle_outer_radius - config.init_circle_inner_radius);
                if(circle_normed_sdf > 1)
                    circle_normed_sdf = 1;
                if(circle_normed_sdf < 0)
                    circle_normed_sdf = 0;

                double cube_sdf = is_within_cube ? 1 : 0;
                double factor = cube_sdf > circle_normed_sdf ? cube_sdf : circle_normed_sdf;

                initial_phi_map[i] = (Real) (factor*config.init_inside_phi + (1 - factor)*config.init_outside_phi);
                initial_T_map[i] = (Real) (factor*config.init_inside_T + (1 - factor)*config.init_outside_T);
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
    //@NOTE: I am failing to link to this TU from nvcc without using at least one cuda function
    // so here goes something harmless.
    cudaGetErrorString(cudaSuccess);
    int ny = config.params.ny;
    int nx = config.params.nx;

    size_t bytes_size = (size_t) (ny * nx) * sizeof(Real);
    Real* initial_F = (Real*) malloc(bytes_size);
    Real* initial_U = (Real*) malloc(bytes_size);

    app->used_states = solver_type_required_history(config.simul_solver);
    sim_solver_reinit(&app->solver, config.simul_solver, nx, ny);
    sim_states_reinit(app->states, app->used_states, config.simul_solver, nx, ny);
    sim_make_initial_conditions(initial_F, initial_U, config);
    
    Sim_Map maps[SIM_MAPS_MAX] = {0};
    sim_solver_get_maps(&app->solver, app->states, app->used_states, app->iter, maps, SIM_MAPS_MAX);
    int Phi_i = maps_find(maps, SIM_MAPS_MAX, "Phi");
    int T_i = maps_find(maps, SIM_MAPS_MAX, "T");

    if(Phi_i != -1)
        sim_modify(maps[Phi_i].data, initial_F, bytes_size, MODIFY_UPLOAD);

    if(T_i != -1)
        sim_modify(maps[T_i].data, initial_U, bytes_size, MODIFY_UPLOAD);

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
    SAVE_CSV = 2,
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
        save_state(app, SAVE_NETCDF | SAVE_BIN | SAVE_CONFIG | SAVE_STATS, 0);

    #define LOG_INFO_CONFIG_FLOAT(var) LOG_INFO("config", #var " = %.2lf", (double) config.params.var);
    #define LOG_INFO_CONFIG_INT(var) LOG_INFO("config", #var " = %i", (int) config.params.var);

    LOG_INFO("config", "solver = %s", solver_type_to_cstring(config.simul_solver));

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

	    while (!glfwWindowShouldClose(window))
        {
            double frame_start_time = clock_s();

            bool update_screen = frame_start_time - render_last_time > SCREEN_UPDATE_PERIOD;
            bool update_frame_time_display = frame_start_time - time_display_last_time > FPS_DISPLAY_PERIOD;
            bool poll_events = frame_start_time - poll_last_time > POLL_EVENTS_PERIOD;

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

            if(app->sim_time >= config.simul_stop_time && end_reached == false)
            {
                LOG_INFO("app", "reached stop time %lfs. Simulation paused.", config.simul_stop_time);
                app->is_in_step_mode = true;
                end_reached = true;
            }

            if(update_screen)
            {
                render_last_time = frame_start_time;
                int target = app->render_target;

                Sim_Map maps[SIM_MAPS_MAX] = {0};
                sim_solver_get_maps(&app->solver, app->states, app->used_states, app->iter, maps, SIM_MAPS_MAX);
                Sim_Map selected_map = {0};
                if(0 <= target && target < SIM_MAPS_MAX)
                    selected_map = maps[target];

                draw_sci_cuda_memory("main", selected_map.nx, selected_map.ny, (float) app->config.app_display_min, (float) app->config.app_display_max, config.app_linear_filtering, selected_map.data);
                glfwSwapBuffers(window);
            }

            if(update_frame_time_display)
            {
                glfwSetWindowTitle(window, format_string("%s step: %3.3lfms | real: %8.6lfms", solver_type_to_cstring(app->config.simul_solver), processing_time * 1000, app->sim_time*1000).c_str());
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

                app->config.params.do_debug = app->is_in_debug_mode;

                Sim_Step_Info step_info = {app->iter, app->sim_time};
                double solver_start_time = clock_s();
                app->sim_time += sim_solver_step(&app->solver, app->states, app->used_states, step_info, app->config.params, &app->stats);
                double solver_end_time = clock_s();

                processing_time = solver_end_time - solver_start_time;
                app->iter += 1;
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
        app->is_in_debug_mode = false;
        size_t iters = (size_t) ceil(config.simul_stop_time / config.params.dt);
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

            Sim_Step_Info step_info = {app->iter, app->sim_time};
            app->sim_time += sim_solver_step(&app->solver, app->states, app->used_states, step_info, app->config.params, &app->stats);
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
        if(key == GLFW_KEY_F9) new_render_target = 8;
        if(key == GLFW_KEY_F10) new_render_target = 9;
        
        if(new_render_target != -1)
        {
            Sim_Map maps[SIM_MAPS_MAX] = {0};
            sim_solver_get_maps(&app->solver, app->states, app->used_states, app->iter, maps, SIM_MAPS_MAX);

            if(0 <= new_render_target && new_render_target < SIM_MAPS_MAX)
            {
                if(maps[new_render_target].name == NULL)
                    maps[new_render_target].name = "<EMPTY>";

                LOG_INFO("APP", "redering %s", maps[new_render_target].name);
                app->render_target = new_render_target;
            }
            else
            {
                LOG_INFO("APP", "current render target %i outside of the allowed range [0, %i)", new_render_target, SIM_MAPS_MAX);
            }
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

bool save_bin_map_file(const char* filename, int nx, int ny, size_t iter, double t, const double* T, const double* phi);
bool save_csv_map_file(const char* filename, int nx, int ny, size_t iter, double t, const double* T, const double* phi);

bool load_bin_map_file(const char* filename, int* nx, int* ny, size_t* iter, double* t, double** T, double** phi);
bool load_csv_map_file(const char* filename, int* nx, int* ny, size_t* iter, double* t, double** T, double** phi);

bool save_csv_stat_file(const char* filename, size_t from, size_t to, Sim_Stats stats);
bool save_netcfd_file(const char* filename, App_State* app);

#include <cstdio>
#define BIN_FILE_MAGIC 0x11223344
bool save_bin_map_file(const char* filename, int nx, int ny, size_t iter, double t, const double* T, const double* phi)
{
    bool state = false;
    FILE* file = fopen(filename, "wb");
    if(file != NULL)
    {
        size_t N = (size_t) (nx*ny);
        int magic = BIN_FILE_MAGIC;
        fwrite(&magic, sizeof magic, 1, file);
        fwrite(&nx, sizeof nx, 1, file);
        fwrite(&ny, sizeof ny, 1, file);
        fwrite(&iter, sizeof iter, 1, file);
        fwrite(&t, sizeof t, 1, file);
        fwrite(T, sizeof(double), N, file);
        fwrite(phi, sizeof(double), N, file);

        state = ferror(file) == 0;
        fclose(file);
    }
    if(state == false)
        LOG_ERROR("APP", "Error saving bin file '%s'", filename);
    return state;
}

bool load_bin_map_file(const char* filename, int* nx, int* ny, size_t* iter, double* t, double** T, double** phi)
{
    bool state = false;
    FILE* file = fopen(filename, "rb");
    if(file != NULL)
    {
        int magic = 0;
        fread(&magic, sizeof magic, 1, file);
        state = magic == BIN_FILE_MAGIC;

        fread(nx, sizeof *nx, 1, file);
        fread(nx, sizeof *nx, 1, file);
        fread(ny, sizeof *ny, 1, file);
        fread(iter, sizeof *iter, 1, file);
        fread(t, sizeof *t, 1, file);

        *T = NULL;
        *phi = NULL;
        state = state && ferror(file) == 0;
        if(state && nx > 0 && ny > 0)
        {
            size_t map_size = (size_t) (*nx**ny)*sizeof(double);
            *T = (double*) malloc(map_size);
            *phi = (double*) malloc(map_size);
        
            fwrite(T, map_size, 1, file);
            fwrite(phi, map_size, 1, file);

            state = ferror(file) == 0;
        }
        fclose(file);
    }
    if(state == false)
        LOG_ERROR("APP", "Error loading bin file '%s'", filename);
    return state;
}
bool save_csv_map_file(const char* filename, int nx, int ny, size_t iter, double t, const double* T, const double* phi)
{
    bool state = false;
    FILE* file = fopen(filename, "wb");
    if(file != NULL)
    {
        fprintf(file, "%i,%i,%lli,%.8lf\n", nx, ny, (long long) iter, t);
        for(int y = 0; y < ny; y++)
        {
            if(nx > 0)
                fprintf(file, "%lf", T[0 + y*nx]);

            for(int x = 1; x < nx; x++)
                fprintf(file, ",%lf", T[x + y*nx]);

            fprintf(file, "\n");
        }

        for(int y = 0; y < ny; y++)
        {
            if(nx > 0)
                fprintf(file, "%lf", phi[0 + y*nx]);

            for(int x = 1; x < nx; x++)
                fprintf(file, ",%lf", phi[x + y*nx]);

            fprintf(file, "\n");
        }

        state = ferror(file) == 0;
        fclose(file);
    }
    if(state == false)
        LOG_ERROR("APP", "Error saving cvs file '%s'", filename);
    return state;
}

bool load_csv_map_file(const char* filename, int* nx, int* ny, size_t* iter, double* t, double** T, double** phi)
{
    bool state = false;
    FILE* file = fopen(filename, "rb");
    if(file != NULL)
    {
        long long iterll = 0;
        state = 4 == fscanf(file, "%i,%i,%lli,%lf\n", nx, ny, &iterll, t);
        *iter = (size_t) iterll;
        for(int y = 0; y < *ny && state; y++)
        {
            if(nx > 0)
                state = 1 == fscanf(file, "%lf", T[0 + y**nx]);

            for(int x = 1; x < *nx && state; x++)
                state = 1 == fscanf(file, ",%lf", T[x + y**nx]);

            fscanf(file, "\n");
        }

        for(int y = 0; y < *ny && state; y++)
        {
            if(nx > 0)
                state = 1 == fscanf(file, "%lf", phi[0 + y**nx]);

            for(int x = 1; x < *nx && state; x++)
                state = 1 == fscanf(file, ",%lf", phi[x + y**nx]);

            fscanf(file, "\n");
        }

        state = state && ferror(file) == 0;
        fclose(file);
    }
    if(state == false)
        LOG_ERROR("APP", "Error loading csv file '%s'", filename);
    return state;
}

struct Small_String {
    char data[16];
};

Small_String float_array_at_conv(Float_Array arr, size_t at)
{
    Small_String out = {0};
    if(at < arr.len)
        snprintf(out.data, sizeof out.data, "%lf", arr.data[at]);
    return out;
}

bool save_csv_stat_file(const char* filename, size_t from, size_t to, Sim_Stats stats, bool append)
{
    bool state = false;
    FILE* file = fopen(filename, append ? "ab" : "wb");
    if(file != NULL)
    {
        if(append == false)
        {
            fprintf(file, 
                "\"time\",\"iter\",\"T_delta_L1\",\"T_delta_L2\",\"T_delta_max\",\"T_delta_min\","
                "\"phi_delta_L1\",\"phi_delta_L2\",\"phi_delta_max\",\"phi_delta_min\",\"phi_iters\",\"T_iters\"");
                
            for(int s = 0; s < (int) stats.step_res_count; s++)
                fprintf(file, ",\"step_res_L1[%i]\",\"step_res_L1[%i]\",\"step_res_L1[%i]\",\"step_res_L1[%i]\"", s, s, s, s);

            fprintf(file, "\n");
        }

        for(size_t y = from; y < to; y++)
        {
            #define G(name) float_array_at_conv(stats.vectors.name, y).data
            fprintf(file, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", 
                G(time),G(iter),G(T_delta_L1),G(T_delta_L2),G(T_delta_max),G(T_delta_min),
                G(phi_delta_L1),G(phi_delta_L2),G(phi_delta_max),G(phi_delta_min),G(phi_iters),G(T_iters));
                
            for(int s = 0; s < (int) stats.step_res_count; s++)
                fprintf(file, ",%s,%s,%s,%s", G(step_res_L1[s]),G(step_res_L1[s]),G(step_res_L1[s]),G(step_res_L1[s]));

            fprintf(file, "\n");
            #undef G
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
        solver_type_to_cstring(config.simul_solver),
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
        std::string file = format_string("%s/%s_nc_%04i.nc", save_folder.data(), solver_type_to_cstring(app->config.simul_solver), snapshot_index);
        state = save_netcfd_file(file.data(), app) && state;
    }

    if(flags & SAVE_CSV)    
    {
        LOG_WARN("APP", "Save CSV should not be used!");
        // std::string file = format_string("%s_csv_%04i.csv", solver_type_to_cstring(app->config.simul_solver), snapshot_index);
        // state = save_netcfd_file(file.data(), app) && state;
    }

    if(flags & SAVE_BIN)    
    {
        Sim_Map maps[SIM_MAPS_MAX] = {0};
        sim_solver_get_maps(&app->solver, app->states, app->used_states, app->iter, maps, SIM_MAPS_MAX);
        int F_i = maps_find(maps, SIM_MAPS_MAX, "Phi");
        int U_i = maps_find(maps, SIM_MAPS_MAX, "T");

        if(F_i == -1 || U_i == -1)
            LOG_ERROR("APP", "Cannot find phi or T maps");
        else
        {
            int nx = maps[F_i].nx;
            int ny = maps[F_i].ny;
            size_t N = (size_t) (nx*ny);
            double* F = (double*) malloc(N*sizeof(double));
            double* U = (double*) malloc(N*sizeof(double));

            sim_modify_double(maps[F_i].data, F, N, MODIFY_DOWNLOAD);
            sim_modify_double(maps[U_i].data, U, N, MODIFY_DOWNLOAD);

            std::string file = format_string("%s/%s_bin_%04i.bin", save_folder.data(), solver_type_to_cstring(app->config.simul_solver), snapshot_index);
            state = save_bin_map_file(file.data(), nx, ny, app->iter, app->sim_time, U, F) && state;

            free(F);
            free(U);
        }
    }

    if((flags & SAVE_STATS))    
    {
        std::string file = format_string("%s/stats.csv", save_folder.data());
        state = save_csv_stat_file(file.data(), 0, app->stats.vectors.time.len, app->stats, app->count_written_stats != 0) && state;
        for(size_t i = 0; i < sizeof(app->stats.vectors)/sizeof(Float_Array); i++)
        {
            Float_Array* arr = ((Float_Array*) (void*) &app->stats.vectors) + i;
            float_array_clear(arr);
        }

        app->count_written_stats += 1;
    } 

    if((flags & SAVE_CONFIG) && app->count_written_configs == 0)    
    {
        std::string file = format_string("%s/config.ini", save_folder.data());
        state = save_netcfd_file(file.data(), app) && state;
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