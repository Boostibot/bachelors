#define JOT_ALL_IMPL
#include "config.h"
#include "integration_methods.h"
#include "simulation.h"
#include "log.h"
#include "assert.h"

#include <cmath>
#include <stddef.h>

#include "assert.h"
#include <cuda_runtime.h> 
#include <netcdf.h>

#define WINDOW_TITLE        "sim"
#define DEF_WINDOW_WIDTH	800 
#define DEF_WINDOW_HEIGHT	800

#define FPS_DISPLAY_FREQ    (double) 50000
#define RENDER_FREQ         (double) 30
#define IDLE_RENDER_FREQ    (double) 20
#define POLL_FREQ           (double) 30
#define FREE_RUN_SYM_FPS    (double) 200000000
#define COMPILE_GRAPHICS  

static double clock_s();
static void wait_s(double seconds);

typedef struct App_State {
    bool is_in_step_mode;
    bool is_in_debug_mode;
    uint8_t render_target; //0:phi 1:T rest: debug_option at index - 2; 
    double remaining_steps;
    double step_by;
    double dt;
    int iter;
    int snapshot_index;
    time_t init_time;

    Allen_Cahn_Stats stats_last;
    Allen_Cahn_Stats stats_summed;
    int stats_summed_over_iters;

    Allen_Cahn_Config   config;

    Sim_Solver solver;
    Sim_State  states[SIM_HISTORY_MAX];
    int used_states;
} App_State;

void sim_make_initial_conditions(Real* initial_phi_map, Real* initial_T_map, Allen_Cahn_Config config)
{
    Allen_Cahn_Params params = config.params;
    Allen_Cahn_Initial_Conditions initial = config.initial_conditions;
    for(size_t y = 0; y < (size_t) params.ny; y++)
    {
        for(size_t x = 0; x < (size_t) params.nx; x++)
        {
            Vec2 pos = Vec2{((x+0.5)) / params.nx * params.L0, ((y+0.5)) / params.ny * params.L0}; 
            size_t i = x + y*(size_t) params.nx;

            double center_dist = hypot(initial.circle_center.x - pos.x, initial.circle_center.y - pos.y);

            bool is_within_cube = (initial.square_from.x <= pos.x && pos.x < initial.square_to.x) && 
			    (initial.square_from.y <= pos.y && pos.y < initial.square_to.y);

            double circle_normed_sdf = (initial.circle_outer_radius - center_dist) / (initial.circle_outer_radius - initial.circle_inner_radius);
            if(circle_normed_sdf > 1)
                circle_normed_sdf = 1;
            if(circle_normed_sdf < 0)
                circle_normed_sdf = 0;

            double cube_sdf = is_within_cube ? 1 : 0;
            double factor = cube_sdf > circle_normed_sdf ? cube_sdf : circle_normed_sdf;

            initial_phi_map[i] = (Real) (factor*initial.inside_phi + (1 - factor)*initial.outside_phi);
            initial_T_map[i] = (Real) (factor*initial.inside_T + (1 - factor)*initial.outside_T);
        }
    }
}

#include <filesystem>

template <typename Func>
struct Defered
{
    Func func;
    Defered(Func f) : func(f) {} 
    ~Defered() { func(); }
};

#define CONCAT2(x, y) x ## y
#define CONCAT(x, y) CONCAT2(x, y)
#define UNIQUE(name) CONCAT(name, __LINE__)

//This will only work with c++17 or later but whatever
#define defer(statement) auto UNIQUE(_defered_) = Defered{[&]{ statement; }}

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

void simulation_state_reload(App_State* app, Allen_Cahn_Config config)
{
    //@NOTE: I am failing to link to this TU from nvcc without using at least one cuda function
    // so here goes something harmless.
    cudaGetErrorString(cudaSuccess);
    
    int ny = config.params.ny;
    int nx = config.params.nx;

    size_t bytes_size = (size_t) (ny * nx) * sizeof(Real);
    Real* initial_F = (Real*) malloc(bytes_size);
    Real* initial_U = (Real*) malloc(bytes_size);
    defer(free(initial_F));
    defer(free(initial_U));

    app->used_states = solver_type_required_history(config.solver);
    sim_solver_reinit(&app->solver, config.solver, nx, ny);
    sim_states_reinit(app->states, app->used_states, config.solver, nx, ny);
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

    app->init_time = time(NULL);   // get time now
}


#ifdef COMPILE_GRAPHICS
#include "gl.h"
#include <GLFW/glfw3.h>
// #include "external/glfw/glfw3.h"

void glfw_resize_func(GLFWwindow* window, int width, int heigth);
void glfw_key_func(GLFWwindow* window, int key, int scancode, int action, int mods);
#endif

bool save_netcfd_file(App_State* app);


int main()
{
    Allen_Cahn_Config config = {0};
    if(allen_cahn_read_config("config.ini", &config) == false)
        return 1;

    if(config.run_tests)
        run_tests();
    if(config.run_benchmarks)
        run_benchmarks(config.params.nx * config.params.ny);
    if(config.run_simulation == false)
        return 0;

    App_State app_data = {0};
    App_State* app = (App_State*) &app_data;
    app->is_in_step_mode = true;
    app->remaining_steps = 0;
    app->step_by = 1;
    
    simulation_state_reload(app, config);
    
    if(config.snapshots.snapshot_initial_conditions)
        save_netcfd_file(app);

    #define LOG_INFO_CONFIG_FLOAT(var) LOG_INFO("config", #var " = %.2lf", (double) config.params.var);
    #define LOG_INFO_CONFIG_INT(var) LOG_INFO("config", #var " = %i", (int) config.params.var);

    LOG_INFO("config", "solver = %s", solver_type_to_cstring(config.solver));

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
    LOG_INFO_CONFIG_FLOAT(Tinit);

    LOG_INFO_CONFIG_FLOAT(S);
    LOG_INFO_CONFIG_FLOAT(theta0);

    #undef LOG_INFO_CONFIG_FLOAT
    #undef LOG_INFO_CONFIG_INT

    #ifndef COMPILE_GRAPHICS
    config.interactive_mode = false;
    #endif // COMPILE_GRAPHICS

    //OPENGL setup
    if(config.interactive_mode)
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

        double curr_real_time = 0;
	    while (!glfwWindowShouldClose(window))
        {
            double frame_start_time = clock_s();

            double next_snapshot_every = (double) (snapshot_every_i + 1) * config.snapshots.snapshot_every;
            double next_snapshot_times = (double) (snapshot_times_i + 1) * config.stop_after / config.snapshots.snapshot_times;

            if(curr_real_time >= next_snapshot_every)
            {
                snapshot_every_i += 1;
                save_netcfd_file(app);
            }

            if(curr_real_time >= next_snapshot_times && end_reached == false)
            {
                snapshot_times_i += 1;
                save_netcfd_file(app);
            }

            if(curr_real_time >= config.stop_after && end_reached == false)
            {
                LOG_INFO("app", "reached stop time %lfs. Simulation paused.", config.stop_after);
                app->is_in_step_mode = true;
                end_reached = true;
            }

            bool update_screen = frame_start_time - render_last_time > 1.0/RENDER_FREQ;
            bool update_frame_time_display = frame_start_time - time_display_last_time > 1.0/FPS_DISPLAY_FREQ;
            bool poll_events = frame_start_time - poll_last_time > 1.0/POLL_FREQ;

            if(update_screen)
            {
                render_last_time = frame_start_time;
                int target = app->render_target;

                Sim_Map maps[SIM_MAPS_MAX] = {0};
                sim_solver_get_maps(&app->solver, app->states, app->used_states, app->iter, maps, SIM_MAPS_MAX);
                Sim_Map selected_map = {0};
                if(0 <= target && target < SIM_MAPS_MAX)
                    selected_map = maps[target];

                draw_sci_cuda_memory("main", selected_map.nx, selected_map.ny, (float) app->config.display_min, (float) app->config.display_max, config.linear_filtering, selected_map.data);
                glfwSwapBuffers(window);
            }

            if(update_frame_time_display)
            {
                glfwSetWindowTitle(window, format_string("%s step: %3.3lfms | real: %8.6lfms", solver_type_to_cstring(app->config.solver), processing_time * 1000, curr_real_time*1000).c_str());
                time_display_last_time = frame_start_time;
            }

            bool step_sym = false;
            if(app->is_in_step_mode)
                step_sym = app->remaining_steps > 0.5;
            else
                step_sym = frame_start_time - simulated_last_time > 1.0/app->step_by/FREE_RUN_SYM_FPS;

            if(step_sym)
            {
                simulated_last_time = frame_start_time;
                app->remaining_steps -= 1;

                Allen_Cahn_Stats stats = {0};
                app->config.params.do_debug = app->is_in_debug_mode;
                double solver_start_time = clock_s();
                curr_real_time += sim_solver_step(&app->solver, app->states, app->used_states, app->iter, app->config.params, &stats);
                double solver_end_time = clock_s();

                app->stats_last = stats;
                app->stats_summed.Phi_errror += stats.Phi_errror;
                app->stats_summed.T_error += stats.T_error;
                app->stats_summed.Phi_iters += stats.Phi_iters;
                app->stats_summed.T_iters += stats.T_iters;
                app->stats_summed.step_residuals = stats.step_residuals;
                for(size_t z = 0; z < sizeof(stats.L2_step_residuals) / sizeof(int); z++)
                {
                    app->stats_summed.L2_step_residuals[z] += stats.L2_step_residuals[z];
                    app->stats_summed.Lmax_step_residuals[z] += stats.Lmax_step_residuals[z];
                }
                app->stats_summed_over_iters += 1;

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
            double idle_frame_time = 1/IDLE_RENDER_FREQ;
            bool do_frame_limiting = simulated_last_time + 0.5 < frame_start_time;
            do_frame_limiting = true;
            if(do_frame_limiting && app->dt < idle_frame_time)
                wait_s(idle_frame_time - app->dt);
        }
    
        glfwDestroyWindow(window);
        glfwTerminate();
        #endif
    }
    else
    {
        app->is_in_debug_mode = false;
        int iters = (int) (config.stop_after / config.params.dt);
        int snapshot_every_i = 0;
        int snapshot_times_i = 0;
        bool end_reached = false;
        double start_time = clock_s();
        double last_notif_time = 0;

        double curr_real_time = 0;
        for(; app->iter <= iters; app->iter++)
        {
            double now = clock_s();
            
            double next_snapshot_every = (double) (snapshot_every_i + 1) * config.snapshots.snapshot_every;
            double next_snapshot_times = (double) (snapshot_times_i + 1) * config.stop_after / config.snapshots.snapshot_times;

            if(curr_real_time >= next_snapshot_every)
            {
                snapshot_every_i += 1;
                save_netcfd_file(app);
            }

            if(curr_real_time >= next_snapshot_times && end_reached == false)
            {
                snapshot_times_i += 1;
                save_netcfd_file(app);
            }

            //make sure we say 0% and 100%
            if(now - last_notif_time > 1 || app->iter == iters || app->iter == 0)
            {
                last_notif_time = now;
                LOG_INFO("app", "... completed %2lf%%", (double) app->iter * 100 / iters);

                if(app->iter == iters)
                    break;
            }

            Allen_Cahn_Stats stats = {0};
            app->config.params.do_debug = app->is_in_debug_mode;
            curr_real_time += sim_solver_step(&app->solver, app->states, app->used_states, app->iter, app->config.params, &stats);


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
	// make sure the viewport matches the new window dimensions; note that width and 
	// heigth will be significantly larger than specified on retina displays.
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
        if(key == GLFW_KEY_A)
        {
            app->config.params.do_anisotropy = !app->config.params.do_anisotropy;
            LOG_INFO("APP", "Anisotropy %s", app->config.params.do_anisotropy ? "true" : "false");
        }
        if(key == GLFW_KEY_D)
        {
            app->is_in_debug_mode = !app->is_in_debug_mode;
            LOG_INFO("APP", "Debug %s", app->is_in_debug_mode ? "true" : "false");
        }
        if(key == GLFW_KEY_L)
        {
            app->config.linear_filtering = !app->config.linear_filtering;
            LOG_INFO("APP", "Linear FIltering %s", app->config.linear_filtering ? "true" : "false");
        }
        if(key == GLFW_KEY_C)
        {
            app->config.params.do_corrector_loop = !app->config.params.do_corrector_loop;
            LOG_INFO("APP", "Corrector loop %s", app->config.params.do_corrector_loop ? "true" : "false");
        }
        if(key == GLFW_KEY_S)
        {
            LOG_INFO("APP", "On demand snapshot triggered");
            save_netcfd_file(app);
        }
        if(key == GLFW_KEY_R)
        {
            LOG_INFO("APP", "Input range to display in form 'MIN space MAX'");

            double new_display_max = app->config.display_max;
            double new_display_min = app->config.display_min;
            if(scanf("%lf %lf", &new_display_min, &new_display_max) != 2)
            {
                LOG_INFO("APP", "Bad range syntax!");
            }
            else
            {
                LOG_INFO("APP", "displaying range [%.2lf, %.2lf]", new_display_min, new_display_max);
                app->config.display_max = (Real) new_display_max;
                app->config.display_min = (Real) new_display_min;
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

        uint8_t new_render_target = (uint8_t) -1;
        //Switching of render targets. 0 is Phi 1 is T
        if(key == GLFW_KEY_F1) new_render_target= 0;
        if(key == GLFW_KEY_F2) new_render_target= 1;
        if(key == GLFW_KEY_F3) new_render_target= 2;
        if(key == GLFW_KEY_F4) new_render_target= 3;
        if(key == GLFW_KEY_F5) new_render_target= 4;
        if(key == GLFW_KEY_F6) new_render_target= 5;
        if(key == GLFW_KEY_F7) new_render_target= 6;
        if(key == GLFW_KEY_F8) new_render_target= 7;
        if(key == GLFW_KEY_F9) new_render_target= 8;
        if(key == GLFW_KEY_F10) new_render_target= 9;
        
        if(new_render_target != (uint8_t) -1)
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


bool save_netcfd_file(App_State* app)
{
    enum {NDIMS = 2};
    int dataset_ID = 0;
    int dim_ids[NDIMS] = {0};
    Allen_Cahn_Params params = app->config.params;

    int iter = (int) app->iter;
    double real_time = (Real) app->iter * app->config.params.dt;
    
    if(app->config.snapshots.snapshot_initial_conditions == false)
        app->snapshot_index += 1;

    //creates a path in the form "path/to/snapshots/prefix_2024_03_01__10_22_43_postfix/0001.nc"
    std::string composed_filename;

    {
        Allen_Cahn_Snapshots* snapshots = &app->config.snapshots;
        tm* t = localtime(&app->init_time);

        std::filesystem::path snapshot_folder = snapshots->folder;

        std::string folder = format_string("%s%04i-%02i-%02i__%02i-%02i-%02i__%s%s", 
            snapshots->prefix.data(), 
            t->tm_year + 1900, t->tm_mon, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec,
            solver_type_to_cstring(app->config.solver),
            snapshots->postfix.data()
        );
        std::error_code error = {};
        if(snapshot_folder.empty() == false)
            std::filesystem::create_directory(snapshot_folder, error);

        

        snapshot_folder.append(folder.c_str()); 
        std::filesystem::create_directory(snapshot_folder, error);

        if(app->snapshot_index <= 1)
        {
            std::filesystem::path config_path = snapshot_folder; 
            config_path.append("config.ini");
            file_write_entire(config_path.c_str(), app->config.entire_config_file);
        }

        //Saving of stats!
        if(app->config.params.do_stats)
        {
            std::filesystem::path stats_path = snapshot_folder; 
            stats_path.append(format_string("stats_%04i.txt", app->snapshot_index));

            Allen_Cahn_Stats cur_stats = app->stats_last;
            Allen_Cahn_Stats avg_stats = app->stats_summed;
            int iters = app->stats_summed_over_iters;
            if(iters <= 0)
                iters = 1;
            avg_stats.Phi_errror /= iters;
            avg_stats.T_error /= iters;
            avg_stats.Phi_iters /= iters;
            avg_stats.T_iters /= iters;
            for(size_t z = 0; z < sizeof(avg_stats.L2_step_residuals) / sizeof(int); z++)
            {
                avg_stats.L2_step_residuals[z] /= iters;
                avg_stats.Lmax_step_residuals[z] /= iters;
            }

            //Clear any stats @TODO: less OOP. This functions should not modify app state in any way!
            app->stats_summed_over_iters = 0;
            app->stats_summed = Allen_Cahn_Stats{0};

            int l_i = avg_stats.step_residuals - 1;
            if(l_i < 0)
                l_i = 0;

            file_write_entire(stats_path.c_str(), format_string(
                "     L2     Lmax\n"
                "avg: %e, %e\n" 
                "cur: %e, %e\n", 
                avg_stats.L2_step_residuals[l_i], avg_stats.Lmax_step_residuals[l_i],
                cur_stats.L2_step_residuals[l_i], cur_stats.Lmax_step_residuals[l_i]
            ));
        }

        std::filesystem::path snapshot_path = snapshot_folder; 
        snapshot_path.append(format_string("%s_%04i.nc", solver_type_to_cstring(app->config.solver), app->snapshot_index).c_str());

        composed_filename = snapshot_path.c_str();

        if(error)
        {
            LOG_INFO("app", "encoutered error '%s' while creating a folder on the final snapshot path '%s'",
                error.message().c_str(), composed_filename.c_str()
            );

            return false;
        }
    }

    if(app->config.snapshots.snapshot_initial_conditions == true)
        app->snapshot_index += 1;
        
    const char* filename = composed_filename.data();

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
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "Tinit", real_type, 1, &params.Tinit);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "S", real_type, 1, &params.S);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "nx", real_type, 1, &params.nx);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "theta0", real_type, 1, &params.theta0);
    nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "do_anisotropy", NC_BYTE, 1, &params.do_anisotropy);

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

    defer(free(F));
    defer(free(U));

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

    if(nc_error != NC_NOERR) {
        LOG_INFO("APP", "NetCDF error while outputing data: %s.", nc_strerror(nc_error));
        return false;
    }

    return true;
}
