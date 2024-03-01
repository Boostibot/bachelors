
#define JOT_ALL_IMPL
#include "config.h"
#include "integration_methods.h"
#include "kernel.h"
#include "log.h"
#include "assert.h"

#include <cmath>
#include <stddef.h>

#include "assert.h"
#include <cuda_runtime.h> //without this it fails to link
#include <netcdf>

const double FPS_DISPLAY_FREQ = 50000;
const double RENDER_FREQ = 30;
const double IDLE_RENDER_FREQ = 20;
const double POLL_FREQ = 30;

const double FREE_RUN_SYM_FPS = 200;

#define WINDOW_TITLE        "sym"
#define TARGET_FRAME_TIME	16.0 
#define DEF_WINDOW_WIDTH	800 
#define DEF_WINDOW_HEIGHT	800
#define DO_GRAPHICAL_BUILD  

static double clock_s();
static void wait_s(double seconds);

typedef struct App_State {
    bool is_in_step_mode;
    bool is_in_debug_mode;
    uint8_t render_target; //0:phi 1:T rest: debug_option at index - 2; 
    double remaining_steps;
    double step_by;
    double dt;
    size_t iter;

    Allen_Cahn_Config   config;
    Explicit_State      expli;
    Debug_State         debug;
    Semi_Implicit_State impli;

    Real* initial_F;
    Real* initial_U;
} App_State;

void allen_cahn_set_initial_conditions(Real* initial_phi_map, Real* initial_T_map, Allen_Cahn_Config config);

#include <filesystem>
void simulation_state_reload(App_State* state, Allen_Cahn_Config config)
{
    //@NOTE: I am failing to link to this TU from nvcc without using at least one cuda function
    // so here goes something harmless.
    cudaGetErrorString(cudaSuccess);
    int n = config.params.mesh_size_y;
    int m = config.params.mesh_size_x;

    size_t bytes_size = (size_t) (n * m) * sizeof(Real);
    state->initial_F = (Real*) realloc(state->initial_F, bytes_size);
    state->initial_U = (Real*) realloc(state->initial_U, bytes_size);

    semi_implicit_state_resize(&state->impli, n, m);
    explicit_state_resize(&state->expli, n, m);
    debug_state_resize(&state->debug, n, m);
    allen_cahn_set_initial_conditions(state->initial_F, state->initial_U, config);

    size_t i = state->iter % ALLEN_CAHN_HISTORY;
    device_modify(state->impli.F[i], state->initial_F, bytes_size, MODIFY_UPLOAD);
    device_modify(state->impli.U[i], state->initial_U, bytes_size, MODIFY_UPLOAD);
    device_modify(state->expli.F[i], state->initial_F, bytes_size, MODIFY_UPLOAD);
    device_modify(state->expli.U[i], state->initial_U, bytes_size, MODIFY_UPLOAD);
    state->config = config;

    if(config.snapshots.folder.size() > 0)
        std::filesystem::create_directory(config.snapshots.folder);
}

void allen_cahn_custom_config(Allen_Cahn_Config* out_config);

void allen_cahn_set_initial_conditions(Real* initial_phi_map, Real* initial_T_map, Allen_Cahn_Config config)
{
    Allen_Cahn_Params params = config.params;
    Allen_Cahn_Initial_Conditions initial = config.initial_conditions;
    for(size_t y = 0; y < (size_t) params.mesh_size_y; y++)
    {
        for(size_t x = 0; x < (size_t) params.mesh_size_x; x++)
        {
            Vec2 pos = Vec2{((Real) (x+0.5)) / params.mesh_size_x * params.L0, ((Real) (y+0.5)) / params.mesh_size_y * params.L0}; 
            size_t i = x + y*(size_t) params.mesh_size_x;

            Real center_dist = (Real) hypot(initial.circle_center.x - pos.x, initial.circle_center.y - pos.y);

            bool is_within_cube = (initial.square_from.x <= pos.x && pos.x < initial.square_to.x) && 
			    (initial.square_from.y <= pos.y && pos.y < initial.square_to.y);

            Real circle_normed_sdf = (initial.circle_outer_radius - center_dist) / (initial.circle_outer_radius - initial.circle_inner_radius);
            if(circle_normed_sdf > 1)
                circle_normed_sdf = 1;
            if(circle_normed_sdf < 0)
                circle_normed_sdf = 0;

            Real cube_sdf = is_within_cube ? 1 : 0;
            Real factor = cube_sdf > circle_normed_sdf ? cube_sdf : circle_normed_sdf;

            initial_phi_map[i] = factor*initial.inside_phi + (1 - factor)*initial.outside_phi;
            initial_T_map[i] = factor*initial.inside_T + (1 - factor)*initial.outside_T;
        }
    }
}

#ifdef DO_GRAPHICAL_BUILD
#include "gl.h"

#include <GLFW/glfw3.h>
// #include "extrenal/glfw/glfw3.h"

void glfw_resize_func(GLFWwindow* window, int width, int heigth);
void glfw_key_func(GLFWwindow* window, int key, int scancode, int action, int mods);
#endif

void save_netcfd_file(App_State* app);

int main()
{
    //Read config
    Allen_Cahn_Config config = {0};
    if(allen_cahn_read_config("config.ini", &config) == false)
        return 1;

    App_State app_ = {0};
    App_State* app = (App_State*) &app_;
    app->is_in_step_mode = true;
    app->remaining_steps = 0;
    app->step_by = 1;
    
    simulation_state_reload(app, config);

    #define LOG_INFO_CONFIG_REAL(var) LOG_INFO("config", #var " = " REAL_FMT_LOW_PREC, config.params.var);
    #define LOG_INFO_CONFIG_INT(var) LOG_INFO("config", #var " = %i", config.params.var);

    LOG_INFO_CONFIG_INT(mesh_size_x);
    LOG_INFO_CONFIG_INT(mesh_size_y);
    LOG_INFO_CONFIG_REAL(L0);

    LOG_INFO_CONFIG_REAL(dt); 
    LOG_INFO_CONFIG_REAL(L);  
    LOG_INFO_CONFIG_REAL(xi); 
    LOG_INFO_CONFIG_REAL(a);  
    LOG_INFO_CONFIG_REAL(b);
    LOG_INFO_CONFIG_REAL(alpha);
    LOG_INFO_CONFIG_REAL(beta);
    LOG_INFO_CONFIG_REAL(Tm);
    LOG_INFO_CONFIG_REAL(Tinit);

    LOG_INFO_CONFIG_REAL(S);
    LOG_INFO_CONFIG_REAL(m);
    LOG_INFO_CONFIG_REAL(theta0);
    LOG_INFO("config", "do_anisotropy = %s", config.params.do_anisotropy ? "true" : "false");

    #undef LOG_INFO_CONFIG_REAL
    #undef LOG_INFO_CONFIG_INT

    #ifndef DO_GRAPHICAL_BUILD
    config.interactive_mode = false;
    #endif // DO_GRAPHICAL_BUILD


    //OPENGL setup
    if(config.interactive_mode)
    {   
        #ifdef DO_GRAPHICAL_BUILD
        app->is_in_debug_mode = true;
        TEST_MSG(glfwInit(), "Failed to init glfw");

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
        TEST_MSG(window != NULL, "Failed to make glfw window");

        //glfwSetWindowUserPointer(window, &app);
        glfwMakeContextCurrent(window);
        glfwSetWindowUserPointer(window, app);
        glfwSetFramebufferSizeCallback(window, glfw_resize_func);
        glfwSetKeyCallback(window, glfw_key_func);
        glfwSwapInterval(0);
        gl_init((void*) glfwGetProcAddress);

        size_t frame_counter = 0;
        double frame_time_sum = 0;
    
        double time_display_last_time_sum = 0;
        double time_display_last_time = 0;
    
        double render_last_time = 0;
        double simulated_last_time = 0;
        double poll_last_time = 0;

	    while (!glfwWindowShouldClose(window))
        {
            double frame_start_time = clock_s();

            bool update_screen = frame_start_time - render_last_time > 1.0/RENDER_FREQ;
            bool update_frame_time_display = frame_start_time - time_display_last_time > 1.0/FPS_DISPLAY_FREQ;
            bool poll_events = frame_start_time - poll_last_time > 1.0/POLL_FREQ;

            //@TODO: Add idle and not idle frequency to lower resource usage!
            if(update_screen)
            {
                render_last_time = frame_start_time;

                Real* selected_map = NULL;
                switch (app->render_target)
                {
                    case 0: selected_map = app->impli.F[frame_counter % ALLEN_CAHN_HISTORY]; break;
                    case 1: selected_map = app->impli.U[frame_counter % ALLEN_CAHN_HISTORY]; break;
                
                    default: {
                        if(0 <= app->render_target - 2 && app->render_target - 2 < ALLEN_CAHN_DEBUG_MAPS)
                            selected_map = app->debug.maps[app->render_target - 2];
                    } break;
                }

                // printf("rendering map %i\n", (int) frame_counter);
                if(selected_map)
                    draw_sci_cuda_memory("main", app->config.params.mesh_size_x, app->config.params.mesh_size_y, (float) app->config.display_min, (float) app->config.display_max, config.linear_filtering, selected_map);

                glfwSwapBuffers(window);
            }

            if(update_frame_time_display)
            {
                double time_sum_delta = frame_time_sum - time_display_last_time_sum;
                if(time_sum_delta != 0)
                    glfwSetWindowTitle(window, std::to_string(time_sum_delta).c_str());

                time_display_last_time = frame_start_time;
                time_display_last_time_sum = frame_time_sum;
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

                Debug_State* debug = app->is_in_debug_mode ? &app->debug : NULL;
                // explicit_solver_step(&app->expli, debug, app->config.params, frame_counter);
                semi_implicit_solver_step(&app->impli, debug, app->config.params, frame_counter);

                double end_start_time = clock_s();
                double delta = end_start_time - frame_start_time;

                frame_time_sum += delta;
                frame_counter += 1;
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
        size_t iters = (size_t) (config.snapshots.sym_time / config.params.dt);
        
        double start_time = clock_s();
        double last_notif_time = 0;
        for(size_t i = 0; i < iters; i++)
        {
            double now = clock_s();
            if(now - last_notif_time > 1)
            {
                last_notif_time = now;
                LOG_INFO("app", "... completed %i%%", (int) (i * 100 / iters));
            }

            explicit_solver_step(&app->expli, NULL, app->config.params, i);
        }
        double end_time = clock_s();
        double runtime = end_time - start_time;

        LOG_INFO("app", "Finished");
        LOG_INFO("app", "runtime: %.2lfs | iters: %lli | average step time: %.2lf ms", runtime, (long long) iters, runtime / (double) iters * 1000 * 1000);
    }

    return 0;    
}

#ifdef DO_GRAPHICAL_BUILD

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

        if(key == GLFW_KEY_R)
        {
            LOG_INFO("APP", "Input range to display in form 'MIN space MAX'");

            Real new_display_max = app->config.display_max;
            Real new_display_min = app->config.display_min;
            if(scanf(REAL_FMT " " REAL_FMT, &new_display_min, &new_display_max) != 2)
            {
                LOG_INFO("APP", "Bad range syntax!");
            }
            else
            {
                LOG_INFO("APP", "displaying range [" REAL_FMT_LOW_PREC ", " REAL_FMT_LOW_PREC "]", new_display_min, new_display_max);
                app->config.display_max = (Real) new_display_max;
                app->config.display_min = (Real) new_display_min;
            }
        }

        if(key == GLFW_KEY_S)
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
            app->render_target = new_render_target;
            const char* render_target_name = "<EMPTY>";
            switch (new_render_target)
            {
                case 0: 
                    render_target_name = "Phi"; 
                    break;
                case 1: 
                    render_target_name = "T"; 
                    break;
            
                default: {
                    int index = new_render_target - 2; 
                    if(0 <= index && index < ALLEN_CAHN_DEBUG_MAPS)
                    {
                        //Set this debug request
                        const char* name = app->debug.names[index];
                        if(strlen(name) != 0)
                            render_target_name = name;
                    }
                } break;
            }

            LOG_INFO("APP", "rendering: %s", render_target_name);
        }

    }
}
#endif
/*
void allen_cahn_custom_config(Allen_Cahn_Config* out_config)
{
    const int _SIZE_X = 1024;
    const int _SIZE_Y = _SIZE_X;
    const Real _dt = 1.0f/200;
    const Real _alpha = 0.5;
    const Real _L = 2;
    const Real _xi = 0.00411f;
    const Real _a = 2;
    const Real _b = 1;
    const Real _beta = 8;
    const Real _Tm = 1;
    const Real _Tini = 0;
    const Real _L0 = 4;

    Allen_Cahn_Scale scale = {0};
    scale.L0 = _L0 / (Real) _SIZE_X;
    scale.Tini = _Tini;
    scale.Tm = _Tm;
    scale.c = 1;
    scale.rho = 1;
    scale.lambda = 1;
    
    Allen_Cahn_Params params = {0};
    params.L0 = _L0;
    params.mesh_size_x = _SIZE_X;
    params.mesh_size_y = _SIZE_Y;
    params.L = allen_cahn_scale_latent_heat(_L, scale);
    params.xi = allen_cahn_scale_xi(_xi, scale);
    params.dt = _dt;
    params.a = _a;
    params.b = _b;
    params.alpha = allen_cahn_scale_alpha(_alpha, scale);
    params.beta = allen_cahn_scale_latent_heat(_beta, scale);
    params.Tm = _Tm;
    
    Allen_Cahn_Initial_Conditions initial_conditions = {0};
    initial_conditions.inside_phi = 1;
    initial_conditions.inside_T = 0;
    initial_conditions.outside_phi = 0;
    initial_conditions.outside_T = 0;
    initial_conditions.circle_center = Vec2{_L0 / 4, _L0 / 4};
    initial_conditions.circle_radius = _L0 / 8;
    initial_conditions.square_from = Vec2{_L0/2 - 0.3f, _L0/2 - 0.3f};
    initial_conditions.square_to = Vec2{_L0/2 + 0.3f, _L0/2 + 0.3f};

    Allen_Cahn_Snapshots snapshots = {0};
    snapshots.folder = "snapshots";
    snapshots.prefix = "v1";
    snapshots.every = 0.1f;
    snapshots.sym_time = -1;
    
    out_config->config_name = "from_code_config";
    out_config->initial_conditions = initial_conditions;
    out_config->params = params;
    out_config->snapshots = snapshots;
}
*/

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



void save_netcfd_file(App_State* app)
{
    (void) app;
    #if 0
		/* prepare the output NetCDF dataset */
        int dataset_ID = 0;
        const char* filename = "snapshot";
        Allen_Cahn_Params params = app->config.params;
        int real_type = sizeof(Real) == sizeof(double) ? NC_DOUBLE : NC_FLOAT;

        #define NC_ERROR_AND(code) (code) != NC_NOERR ? (code) :

        int dim_ids[2] = {0};
		{

			int nc_error = nc_create(filename, NC_CLOBBER, &dataset_ID);
			if(nc_error != NC_NOERR) {
				printf("NetCDF create error: %s.\n", nc_strerror(nc_error));
                return;
			}

			/* define the solution grid dimensions */
			nc_error = NC_ERROR_AND(nc_error) nc_def_dim (dataset_ID, "x", params.mesh_size_x, dim_ids + 0);
			nc_error = NC_ERROR_AND(nc_error) nc_def_dim (dataset_ID, "y", params.mesh_size_y, dim_ids + 1);
		
			if(nc_error != NC_NOERR) {
				printf("NetCDF define dim error: %s.\n", nc_strerror(nc_error));
                return;
			}
		}

		/*
		save computation parameters - these attributes may or may not be used by other postprocessing
		software, by Intertack itself or they can be extracted by the ncdump command line utility
		for informational purpose only.
		*/
		{
            /*
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
            */


            int nc_error = NC_NOERR;

			nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "mesh_size_x", NC_INT, 1, &params.mesh_size_x);
			nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "mesh_size_y", NC_INT, 1, &params.mesh_size_y);
			nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "L0", NC_INT, 1, &params.L0);

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
			nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "m", real_type, 1, &params.m);
			nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "theta0", real_type, 1, &params.theta0);
			nc_error = NC_ERROR_AND(nc_error) nc_put_att(dataset_ID, NC_GLOBAL, "do_anisotropy", NC_BYTE, 1, &params.do_anisotropy);

			if(nc_error != NC_NOERR) {
				printf("NetCDF error while outputing params: %s.\n", nc_strerror(nc_error));
                return;
			}
		}

		nc_enddef(dataset_ID);

        int Phi_ID = 0;
        int T_ID = 0;

        int nc_error = NC_NOERR;
        nc_error = NC_ERROR_AND(nc_error) nc_def_var(dataset_ID, "Phi", real_type, 2, dim_ids, &Phi_ID);
        nc_error = NC_ERROR_AND(nc_error) nc_def_var(dataset_ID, "T", real_type, 2, dim_ids, &T_ID);



		/*
		save auxiliary coordinate arrays to the NetCDF dataset. These arrays contain the respective z,y,x
		coordinates of the grid nodes. If saved as variables with the same names as the dimensions' names
		n3,n2,n1 (ordered by importance), they are taken into account by the VisIt visualization software
		in renderings of NetCDF arrays.
		*/
		{
			double * x_grid_coords, * y_grid_coords, * z_grid_coords;
			int n3_ = grid_IO_mode ? total_n3 : total_N3;
			int n2_ = grid_IO_mode ? n2 : N2;
			int n1_ = grid_IO_mode ? n1 : N1;
			int bcond_thickness_ = grid_IO_mode ? 0 : bcond_thickness;
			int i, j, k;

			/*
			allocate the auxiliary arrays. For code structure improvement, all snapshot saving technicalities
			are concentrated together. Of course, these arrays could be allocated and initialized somewhere above
			only once per each batch mode iteration, but the additional work costs nothing anyway.
			*/
			alloc_error_code = 0;	/* this is in fact redundant as alloc_error_code should still be zero */
			if( (z_grid_coords=(double *)malloc(n3_*sizeof(double))) == NULL ) alloc_error_code=1;
			else if( (y_grid_coords=(double *)malloc(n2_*sizeof(double))) == NULL ) alloc_error_code=1;
			else if( (x_grid_coords=(double *)malloc(n1_*sizeof(double))) == NULL ) alloc_error_code=1;

			if(alloc_error_code) {
				Mmprintf(logfile, "Error: Could not allocate the grid coordinate descriptor arrays.\nStop.\n");
				nc_close(dataset_ID);
				HaltAllRanks(1);
			}

			/* fill the arrays */
			for(k=0;k<n3_;k++) z_grid_coords[k] = L3 * (0.5+k-bcond_thickness_)/total_n3;
			for(j=0;j<n2_;j++) y_grid_coords[j] = L2 * (0.5+j-bcond_thickness_)/n2;
			for(i=0;i<n1_;i++) x_grid_coords[i] = L1 * (0.5+i-bcond_thickness_)/n1;

			nc_put_var_double(dataset_ID, n3_var_ID, z_grid_coords);
			nc_put_var_double(dataset_ID, n2_var_ID, y_grid_coords);
			nc_put_var_double(dataset_ID, n1_var_ID, x_grid_coords);

			/* delete the coordinate arrays */
			free(x_grid_coords);
			free(y_grid_coords);
			free(z_grid_coords);
		}
		/*
		dataset created successfully - gather the solution from all ranks
		NOTE:	Up to this point, the other ranks don't even know that the master is preparing the snapshot.
		*/

		MPIcmd=MPICMD_SNAPSHOT;
		MPI_Bcast(&MPIcmd, 1, MPI_INT, MPIrankmap[0], MPI_COMM_WORLD);

		/* collect the data and save immediately. The auxiliary (boundary condition) nodes are saved if and only if grid_IO_mode==0. */
		for(l=0;l<MPIprocs;l++) {

			int snapshot_bnd_thickness = grid_IO_mode*bcond_thickness;	/* 0 if the whole grid should be output */
			int n1_ = n1;
			int n2_ = n2;
			int n3_;
			int first_row_;

			/*
			calculate the n3, first_row variables as they are in rank l
			(including the master itself if l==0)
			*/
  			n3_ = total_n3/MPIprocs;
			first_row_ = l*n3_;
			if(l < total_n3%MPIprocs) {
				n3_++;
				first_row_ += l;
			} else
				first_row_ += total_n3%MPIprocs;

			/*
			when saving the whole grid, we have to account for the number of ranks and the position of the
			block corresponding to rank l in the grid. The communication arrays ('boundary condition' in the Z
			direction) must not be saved to the output dataset.

			Here n3_ and first_row_ are modified so that they reflect the dimension of the block in the Y direction and
			the position of its first block in the output dataset. If grid_IO_mode==1, then n3_ and first_row_
			have the same value as the respective variables without the trailing underscore in rank l.
			*/
			if(!grid_IO_mode) {
				n1_=N1; n2_=N2;
				if(l==0) n3_ += bcond_thickness; else first_row_ += bcond_thickness;
				if(l==MPIprocs-1) n3_ += bcond_thickness;
			}

			if(l) {
				/* subgridsize in rank 0 may be several rows greater than the actual amount of data, but that doesn't matter */
				MPI_Recv(u_cache, subgridsize, MPI_DOUBLE, MPIrankmap[l], MPIMSG_GRID_U, MPI_COMM_WORLD, &MPIstat);
				MPI_Recv(p_cache, subgridsize, MPI_DOUBLE, MPIrankmap[l], MPIMSG_GRID_P, MPI_COMM_WORLD, &MPIstat);
			} else {
				int i, j, k;
				double * cache_ptr;
				FLOAT * ptr;

				/*
				update the boundary conditions in the event that the auxiliary nodes are required to be saved.
				The bcond_setup() function is defined in the file equation.c
				*/
				if(!grid_IO_mode) bcond_setup(eqSystem.t, solution);

				/*
				Transcribe the solution to the cache before saving (for additional information, see the same
				procedure in the initial conditions loading part. If grid_IO_mode==1, the boundary conditions
				in the plane X-Y are skipped here (the boundary conditions in the direction of Z are skipped
				automatically due to the memory organization of the solution).
				*/

				/* 1. transcribe the temperature field */
				cache_ptr = u_cache;
				for(k=0;k<n3_;k++)
					for(j=0;j<n2_;j++) {
						ptr = u + (k+snapshot_bnd_thickness) * rowsize + (j+snapshot_bnd_thickness) * N1 + snapshot_bnd_thickness;
						for(i=0;i<n1_;i++) *(cache_ptr++)=*(ptr++);
					}

				/* 2. transcribe the phase field */
				cache_ptr = p_cache;
				for(k=0;k<n3_;k++)
					for(j=0;j<n2_;j++) {
						ptr = p + (k+snapshot_bnd_thickness) * rowsize + (j+snapshot_bnd_thickness) * N1 + snapshot_bnd_thickness;
						for(i=0;i<n1_;i++) *(cache_ptr++)=*(ptr++);
					}
			}

			/* save the block to the file */
			{
				/*
				prepare the data subgrid starting corner and dimensions, as required for the call to
				the nc_get_vara_double() function. For more information, see the NetCDF documentation.
				*/
				size_t nc_start[3] = { first_row_, 0, 0 };
				size_t nc_count[3] = { n3_, n2_, n1_ };

				nc_put_vara_double(dataset_ID, u_var_ID, nc_start, nc_count, u_cache);
				nc_put_vara_double(dataset_ID, p_var_ID, nc_start, nc_count, p_cache);
			}

			/* move the progress meter (each star represents data collection from one process) */
			Mmprintf(logfile, "*"); fflush(stdout);
		}

		nc_close(dataset_ID);
		Mmprintf(logfile, "] Done in %s\n", format_time(MPI_Wtime()-AUX_time));

		/* delete the snapshot trigger file */
		if(is_on_demand_snapshot) unlink(snapshot_trigger_file);

		commit_logfile(0);	/* update the log file on disk (non-forced update - see commit_logfile()) */
	}
    #endif
}
