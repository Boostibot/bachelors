#if 0
    #if defined(__NVCOMPILER) || defined(__NVCC__) 


    #pragma nv_diag_suppress 186 /* Dissable: pointless comparison of unsigned integer with zero*/
    #pragma nv_diag_suppress 177 /* Dissable: was declared but never referenced for functions */

    //missing intrinsic?
    void __builtin_ia32_serialize() {}
    #endif

    #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations" /* Dissable deprecated warnings (required for cudaMemcpyToArray for OpenGL interop) */
    #pragma GCC diagnostic ignored "-Wunused-function"
    // #pragma GCC diagnostic ignored "-Wunused-local-typedef"
    #pragma GCC diagnostic ignored "-Wmissing-braces"
    #endif
#endif

#define _CRT_SECURE_NO_WARNINGS
#define JOT_ALL_IMPL
#include "config.h"
#include "integration_methods.h"
#include "kernel.h"
#include "cuda_util.h"
#include "log.h"
#include "assert.h"

#include <cmath>
#include <cstddef>
#include <chrono>
#include <algorithm>
#include <filesystem>

const double FPS_DISPLAY_FREQ = 50000;
const double RENDER_FREQ = 30;
const double POLL_FREQ = 30;

const double FREE_RUN_SYM_FPS = 200;

#define WINDOW_TITLE        "sym"
#define TARGET_FRAME_TIME	16.0 
#define DEF_WINDOW_WIDTH	800 
#define DEF_WINDOW_HEIGHT	800
#define DO_GRAPHICAL_BUILD  

static double clock_s();

typedef struct App_State {
    bool is_in_step_mode;
    uint8_t render_target; //0:phi 1:T rest: debug_option at index - 2; 
    double remaining_steps;
    double step_by;

    Allen_Cahn_Config config;
    Allen_Cahn_Config last_config;
    Allen_Cahn_Maps maps;

    real_t* initial_phi_map;
    real_t* initial_T_map;
} App_State;

bool simulation_state_is_hard_reload(const Allen_Cahn_Config* config, const Allen_Cahn_Config* last_config)
{
    if(config->params.mesh_size_x != last_config->params.mesh_size_x || config->params.mesh_size_y != last_config->params.mesh_size_y)
        return true;

    //@HACK: proper comparison of fields is replaced by comparison of the flat memory because I am lazy.
    Allen_Cahn_Initial_Conditions ini = config->initial_conditions;
    Allen_Cahn_Initial_Conditions ini_last = last_config->initial_conditions;
    ini.start_snapshot = "";
    ini_last.start_snapshot = "";

    if(memcmp(&ini, &ini_last, sizeof ini_last) != 0)
        return true;

    if(config->initial_conditions.start_snapshot == last_config->initial_conditions.start_snapshot)
        return true;

    return false;
}

void allen_cahn_set_initial_conditions(real_t* initial_phi_map, real_t* initial_T_map, Allen_Cahn_Config config);

void simulation_state_reload(App_State* state, Allen_Cahn_Config* config)
{
    if(config == NULL || simulation_state_is_hard_reload(&state->last_config, config))
    {
        free(state->initial_phi_map); state->initial_phi_map = NULL;
        free(state->initial_T_map); state->initial_T_map = NULL;

        cudaFree(state->maps.Phi[0]); state->maps.Phi[0] = NULL;
        cudaFree(state->maps.Phi[1]); state->maps.Phi[1] = NULL;
        cudaFree(state->maps.T[0]); state->maps.T[0] = NULL;
        cudaFree(state->maps.T[1]); state->maps.T[1] = NULL;

        for(size_t i = 0; i < ALLEN_CAHN_DEBUG_MAPS; i++)
        {
            cudaFree(state->maps.debug_maps[i]);
            state->maps.debug_maps[i] = NULL;
        }

        if(config != NULL)
        {
            size_t pixel_count_x = (size_t) config->params.mesh_size_x;
            size_t pixel_count_y = (size_t) config->params.mesh_size_y;
            size_t pixel_count = pixel_count_x * pixel_count_y;
            size_t byte_count = pixel_count * sizeof(real_t);

            state->initial_phi_map = (real_t*) malloc(byte_count);
            state->initial_T_map = (real_t*) malloc(byte_count);
            allen_cahn_set_initial_conditions(state->initial_phi_map, state->initial_T_map, *config);

            CUDA_TEST(cudaMalloc((void**) &state->maps.Phi[0], byte_count));
            CUDA_TEST(cudaMalloc((void**) &state->maps.Phi[1], byte_count));
            CUDA_TEST(cudaMalloc((void**) &state->maps.T[0], byte_count));
            CUDA_TEST(cudaMalloc((void**) &state->maps.T[1], byte_count));
            for(size_t i = 0; i < ALLEN_CAHN_DEBUG_MAPS; i++)
            {
                CUDA_TEST(cudaMalloc((void**) &state->maps.debug_maps[i], byte_count));
                state->maps.debug_request[i] = true;
            }

            CUDA_TEST(cudaMemcpy(state->maps.Phi[0], state->initial_phi_map, byte_count, cudaMemcpyHostToDevice));
            CUDA_TEST(cudaMemcpy(state->maps.T[0], state->initial_T_map, byte_count, cudaMemcpyHostToDevice));
        }
    }

    if(config)
    {
        state->config = *config;
        state->last_config = *config;

        std::filesystem::create_directory(config->snapshots.folder);
    }
}

void allen_cahn_custom_config(Allen_Cahn_Config* out_config);

void allen_cahn_set_initial_conditions(real_t* initial_phi_map, real_t* initial_T_map, Allen_Cahn_Config config)
{
    Allen_Cahn_Params params = config.params;
    Allen_Cahn_Initial_Conditions initial = config.initial_conditions;
    for(size_t y = 0; y < (size_t) params.mesh_size_y; y++)
    {
        for(size_t x = 0; x < (size_t) params.mesh_size_x; x++)
        {
            Vec2 pos = Vec2{((real_t) x+0.5f) / params.mesh_size_x * params.L0, ((real_t) y+0.5f) / params.mesh_size_y * params.L0}; 
            size_t i = x + y*(size_t) params.mesh_size_x;

            real_t center_dist = (real_t) hypot(initial.circle_center.x - pos.x, initial.circle_center.y - pos.y);
            bool is_within_circle = center_dist <= initial.circle_radius;
            
            bool is_within_cube = (initial.square_from.x <= pos.x && pos.x < initial.square_to.x) && 
			    (initial.square_from.y <= pos.y && pos.y < initial.square_to.y);

		    if(is_within_cube || is_within_circle)
		    {
                initial_phi_map[i] = initial.inside_phi;
                initial_T_map[i] = initial.inside_T;
		    }
            else
            {
                initial_phi_map[i] = initial.outside_phi;
                initial_T_map[i] = initial.outside_T;
            }
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
    
    simulation_state_reload(app, &config);

    LOG_INFO("config", "mesh_size_x = %d", config.params.mesh_size_x);
    LOG_INFO("config", "mesh_size_y = %d", config.params.mesh_size_y);
    LOG_INFO("config", "L0 = %f", config.params.L0); 

    LOG_INFO("config", "dt = %f", config.params.dt);
    LOG_INFO("config", "L = %f", config.params.L);
    LOG_INFO("config", "xi = %f", config.params.xi);
    LOG_INFO("config", "a = %f", config.params.a);
    LOG_INFO("config", "b = %f", config.params.b);
    LOG_INFO("config", "alpha = %f", config.params.alpha);
    LOG_INFO("config", "beta = %f", config.params.beta);
    LOG_INFO("config", "Tm = %f", config.params.Tm);
    LOG_INFO("config", "Tinit = %f", config.params.Tinit);

    LOG_INFO("config", "S = %f", config.params.S);
    LOG_INFO("config", "m = %f", config.params.m);
    LOG_INFO("config", "theta0 = %f", config.params.theta0);
    LOG_INFO("config", "do_anisotropy = %d", (int) config.params.do_anisotropy);

    //CUDA setup
    int device_id = 0;
    CUDA_TEST(cudaSetDevice(device_id));
    LOG_INFO("App", "Device set");
    
    int device_processor_count = 0;
    CUDA_TEST(cudaDeviceGetAttribute(&device_processor_count, cudaDevAttrMultiProcessorCount, device_id));
    LOG_INFO("App", "device_processor_count %d", device_processor_count);

    #ifndef DO_GRAPHICAL_BUILD
    config.interactive_mode = false;
    #endif // DO_GRAPHICAL_BUILD


    //OPENGL setup
    if(config.interactive_mode)
    {   
        #ifdef DO_GRAPHICAL_BUILD
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

            if(update_screen)
            {
                render_last_time = frame_start_time;

                real_t* selected_map = NULL;
                switch (app->render_target)
                {
                    case 0: selected_map = app->maps.Phi[0]; break;
                    case 1: selected_map = app->maps.T[0]; break;
                
                    default: {
                        if(0 <= app->render_target - 2 && app->render_target - 2 < ALLEN_CAHN_DEBUG_MAPS)
                            selected_map = app->maps.debug_maps[app->render_target - 2];
                    } break;
                }

                if(selected_map)
                    draw_sci_cuda_memory("main", app->config.params.mesh_size_x, app->config.params.mesh_size_y, config.display_min, config.display_max, config.linear_filtering, selected_map);
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
                CUDA_TEST(kernel_step(&app->maps, app->config.params, device_processor_count, frame_counter));

                double end_start_time = clock_s();
                double delta = end_start_time - frame_start_time;

                frame_time_sum += delta;
                frame_counter += 1;
            }

            //Only swap buffers when draw happened
            if(update_screen)
                glfwSwapBuffers(window);

            if(poll_events)
            {
                poll_last_time = frame_start_time;
                glfwPollEvents();
            }
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

            CUDA_TEST(kernel_step(&app->maps, app->config.params, device_processor_count, i));
        }
        double end_time = clock_s();
        double runtime = end_time - start_time;

        LOG_INFO("app", "Finished");
        LOG_INFO("app", "runtime: %.2lfs | iters: %lli | average step time: %.2lf ms", runtime, (long long) iters, runtime / (double) iters * 1000 * 1000);
    }

    return 0;    
}

#ifdef DO_GRAPHICAL_BUILD

#define MIN(a, b)   ((a) < (b) ? (a) : (b))
#define MAX(a, b)   ((a) > (b) ? (a) : (b))

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
            app->is_in_step_mode = !app->is_in_step_mode;

        uint8_t new_render_target = app->render_target;

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
        
        // if(new_render_target != app->render_target)
        {
            app->render_target = new_render_target;
            const char* render_target_name = "<EMPTY>";
            switch (new_render_target)
            {
                case 0: render_target_name = "Phi"; break;
                case 1: render_target_name = "T"; break;
            
                default: {
                    if(0 <= new_render_target - 2 && new_render_target - 2 < ALLEN_CAHN_DEBUG_MAPS)
                    {
                        const char* name = app->maps.debug_names[new_render_target - 2];
                        if(strlen(name) != 0)
                            render_target_name = name;
                    }
                } break;
            }

            LOG_INFO("APP", "rendering: %s", render_target_name);
        }

        double iters_before = app->step_by;
        if(key == GLFW_KEY_O)
            app->step_by = app->step_by*1.3 + 1;
        if(key == GLFW_KEY_P)
            app->step_by = MAX((app->step_by - 1)/1.3, 1.0);

        if(iters_before != app->step_by)
            LOG_INFO("APP", "Steps per iter %lf", app->step_by);
    }
}
#endif

void allen_cahn_custom_config(Allen_Cahn_Config* out_config)
{
    const int _SIZE_X = 1024;
    const int _SIZE_Y = _SIZE_X;
    const real_t _dt = 1.0f/200;
    const real_t _alpha = 0.5;
    const real_t _L = 2;
    const real_t _xi = 0.00411f;
    const real_t _a = 2;
    const real_t _b = 1;
    const real_t _beta = 8;
    const real_t _Tm = 1;
    const real_t _Tini = 0;
    const real_t _L0 = 4;

    Allen_Cahn_Scale scale = {0};
    scale.L0 = _L0 / (real_t) _SIZE_X;
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

static double clock_s()
{
    static int64_t init_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int64_t now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    double unit = (double) std::chrono::high_resolution_clock::period::den;
    double clock = (double) (now - init_time) / unit;
    return clock;
}