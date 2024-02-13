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

// #define USE_OPENGL

#define JOT_ALL_IMPL
#include "lib/platform.h"

#include "config.h"
#include "integration_methods.h"
#include "kernel.h"
#include "cuda_util.h"
#include "cuprintf.h"

#include "lib/log.h"
#include "lib/logger_file.h"
#include "lib/allocator_debug.h"
#include "lib/allocator_malloc.h"
#include "lib/error.h"
#include "lib/time.h"
#include "lib/image.h"
#include "lib/format_netbpm.h"
#include "lib/math.h"

#include <GLFW/glfw3.h>

#include <stddef.h>

const i32 SCR_WIDTH = 1000;
const i32 SCR_HEIGHT = 1000;

const f64 FPS_DISPLAY_FREQ = 50000;
const f64 RENDER_FREQ = 30;
const f64 POLL_FREQ = 30;

const f64 FREE_RUN_SYM_FPS = 200;

#define CONFIG_FILE "config.lpf"

typedef struct Simulation_State {
    Allen_Cahn_Config config;
    Allen_Cahn_Config last_config;
    i64 epoch_time;
    Platform_Calendar_Time calendar_time;
    Allocator* alloc;
    
    f32* initial_phi_map;
    f32* initial_T_map;

    f32* phi_map;
    f32* T_map;
    f32* next_phi_map;
    f32* next_T_map;

    f32* display_map1;
    f32* display_map2;
    f32* display_map3;
    
    Compute_Texture output_phi_map;
    Compute_Texture output_T_map;
} Simulation_State;

typedef struct App_State {
    GLFWwindow* window;

    bool is_in_step_mode;
    bool render_phi;
    f64 remaining_steps;
    f64 step_by;

    i64 queued_reloads;
    i64 queued_reloads_last;

    Simulation_State simulation_state;
} App_State;

bool simulation_state_is_hard_reload(const Allen_Cahn_Config* config, const Allen_Cahn_Config* last_config)
{
    if(config->params.mesh_size_x != last_config->params.mesh_size_x || config->params.mesh_size_y != last_config->params.mesh_size_y)
        return true;

    //@HACK: proper comparison of fields is replaced by comparison of the flat memory because I am lazy.
    String_Builder null_sttring_builder = {0};
    Allen_Cahn_Initial_Conditions ini = config->initial_conditions;
    Allen_Cahn_Initial_Conditions ini_last = last_config->initial_conditions;
    ini.start_snapshot = null_sttring_builder;
    ini_last.start_snapshot = null_sttring_builder;

    if(memcmp(&ini, &ini_last, sizeof ini_last) != 0)
        return true;

    if(builder_is_equal(config->initial_conditions.start_snapshot, last_config->initial_conditions.start_snapshot) == false)
        return true;

    return false;
}

void simulation_state_deinit(Simulation_State* state)
{
    (void) state;
}

void allen_cahn_set_initial_conditions(f32* initial_phi_map, f32* initial_T_map, Allen_Cahn_Config config);

void simulation_state_reload(Simulation_State* state, Allen_Cahn_Config* config)
{
    if(state->alloc == NULL)
        state->alloc = allocator_get_default();

    if(config == NULL || simulation_state_is_hard_reload(&state->last_config, config))
    {
        isize old_pixel_count = state->last_config.params.mesh_size_x * state->last_config.params.mesh_size_y;
        allocator_deallocate(state->alloc, state->initial_phi_map, old_pixel_count*sizeof(f32), DEF_ALIGN, SOURCE_INFO());
        allocator_deallocate(state->alloc, state->initial_T_map, old_pixel_count*sizeof(f32), DEF_ALIGN, SOURCE_INFO());

        cudaFree(state->phi_map);
        cudaFree(state->T_map);
        cudaFree(state->next_phi_map);
        cudaFree(state->next_T_map);

        compute_texture_deinit(&state->output_phi_map);
        compute_texture_deinit(&state->output_T_map);

        if(config != NULL)
        {
            isize pixel_count_x = config->params.mesh_size_x;
            isize pixel_count_y = config->params.mesh_size_y;
            isize pixel_count = pixel_count_x * pixel_count_y;

            state->initial_phi_map = (f32*) allocator_allocate_cleared(state->alloc, pixel_count * sizeof(f32), DEF_ALIGN, SOURCE_INFO());
            state->initial_T_map = (f32*) allocator_allocate_cleared(state->alloc, pixel_count * sizeof(f32), DEF_ALIGN, SOURCE_INFO());
            allen_cahn_set_initial_conditions(state->initial_phi_map, state->initial_T_map, *config);

            CUDA_TEST(cudaMalloc((void**)&state->phi_map,          pixel_count * sizeof(f32)));
            CUDA_TEST(cudaMalloc((void**)&state->T_map,            pixel_count * sizeof(f32)));
            CUDA_TEST(cudaMalloc((void**)&state->next_phi_map,     pixel_count * sizeof(f32)));
            CUDA_TEST(cudaMalloc((void**)&state->next_T_map,       pixel_count * sizeof(f32)));

            CUDA_TEST(cudaMemcpy(state->phi_map, state->initial_phi_map, pixel_count * sizeof(f32), cudaMemcpyHostToDevice));
            CUDA_TEST(cudaMemcpy(state->T_map, state->initial_T_map, pixel_count * sizeof(f32), cudaMemcpyHostToDevice));
    
            state->output_phi_map  = compute_texture_make(pixel_count_x, pixel_count_y, PIXEL_FORMAT_F32, 1);
            state->output_T_map    = compute_texture_make(pixel_count_x, pixel_count_y, PIXEL_FORMAT_F32, 1);
        }
    }

    //@TODO: not leak memeory here
    if(state->epoch_time == 0)
        state->epoch_time = platform_epoch_time();

    state->calendar_time = platform_local_calendar_time_from_epoch_time(state->epoch_time);
    if(config)
    {
        state->config = *config;
        state->last_config = *config;
        platform_directory_create(string_from_builder(config->snapshots.folder), NULL);
    }
}

void app_state_init(App_State* state, GLFWwindow* window)
{
    state->window = window;
    state->is_in_step_mode = true;
    state->remaining_steps = 0;
    state->step_by = 1;
    state->render_phi = true;
}


Error cuda_error(cudaError_t error);
String cuda_translate_error(u32 code, void* context);
void render_cuda_memory(App_State* app, Compute_Texture texture, const f32* cuda_memory, f32 min, f32 max);
void allen_cahn_custom_config(Allen_Cahn_Config* out_config);

bool app_poll_reloads(App_State* state)
{
    if(state->queued_reloads > platform_atomic_excahnge64(&state->queued_reloads_last, state->queued_reloads))
        return true;
    else
        return false;
}

bool queue_file_reload(void* context)
{
    App_State* state = (App_State*) context;
    platform_atomic_add64(&state->queued_reloads, 1);
    return true;
}

void allen_cahn_set_initial_conditions(f32* initial_phi_map, f32* initial_T_map, Allen_Cahn_Config config)
{
    Allen_Cahn_Params params = config.params;
    Allen_Cahn_Initial_Conditions initial = config.initial_conditions;
    for(isize y = 0; y < params.mesh_size_y; y++)
    {
        for(isize x = 0; x < params.mesh_size_x; x++)
        {
            Vec2 pos = vec2((f32) x / params.mesh_size_x * params.sym_size, (f32) y / params.mesh_size_y * params.sym_size); 
            isize i = x + y*params.mesh_size_x;

		    if(((initial.square_from.x <= pos.x && pos.x < initial.square_to.x) && 
			    (initial.square_from.y <= pos.y && pos.y < initial.square_to.y))
			    || vec2_len(vec2_sub(initial.circle_center, pos)) < initial.circle_radius)
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

void run_func_allen_cahn_cuda(void* context)
{
    LOG_INFO("App", "current working dir: '%s'", platform_directory_get_current_working());

    GLFWwindow* window = (GLFWwindow*) context;
    App_State* app = (App_State*) glfwGetWindowUserPointer(window); (void) app;
    Simulation_State* simualtion = &app->simulation_state;
    
    queue_file_reload(app);
    cudaPrintfInit();

    Platform_File_Watch file_watch = {0};
    Error watch_error = error_from_platform(platform_file_watch(&file_watch, STRING("."), PLATFORM_FILE_WATCH_CHANGE, queue_file_reload, app));
    ASSERT_MSG(error_is_ok(watch_error), "file watch failed %s", error_code(watch_error));

    int device_id = 0;
    CUDA_TEST(cudaSetDevice(device_id));
    
    int device_processor_count = 0;
    cudaDeviceGetAttribute(&device_processor_count, cudaDevAttrMultiProcessorCount, device_id);

    i64 frame_counter = 0;
    f64 frame_time_sum = 0;
    
    f64 fps_display_last_time_sum = 0;
    f64 fps_display_last_time = 0;
    
    f64 poll_last_time = 0;
    f64 render_last_time = 0;
    f64 simulated_last_time = 0;

    f64 simulation_time_sum = 0;

	while (!glfwWindowShouldClose(window))
    {
        f64 now = clock_s();

        if(app_poll_reloads(app))
        {
            Allen_Cahn_Config config = {0};
            TEST(allen_cahn_read_file_config(&config, CONFIG_FILE));
            
            if(config.snapshots.folder.size > 0)
                platform_directory_create(string_from_builder(config.snapshots.folder), NULL);

            simulation_state_reload(simualtion, &config);
        }

        if(now - render_last_time > 1.0/RENDER_FREQ)
        {
            PERF_COUNTER_START(render);
            render_last_time = now;
            if(app->render_phi)
                render_cuda_memory(app, simualtion->output_phi_map, simualtion->phi_map, 0, 1);
            else
                render_cuda_memory(app, simualtion->output_T_map, simualtion->T_map, 0, 1.5);
                
            glfwSwapBuffers(app->window);
            PERF_COUNTER_END(render);
        }

        if(now - fps_display_last_time > 1.0/FPS_DISPLAY_FREQ)
        {
            f64 time_sum_delta = frame_time_sum - fps_display_last_time_sum;
            if(time_sum_delta != 0)
            {
                glfwSetWindowTitle(window, format_ephemeral("iter %lli", (lli) frame_counter).data);
            }

            fps_display_last_time = now;
            fps_display_last_time_sum = frame_time_sum;
        }


        bool step_sym = false;
        if(app->is_in_step_mode)
            step_sym = app->remaining_steps > 0.5;
        else
            step_sym = now - simulated_last_time > 1.0/app->step_by/FREE_RUN_SYM_FPS;

        if(step_sym)
        {
            simulated_last_time = now;
            app->remaining_steps -= 1;

            f64 frame_start_time = clock_s();

            PERF_COUNTER_START(simulation_step);
                CUDA_TEST(kernel_step(simualtion->next_phi_map, simualtion->next_T_map, simualtion->phi_map, simualtion->T_map, simualtion->config.params, device_processor_count, frame_counter));
            PERF_COUNTER_END(simulation_step);

            PERF_COUNTER_START(cudaPrintfDisplay_c);
                CUDA_TEST(cudaPrintfDisplay());
            PERF_COUNTER_END(cudaPrintfDisplay_c);
            
            f64 end_start_time = clock_s();
            f64 delta = end_start_time - frame_start_time;

            frame_time_sum += delta;
            frame_counter += 1;
            simulation_time_sum += simualtion->config.params.dt;

            SWAP(&simualtion->phi_map, &simualtion->next_phi_map, f32*);
            SWAP(&simualtion->T_map, &simualtion->next_T_map, f32*);
        }
        
        if(now - poll_last_time > 1.0/POLL_FREQ)
        {
            PERF_COUNTER_START(poll_evennts);
		        glfwPollEvents();
            PERF_COUNTER_END(poll_evennts);
        }
    }

    platform_file_unwatch(&file_watch);
}

void* glfw_malloc_func(size_t size, void* user);
void* glfw_realloc_func(void* block, size_t size, void* user);
void glfw_free_func(void* block, void* user);
void glfw_error_func(int code, const char* description);
void glfw_resize_func(GLFWwindow* window, int width, int heigth);
void glfw_key_func(GLFWwindow* window, int key, int scancode, int action, int mods);

void run_func(void* context);
void error_func(void* context, Platform_Sandbox_Error error_code);

void platform_test_func()
{
    Platform_Directory_Entry* entries = NULL;
    isize entries_count = 0;
    platform_directory_list_contents_alloc(STRING("."), &entries, &entries_count, 3);

    platform_file_info(STRING("temp.h"), NULL);
    platform_file_info(STRING("main.h"), NULL);
    platform_file_info(STRING("config.h"), NULL);
    platform_file_info(STRING("temp.h"), NULL);

    LOG_INFO("platform", "executable path:     '%s'", platform_get_executable_path());
    LOG_INFO("platform", "current working dir: '%s'", platform_directory_get_current_working());

    String_Builder dir_padding = {0};

    //String_Builder complete_list = {0};

    for(isize i = 0; i < entries_count; i++)
    {
        Platform_Directory_Entry entry = entries[i];
        array_clear(&dir_padding);
        for(isize j = 0; j < entry.directory_depth; j++)
            builder_append(&dir_padding, STRING("  "));

        //builder_append(&complete_list)
        LOG_INFO("dirs", "%lli %s%s", i, cstring_from_builder(dir_padding), entry.path);
    }

    //LOG_INFO("dirs", complete_list.data);
    array_deinit(&dir_padding);
    platform_directory_list_contents_free(entries);
}

#ifdef USE_OPENGL
#include <cuda_gl_interop.h>
#include "render.h"
#endif


int main()
{
    platform_init();
    Malloc_Allocator static_allocator = {0};
    malloc_allocator_init(&static_allocator);
    allocator_set_static(&static_allocator.allocator);
    
    Malloc_Allocator malloc_allocator = {0};
    malloc_allocator_init_use(&malloc_allocator, 0);
    
    Debug_Allocator debug_alloc = {0};
    debug_allocator_init_use(&debug_alloc, &malloc_allocator.allocator, DEBUG_ALLOCATOR_DEINIT_LEAK_CHECK | DEBUG_ALLOCATOR_CAPTURE_CALLSTACK);

    error_system_init(&static_allocator.allocator);
    file_logger_init_use(&global_logger, &malloc_allocator.allocator, &malloc_allocator.allocator);

    GLFWallocator allocator = {0};
    allocator.allocate = glfw_malloc_func;
    allocator.reallocate = glfw_realloc_func;
    allocator.deallocate = glfw_free_func;
    allocator.user = &malloc_allocator;
 
    glfwInitAllocator(&allocator);
    glfwSetErrorCallback(glfw_error_func);
    TEST_MSG(glfwInit(), "Failed to init glfw");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
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
 
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Render", NULL, NULL);
    TEST_MSG(window != NULL, "Failed to make glfw window");

    App_State app = {0};
    app_state_init(&app, window);
    glfwSetWindowUserPointer(window, &app);
    glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, glfw_resize_func);
    glfwSetKeyCallback(window, glfw_key_func);
    glfwSwapInterval(0);

    #ifdef USE_OPENGL
        int version = gladLoadGL((GLADloadfunc) glfwGetProcAddress);
        TEST_MSG(version != 0, "Failed to load opengl with glad");

        gl_debug_output_enable();
    #endif

    platform_exception_sandbox(
        run_func_allen_cahn_cuda, window, 
        error_func, window);

    glfwDestroyWindow(window);
    glfwTerminate();

    debug_allocator_deinit(&debug_alloc);
    
    file_logger_deinit(&global_logger);
    error_system_deinit();

    ASSERT(malloc_allocator.bytes_allocated == 0);
    malloc_allocator_deinit(&malloc_allocator);
    platform_deinit();

    return 0;    
}

void* glfw_malloc_func(size_t size, void* user)
{
    return malloc_allocator_malloc((Malloc_Allocator*) user, size);
}

void* glfw_realloc_func(void* block, size_t size, void* user)
{
    return malloc_allocator_realloc((Malloc_Allocator*) user, block, size);
}

void glfw_free_func(void* block, void* user)
{
    malloc_allocator_free((Malloc_Allocator*) user, block);
}

void glfw_error_func(int code, const char* description)
{
    LOG_ERROR("APP", "GLWF error %d with message: %s", code, description);
}
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
        if(key == GLFW_KEY_F1)
        {
            app->render_phi = !app->render_phi;
            LOG_INFO("APP", "Rendering %s", app->render_phi ? "phi" : "T");
        }

        if(key == GLFW_KEY_C)
        {
            for(Global_Perf_Counter* counter = profile_get_counters(); counter != NULL; counter = counter->next)
            {
                Perf_Counter_Stats stats = perf_counter_get_stats(counter->counter, 1);
		        LOG_INFO("APP", "total: %15.8lf avg: %12.8lf runs: %-8lli σ/μ %13.6lf [%13.6lf %13.6lf] (ms) from %-4lli %s \"%s\"", 
			        stats.total_s*1000,
			        stats.average_s*1000,
                    (lli) stats.runs,
                    stats.normalized_standard_deviation_s,
			        stats.min_s*1000,
			        stats.max_s*1000,
			        (lli) counter->line,
			        counter->function,
			        counter->name
		        );
            }
        }
        
        f64 iters_before = app->step_by;
        if(key == GLFW_KEY_O)
            app->step_by = app->step_by*1.3 + 1;
        if(key == GLFW_KEY_P)
            app->step_by = MAX((app->step_by - 1)/1.3, 1.0);

        if(iters_before != app->step_by)
            LOG_INFO("APP", "Steps per iter %lf", app->step_by);
    }
}

void error_func(void* context, Platform_Sandbox_Error error)
{
    (void) context;
    const char* msg = platform_exception_to_string(error.exception);
    
    LOG_ERROR("APP", "%s exception occured", msg);
    LOG_TRACE("APP", "printing trace:");
    log_group_push();
    log_translated_callstack("APP", LOG_TYPE_TRACE, error.call_stack, error.call_stack_size);
    log_group_pop();
}

#ifdef USE_OPENGL
void render_cuda_memory(App_State* app, Compute_Texture texture, const f32* cuda_memory, f32 min, f32 max)
{
    struct cudaGraphicsResource *cuda_resource = 0; 
            
    //@TODO: the registering and unregistering has a cost!
    //@TODO: fastest would be to write directly into the texture!
    CUDA_TEST(cudaGraphicsGLRegisterImage(&cuda_resource, texture.id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_TEST(cudaGraphicsMapResources(1, &cuda_resource, 0));
    
    #if 1
        //for texture objects
        cudaArray_t mapped_array = {0};
        isize texture_size = texture.width * texture.heigth * texture.format.pixel_size;
        CUDA_TEST(cudaGraphicsSubResourceGetMappedArray(&mapped_array, cuda_resource, 0, 0));
        CUDA_TEST(cudaMemcpyToArray(mapped_array, 0, 0, cuda_memory, texture_size, cudaMemcpyDeviceToDevice));
    #else
        //For gl buffers
        size_t num_bytes = 0;
        unsigned int *mapped_map = NULL;
        CUDA_TEST(cudaGraphicsResourceGetMappedPointer((void **)&mapped_map, &num_bytes, cuda_resource));
        ASSERT(num_bytes <= texture_size);
        cudaMemcpy(mapped_map, cuda_memory, texture_size, cudaMemcpyDeviceToDevice);
    #endif

    CUDA_TEST(cudaGraphicsUnmapResources(1, &cuda_resource, 0));
    CUDA_TEST(cudaGraphicsUnregisterResource(cuda_resource)); 
    (void) app;
    render_sci_texture(texture, min, max);
}
#endif

void allen_cahn_custom_config(Allen_Cahn_Config* out_config)
{
    const i32 _SIZE_X = 1024;
    const i32 _SIZE_Y = _SIZE_X;
    const f32 _dt = 1.0f/200;
    const f32 _alpha = 0.5;
    const f32 _L = 2;
    const f32 _xi = 0.00411f;
    const f32 _a = 2;
    const f32 _b = 1;
    const f32 _beta = 8;
    const f32 _Tm = 1;
    const f32 _Tini = 0;
    const f32 _L0 = 4;

    Allen_Cahn_Scale scale = {0};
    scale.L0 = _L0 / (f32) _SIZE_X;
    scale.Tini = _Tini;
    scale.Tm = _Tm;
    scale.c = 1;
    scale.rho = 1;
    scale.lambda = 1;
    
    Allen_Cahn_Params params = {0};
    params.sym_size = _L0;
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
    initial_conditions.circle_center = vec2(_L0 / 4, _L0 / 4);
    initial_conditions.circle_radius = _L0 / 8;
    initial_conditions.square_from = vec2(_L0/2 - 0.3f, _L0/2 - 0.3f);
    initial_conditions.square_to = vec2(_L0/2 + 0.3f, _L0/2 + 0.3f);

    Allen_Cahn_Snapshots snapshots = {0};
    snapshots.folder = builder_from_cstring("snapshots", NULL);
    snapshots.prefix = builder_from_cstring("v1", NULL);
    snapshots.every = 0.1f;
    snapshots.sym_time = -1;
    
    out_config->config_name = builder_from_cstring("from_code_config", NULL);
    out_config->initial_conditions = initial_conditions;
    out_config->params = params;
    out_config->snapshots = snapshots;
}

#if PLATFORM_OS == PLATFORM_OS_WINDOWS
    #include "lib/platform_windows.c"
#elif PLATFORM_OS == PLATFORM_OS_UNIX
    #include "lib/platform_linux.c"
#else
    #error Provide support for this operating system or define PLATFORM_OS to one of the values in platform.h
#endif