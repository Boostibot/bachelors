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

#define JOT_ALL_IMPL
#include "lib/platform.h"

#include "config.h"
#include "integration_methods.h"
#include "kernel.h"
#include "cuda_util.h"
#include "cuprintf.h"

#if 1
#include "lib/log.h"
#include "lib/logger_file.h"
#include "lib/allocator_debug.h"
#include "lib/allocator_malloc.h"
#include "lib/error.h"
#include "lib/time.h"
#include "lib/image.h"
#include "lib/format_netbpm.h"
#include "lib/math.h"

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
} Simulation_State;

typedef struct App_State {
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

void app_state_init(App_State* state)
{
    state->is_in_step_mode = true;
    state->remaining_steps = 0;
    state->step_by = 1;
    state->render_phi = true;
}


Error cuda_error(cudaError_t error);
String cuda_translate_error(u32 code, void* context);
void allen_cahn_custom_config(Allen_Cahn_Config* out_config);

bool app_poll_reloads(App_State* state)
{
    if(state->queued_reloads > platform_atomic_excahnge64(&state->queued_reloads_last, state->queued_reloads))
        return true;
    else
        return false;
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

    App_State* app = (App_State*) context;
    Simulation_State* simualtion = &app->simulation_state;
    

    cudaPrintfInit(0);

    int device_id = 0;
    CUDA_TEST(cudaSetDevice(device_id));
    LOG_INFO("App", "Device set");
    
    int device_processor_count = 0;
    CUDA_TEST(cudaDeviceGetAttribute(&device_processor_count, cudaDevAttrMultiProcessorCount, device_id));
    LOG_INFO("App", "device_processor_count %d", device_processor_count);

    Allen_Cahn_Config config = {0};
    TEST(allen_cahn_read_file_config(&config, CONFIG_FILE));
    simulation_state_reload(simualtion, &config);
    LOG_INFO("App", "Config '%s' read", cstring_from_builder(config.config_name));

    i64 frame_counter = 0;
    f64 frame_time_sum = 0;
    
    f64 fps_display_last_time_sum = 0;
    f64 fps_display_last_time = 0;
    
    f64 poll_last_time = 0;
    f64 render_last_time = 0;
    f64 simulated_last_time = 0;

    f64 simulation_time_sum = 0;

    while(true)
    {
        f64 now = clock_s();
        simulated_last_time = now;
        app->remaining_steps -= 1;

        f64 frame_start_time = clock_s();

        PERF_COUNTER_START(simulation_step);
            CUDA_TEST(kernel_step(simualtion->next_phi_map, simualtion->next_T_map, simualtion->phi_map, simualtion->T_map, simualtion->config.params, device_processor_count, frame_counter));
        PERF_COUNTER_END(simulation_step);

        PERF_COUNTER_START(cudaPrintfDisplay_c);
            CUDA_TEST(cudaPrintfDisplay(NULL, 0));
        PERF_COUNTER_END(cudaPrintfDisplay_c);
        
        f64 end_start_time = clock_s();
        f64 delta = end_start_time - frame_start_time;

        frame_time_sum += delta;
        frame_counter += 1;
        simulation_time_sum += simualtion->config.params.dt;

        SWAP(&simualtion->phi_map, &simualtion->next_phi_map, f32*);
        SWAP(&simualtion->T_map, &simualtion->next_T_map, f32*);

        if(frame_counter % 100 == 0)
            LOG_DEBUG("APP", "iter: %lli", frame_counter);
    }
}

void run_func(void* context);
void error_func(void* context, Platform_Sandbox_Error error_code);

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
    LOG_INFO("App", "Systems initialized");
    LOG_INFO("App", "Current working dir: '%s'", platform_directory_get_current_working());

    App_State app = {0};
    app_state_init(&app);

    platform_exception_sandbox(
        run_func_allen_cahn_cuda, &app, 
        error_func, &app);

    debug_allocator_deinit(&debug_alloc);
    
    file_logger_deinit(&global_logger);
    error_system_deinit();

    ASSERT(malloc_allocator.bytes_allocated == 0);
    malloc_allocator_deinit(&malloc_allocator);
    platform_deinit();

    return 0;    
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
#endif