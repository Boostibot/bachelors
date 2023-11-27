
#define _CRT_SECURE_NO_WARNINGS
#define JOT_ALL_IMPL
#define JMAPI __host__ __device__ static inline

#include "config.h"
#include "integration_methods.h"
#include "render.h"
#include "cuprintf.cuh"
#include "cuprintf.cu"

#include "lib/platform.h"
#include "lib/log.h"
#include "lib/logger_file.h"
#include "lib/allocator_debug.h"
#include "lib/allocator_malloc.h"
#include "lib/error.h"
#include "lib/time.h"
#include "lib/image.h"
#include "lib/format_netbpm.h"
#include "lib/math.h"

#include "glfw/glfw3.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const i32 SCR_WIDTH = 1000;
const i32 SCR_HEIGHT = 1000;

const f64 FPS_DISPLAY_FREQ = 50000;
const f64 RENDER_FREQ = 30;

const f64 FREE_RUN_SYM_FPS = 200;

#define CONFIG_FILE "config.lpf"

typedef struct App_State {
    GLFWwindow* window;

    bool is_in_step_mode;
    bool render_phi;
    f64 remaining_steps;
    f64 step_by;
} App_State;

void app_state_init(App_State* state, GLFWwindow* window)
{
    state->window = window;
    state->is_in_step_mode = true;
    state->remaining_steps = 0;
    state->step_by = 1;
    state->render_phi = true;
}

JMAPI f32 map_at(const f32* map, int x, int y, Allen_Cahn_Params params)
{
    int x_mod = x % params.mesh_size_x;
    int y_mod = y % params.mesh_size_y;

    return map[x_mod + y_mod*params.mesh_size_x];
}

JMAPI f32 allen_cahn_reaction_term_0(f32 phi)
{
	return phi*(1 - phi)*(phi - 1.0f/2);
}

JMAPI f32 allen_cahn_reaction_term_1(f32 phi, f32 T, f32 xi, Allen_Cahn_Params params)
{
    f32 mK = 1;
	return (params.a*allen_cahn_reaction_term_0(phi) - params.b*params.beta*xi*(T - params.Tm))*mK;
}

JMAPI f32 allen_cahn_reaction_term_2(f32 phi, f32 T, f32 xi, Vec2 grad_phi, Allen_Cahn_Params params)
{
    f32 mK = 1;
	f32 grad_val = vec2_len(grad_phi);
	return (params.a*allen_cahn_reaction_term_0(phi) - params.b*params.beta*xi*xi*grad_val*(T - params.Tm))*mK;
}

__global__ void allen_cahn_simulate(f32* phi_map_next, f32* T_map_next, const f32* phi_map, const f32* T_map, Allen_Cahn_Params params, isize iter)
{
    f32 tau = 1;
    f32 mK = 1;
    f32 grad_interp = 0.5f;

    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < params.mesh_size_x; x += blockDim.x * gridDim.x) 
    {
        for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < params.mesh_size_x; y += blockDim.y * gridDim.y) 
        {
	        f32 T = map_at(T_map, x, y, params);
	        f32 phi = map_at(phi_map, x, y, params);

	        f32 phi_py = map_at(phi_map, x, y + 1, params);
	        f32 phi_my = map_at(phi_map, x, y - 1, params);
	        f32 phi_px = map_at(phi_map, x + 1, y, params);
	        f32 phi_mx = map_at(phi_map, x - 1, y, params);

	        f32 T_py = map_at(T_map, x, y + 1, params);
	        f32 T_my = map_at(T_map, x, y - 1, params);
	        f32 T_px = map_at(T_map, x + 1, y, params);
	        f32 T_mx = map_at(T_map, x - 1, y, params);
            
            if(y == 0)
                if(T > 0 || phi > 0)
                {
                    //cuPrintf("%lli (%d, %d) {T: %16.8f phi: %16.8f} \n", iter, x, y, T, phi);
                }

	        f32 sum_phi_neigbours = 0
		        + tau*(phi_py - phi)
		        + tau*(phi_my - phi)
		        + tau*(phi_px - phi)
		        + tau*(phi_mx - phi);
		
	        f32 sum_T_neigbours = 0
		        + tau*(T_py - T)
		        + tau*(T_my - T)
		        + tau*(T_px - T)
		        + tau*(T_mx - T);

	        Vec2 grad_phi = {
		        (phi_px - phi_mx) * grad_interp,
		        (phi_py - phi_my) * grad_interp
	        };
        
	        f32 reaction_term = allen_cahn_reaction_term_2(phi, T, params.xi, grad_phi, params);
	        //f32 reaction_term = 0;
	        f32 phi_dt = (sum_phi_neigbours/mK + reaction_term/(params.xi*params.xi)) / params.alpha;
	        f32 T_dt = sum_T_neigbours / mK + params.L * phi_dt;

	        f32 phi_next = phi_dt * params.dt + phi;
	        f32 T_next = T_dt * params.dt + T;
		
            phi_map_next[x + y*params.mesh_size_x] = phi_next;
            T_map_next[x + y*params.mesh_size_x] = T_next;
        }
    }
}

void _test_cuda(cudaError_t error, const char* expression, Source_Info info)
{
    if(error != cudaSuccess)
    {
        assertion_report(expression, info, "cuda failed with error %s", cudaGetErrorString(error));
        platform_trap();
        platform_abort();
    }
}

#define CUDA_TEST(status) _test_cuda((status), #status, SOURCE_INFO())
#define CUDA_ERR_AND(err) (err) != cudaSuccess ? (err) :

Error cuda_error(cudaError_t error);
String cuda_translate_error(u32 code, void* context);
void render_cuda_memory(App_State* app, Compute_Texture texture, const f32* cuda_memory, f32 min, f32 max);
void allen_cahn_custom_config(Allen_Cahn_Config* out_config);

bool file_change_func(void* context)
{
    Allen_Cahn_Config* config = (Allen_Cahn_Config*) context;
    Allocator_Set prev_set = allocator_set_both(config->allocator, config->allocator);
    TEST(allen_cahn_read_file_config(config, CONFIG_FILE));
    allocator_set(prev_set);

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
    compare_rk4();
    cudaPrintfInit();
    
    Allen_Cahn_Config config = {0};
    config.allocator = allocator_get_default();
    allen_cahn_custom_config(&config);
    file_change_func(&config);
    //TEST(allen_cahn_read_file_config(&config, CONFIG_FILE));
    
    Platform_File_Watch file_watch = {0};
    Error watch_error = error_from_platform(platform_file_watch(&file_watch, ".", PLATFORM_FILE_WATCH_CHANGE, file_change_func, &config));
    ASSERT_MSG(error_is_ok(watch_error), "file watch failed %s", error_code(watch_error).data);

    int device_id = 0;
    CUDA_TEST(cudaSetDevice(device_id));
    
    int device_processor_count = 0;
    cudaDeviceGetAttribute(&device_processor_count, cudaDevAttrMultiProcessorCount, device_id);
    
    isize pixel_count = config.params.mesh_size_x * config.params.mesh_size_y;

    f32* initial_phi_map = (f32*) allocator_allocate_cleared(allocator_get_default(), pixel_count * sizeof(f32), DEF_ALIGN, SOURCE_INFO());
    f32* initial_T_map = (f32*) allocator_allocate_cleared(allocator_get_default(), pixel_count * sizeof(f32), DEF_ALIGN, SOURCE_INFO());

    f32* phi_map = NULL;
    f32* T_map = NULL;
    f32* next_phi_map = NULL;
    f32* next_T_map = NULL;

    f32* display_map1 = NULL;
    f32* display_map2 = NULL;
    f32* display_map3 = NULL;

    CUDA_TEST(cudaMalloc((void**)&phi_map,          pixel_count * sizeof(f32)));
    CUDA_TEST(cudaMalloc((void**)&T_map,            pixel_count * sizeof(f32)));
    CUDA_TEST(cudaMalloc((void**)&next_phi_map,     pixel_count * sizeof(f32)));
    CUDA_TEST(cudaMalloc((void**)&next_T_map,       pixel_count * sizeof(f32)));
    //CUDA_TEST(cudaMalloc((void**)&display_map1,     pixel_count * sizeof(f32)));
    //CUDA_TEST(cudaMalloc((void**)&display_map2,     pixel_count * sizeof(f32)));
    //CUDA_TEST(cudaMalloc((void**)&display_map3,     pixel_count * sizeof(f32)));

    allen_cahn_set_initial_conditions(initial_phi_map, initial_T_map, config);

    CUDA_TEST(cudaMemcpy(phi_map, initial_phi_map, pixel_count * sizeof(f32), cudaMemcpyHostToDevice));
    CUDA_TEST(cudaMemcpy(T_map, initial_T_map, pixel_count * sizeof(f32), cudaMemcpyHostToDevice));

    Compute_Texture output_phi_map  = compute_texture_make(config.params.mesh_size_x, config.params.mesh_size_y, PIXEL_FORMAT_F32, 1);
    Compute_Texture output_T_map    = compute_texture_make(config.params.mesh_size_x, config.params.mesh_size_y, PIXEL_FORMAT_F32, 1);

    Platform_Calendar_Time calendar_time = platform_epoch_time_to_calendar_time(platform_local_epoch_time());
    String_Builder serialized_image = {0};

    platform_directory_create(config.snapshots.folder.data);

    i64 save_counter = 0;
    i64 frame_counter = 0;
    f64 frame_time_sum = 0;
    
    f64 fps_display_last_time_sum = 0;
    f64 fps_display_last_time = 0;
    i64 fps_display_last_frames = 0;
    
    f64 render_last_time = 0;
    f64 simulated_last_time = 0;

    f64 simulation_time_sum = 0;

    GLFWwindow* window = (GLFWwindow*) context;
    App_State* app = (App_State*) glfwGetWindowUserPointer(window); (void) app;
	while (!glfwWindowShouldClose(window))
    {
        f64 now = clock_s();
        //if(0)
        if(now - render_last_time > 1.0/RENDER_FREQ)
        {
            render_last_time = now;
            if(app->render_phi)
                render_cuda_memory(app, output_phi_map, phi_map, 0, 1);
            else
                render_cuda_memory(app, output_T_map, T_map, 0, 1.5);
                
            glfwSwapBuffers(app->window);
        }

        if(now - fps_display_last_time > 1.0/FPS_DISPLAY_FREQ)
        {
            f64 time_sum_delta = frame_time_sum - fps_display_last_time_sum;
            f64 counter_delta = (f64) (frame_counter - fps_display_last_frames);
            f64 avg_fps = 0;
            if(time_sum_delta != 0)
            {
                avg_fps = counter_delta / time_sum_delta;
                glfwSetWindowTitle(window, format_ephemeral("iter %lli", (lli) frame_counter).data);
            }

            fps_display_last_time = now;
            fps_display_last_frames = frame_counter;
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

            dim3 bs(64, 1);
            dim3 grid(device_processor_count, 1);
            allen_cahn_simulate<<<grid, bs>>>(next_phi_map, next_T_map, phi_map, T_map, config.params, frame_counter);
            CUDA_TEST(cudaGetLastError());
            CUDA_TEST(cudaDeviceSynchronize());
            cudaPrintfDisplay();
            
            f64 end_start_time = clock_s();

            f64 delta = end_start_time - frame_start_time;
            
            #if 0
            if(SAVE_EVERY > 0 && frame_counter % SAVE_EVERY == 0)
            {
                PERF_COUNTER_START(image_saving);
                Image_Builder pixels = {0};
                String file_name = {0};
                image_builder_init(&pixels, NULL, 1, PIXEL_FORMAT_U8);

                {
                    compute_texture_get_pixels_converted(&pixels, next_phi_map);
                    netbpm_format_pgm_write_into(&serialized_image, image_from_builder(pixels));
                
                    file_name = format_ephemeral("%s/%s_%lld-%lld-%lld_%lld-%lld-%lld_iter_%lld_phi.pgm", SAVE_FOLDER, SAVE_PREFIX, 
                        (lli) calendar_time.year, (lli) calendar_time.month, (lli) calendar_time.day, 
                        (lli) calendar_time.hour, (lli) calendar_time.minute, (lli) calendar_time.second, 
                        (lli) frame_counter);

                    file_write_entire(file_name, string_from_builder(serialized_image));
                }

                {
                    compute_texture_get_pixels_converted(&pixels, next_T_map);
                    netbpm_format_pgm_write_into(&serialized_image, image_from_builder(pixels));
                
                    file_name = format_ephemeral("%s/%s_%lld-%lld-%lld_%lld-%lld-%lld_iter_%lld_T.pgm", SAVE_FOLDER, SAVE_PREFIX, 
                        (lli) calendar_time.year, (lli) calendar_time.month, (lli) calendar_time.day, 
                        (lli) calendar_time.hour, (lli) calendar_time.minute, (lli) calendar_time.second, 
                        (lli) frame_counter);

                    file_write_entire(file_name, string_from_builder(serialized_image));
                }

                image_builder_deinit(&pixels);
                PERF_COUNTER_END(image_saving);
                
                save_counter ++;
                if(save_counter > PAUSE_AFTER_SAVES)
                {
                    save_counter = 0;
                    app->is_in_step_mode = true;
                    app->remaining_steps = 0;
                }
            }
            #endif

            frame_time_sum += delta;
            frame_counter += 1;
            simulation_time_sum += config.params.dt;

            SWAP(&phi_map, &next_phi_map, f32*);
            SWAP(&T_map, &next_T_map, f32*);

        }
        
		glfwPollEvents();
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
void error_func(void* context, Platform_Sandox_Error error_code);

int main()
{
    platform_init();
    Malloc_Allocator static_allocator = {0};
    malloc_allocator_init(&static_allocator);
    allocator_set_static(&static_allocator.allocator);
    
    Malloc_Allocator malloc_allocator = {0};
    malloc_allocator_init_use(&malloc_allocator, 0);
    
    error_system_init(&static_allocator.allocator);
    file_logger_init_use(&global_logger, &malloc_allocator.allocator, &malloc_allocator.allocator);

    Debug_Allocator debug_alloc = {0};
    debug_allocator_init_use(&debug_alloc, DEBUG_ALLOCATOR_DEINIT_LEAK_CHECK | DEBUG_ALLOCATOR_CAPTURE_CALLSTACK);

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

    int version = gladLoadGL((GLADloadfunc) glfwGetProcAddress);
    TEST_MSG(version != 0, "Failed to load opengl with glad");

    gl_debug_output_enable();

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

void error_func(void* context, Platform_Sandox_Error error_code)
{
    (void) context;
    const char* msg = platform_sandbox_error_to_string(error_code);
    
    LOG_ERROR("APP", "%s exception occured", msg);
    LOG_TRACE("APP", "printing trace:");
    log_group_push();
    log_callstack("APP", LOG_TYPE_ERROR, -1, 1);
    log_group_pop();
}

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

#include "lib/platform_windows.c"