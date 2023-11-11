
#define LIB_ALL_IMPL
#define GLAD_GL_IMPLEMENTATION
//#define DEBUG
//#define LIB_MEM_DEBUG

#include "lib/log.h"
#include "lib/logger_file.h"
#include "lib/allocator_debug.h"
#include "lib/allocator_malloc.h"
#include "lib/error.h"
#include "lib/time.h"

#include "gl.h"
#include "gl_debug_output.h"
#include "gl_shader_util.h"

#include "glfw/glfw3.h"

const i32 SCR_WIDTH = 1000;
const i32 SCR_HEIGHT = 1000;

const f64 FPS_DISPLAY_FREQ = 50000;
const f64 RENDER_FREQ = 30;

const u32 WORK_GROUP_SIZE_X = 32;
const u32 WORK_GROUP_SIZE_Y = 32;
const u32 WORK_GROUP_SIZE_Z = 1;

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
    state->remaining_steps = 1;
    state->step_by = 1;
    state->render_phi = true;
}

typedef struct Compute_Texture {
    GLuint id;
    GLuint format;
    GLuint channels;
    GLuint pixel_type;
} Compute_Texture;

void render_sci_texture(App_State* app, Compute_Texture texture, f32 min, f32 max);
Compute_Texture compute_texture_make(isize width, isize heigth, GLuint type, GLuint channels);
void compute_texture_bind(Compute_Texture texture, GLenum access, isize slot);
void compute_texture_deinit(Compute_Texture* texture);

typedef struct Allen_Cahn_Scale {
    f32 L0;
    f32 Tm;
    f32 Tini;
    f32 c;
    f32 rho;
    f32 lambda;
} Allen_Cahn_Scale;

f32 allen_cahn_scale_heat(f32 T, Allen_Cahn_Scale scale)
{
    return 1 + (T - scale.Tm)/(scale.Tm - scale.Tini);
}

f32 allen_cahn_scale_latent_heat(f32 L, Allen_Cahn_Scale scale)
{
    return L * scale.rho * scale.c/(scale.Tm - scale.Tini);
}

f32 allen_cahn_scale_pos(f32 x, Allen_Cahn_Scale scale)
{
    return x / scale.L0;
}

f32 allen_cahn_scale_xi(f32 xi, Allen_Cahn_Scale scale)
{
    (void) scale;
    //return xi;
    return xi / scale.L0;
}

f32 allen_cahn_scale_time(f32 t, Allen_Cahn_Scale scale)
{
    const f32 _t0 = (scale.rho*scale.c/scale.lambda)*scale.L0*scale.L0;
    return t / _t0;
}

f32 allen_cahn_scale_beta(f32 beta, Allen_Cahn_Scale scale)
{
    return beta * scale.L0 * (scale.Tm - scale.Tini);
}

f32 allen_cahn_scale_alpha(f32 alpha, Allen_Cahn_Scale scale)
{
    return alpha * scale.lambda / (scale.rho * scale.c);
}


// Keep in mind:
// We wouldnt be making simulation if we knew what was gonna happen 

void run_func_allen_cahn(void* context)
{
    const i32 _SIZE_X = 1024;
    const i32 _SIZE_Y = _SIZE_X;
    
    const f32 _INITIAL_PHI = 1;
    const f32 _INITIAL_T = 0;
    const f32 _AROUND_PHI = 0;
    const f32 _AROUND_T = 0;

    const i32 _INITIAL_SIZE_X = _SIZE_X/8;
    const i32 _INITIAL_SIZE_Y = _INITIAL_SIZE_X;
    const i32 _INITIAL_RADIUS = _INITIAL_SIZE_X;
    
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

    const f64 FREE_RUN_SYM_FPS = 100;

    Allen_Cahn_Scale scale = {0};
    scale.L0 = _L0 / (f32) _SIZE_X;
    scale.Tini = _Tini;
    scale.Tm = _Tm;
    scale.c = 1;
    scale.rho = 1;
    scale.lambda = 1;

    Compute_Texture phi_map         = compute_texture_make(_SIZE_X, _SIZE_Y, GL_FLOAT, 1);
    Compute_Texture T_map           = compute_texture_make(_SIZE_X, _SIZE_Y, GL_FLOAT, 1);
    Compute_Texture next_phi_map    = compute_texture_make(_SIZE_X, _SIZE_Y, GL_FLOAT, 1);
    Compute_Texture next_T_map      = compute_texture_make(_SIZE_X, _SIZE_Y, GL_FLOAT, 1);
    Compute_Texture output_phi_map  = compute_texture_make(_SIZE_X, _SIZE_Y, GL_FLOAT, 1);
    Compute_Texture output_T_map    = compute_texture_make(_SIZE_X, _SIZE_Y, GL_FLOAT, 1);

    GLFWwindow* window = context;
    App_State* app = (App_State*) glfwGetWindowUserPointer(window); (void) app;

    Render_Shader compute_shader = {0};

    Error error = compute_shader_init_from_disk(&compute_shader, STRING("shaders/allen_cahn.comp"), WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, WORK_GROUP_SIZE_Z);
    TEST_MSG(error_is_ok(error), "Error while loading shaders!");

    i64 frame_counter = 0;
    f64 frame_time_sum = 0;
    
    f64 fps_display_last_time_sum = 0;
    f64 fps_display_last_time = 0;
    i64 fps_display_last_frames = 0;
    
    f64 render_last_time = 0;
    f64 simulated_last_time = 0;

    f64 simulation_time_sum = 0;

	while (!glfwWindowShouldClose(window))
    {
        f64 now = clock_s();
        if(now - render_last_time > 1.0/RENDER_FREQ)
        {
            render_last_time = now;
            if(app->render_phi)
                render_sci_texture(app, output_phi_map, 0, 1);
            else
                render_sci_texture(app, output_T_map, 0, 2);
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

            compute_texture_bind(phi_map, GL_WRITE_ONLY, 0);
            compute_texture_bind(T_map, GL_WRITE_ONLY, 1);

            compute_texture_bind(next_phi_map, GL_READ_ONLY, 2);
            compute_texture_bind(next_T_map, GL_READ_ONLY, 3);
            
            //for debug
            compute_texture_bind(output_phi_map, GL_WRITE_ONLY, 4);
            compute_texture_bind(output_T_map, GL_WRITE_ONLY, 5);

            f64 frame_start_time = clock_s();
            
            PERF_COUNTER_START(uniform_lookup_c);
            render_shader_set_i32(&compute_shader, "_SIZE_X", _SIZE_X);
            render_shader_set_i32(&compute_shader, "_SIZE_Y", _SIZE_Y);
            
            render_shader_set_i32(&compute_shader, "_frame_i", frame_counter == 0 ? 0 : 1);
            render_shader_set_i32(&compute_shader, "_INITIAL_SIZE_X", _INITIAL_SIZE_X);
            render_shader_set_i32(&compute_shader, "_INITIAL_SIZE_Y", _INITIAL_SIZE_Y);
            render_shader_set_i32(&compute_shader, "_INITIAL_RADIUS", _INITIAL_RADIUS);
            render_shader_set_f32(&compute_shader, "_INITIAL_PHI", _INITIAL_PHI);
            render_shader_set_f32(&compute_shader, "_INITIAL_T", _INITIAL_T);
            render_shader_set_f32(&compute_shader, "_AROUND_PHI", _AROUND_PHI);
            render_shader_set_f32(&compute_shader, "_AROUND_T", _AROUND_T);

            render_shader_set_f32(&compute_shader, "_dt", _dt);
            render_shader_set_f32(&compute_shader, "_L", allen_cahn_scale_latent_heat(_L, scale));
            render_shader_set_f32(&compute_shader, "_xi", allen_cahn_scale_xi(_xi, scale));
            render_shader_set_f32(&compute_shader, "_a", _a);
            render_shader_set_f32(&compute_shader, "_b", _b);
            render_shader_set_f32(&compute_shader, "_alpha", allen_cahn_scale_alpha(_alpha, scale));
            render_shader_set_f32(&compute_shader, "_beta", allen_cahn_scale_latent_heat(_beta, scale));
            //render_shader_set_f32(&compute_shader, "_Tm", _Tm);
            PERF_COUNTER_END(uniform_lookup_c);
            
            PERF_COUNTER_START(compute_shader_run_c);
            compute_shader_dispatch(&compute_shader, _SIZE_X, _SIZE_Y, 1);

		    glMemoryBarrier(GL_ALL_BARRIER_BITS);
            PERF_COUNTER_END(compute_shader_run_c);

            f64 end_start_time = clock_s();

            f64 delta = end_start_time - frame_start_time;
            frame_time_sum += delta;
            frame_counter += 1;
            simulation_time_sum += _dt;

            SWAP(&phi_map, &next_phi_map, Compute_Texture);
            SWAP(&T_map, &next_T_map, Compute_Texture);
        }
        
		glfwPollEvents();
    }

    compute_texture_deinit(&phi_map);
    compute_texture_deinit(&T_map);
    compute_texture_deinit(&next_phi_map);
    compute_texture_deinit(&next_T_map);
    render_shader_deinit(&compute_shader);
}

void* glfw_malloc_func(size_t size, void* user);
void* glfw_realloc_func(void* block, size_t size, void* user);
void glfw_free_func(void* block, void* user);
void glfw_error_func(int code, const char* description);
void glfw_resize_func(GLFWwindow* window, int width, int height);
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
        run_func_allen_cahn, window, 
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
void glfw_resize_func(GLFWwindow* window, int width, int height)
{
    (void) window;
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
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

EXPORT void assertion_report(const char* expression, Source_Info info, const char* message, ...)
{
    LOG_FATAL("TEST", "TEST(%s) TEST/ASSERTION failed! " SOURCE_INFO_FMT, expression, SOURCE_INFO_PRINT(info));
    if(message != NULL && strlen(message) != 0)
    {
        LOG_FATAL("TEST", "with message:\n", message);
        va_list args;
        va_start(args, message);
        vlog_message("TEST", LOG_TYPE_FATAL, SOURCE_INFO(), message, args);
        va_end(args);
    }
        
    log_group_push();
    log_callstack("TEST", LOG_TYPE_FATAL, -1, 1);
    log_group_pop();
    log_flush_all();

    platform_abort();
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

const char* get_memory_unit(isize bytes, isize *unit_or_null)
{
    isize GB = (isize) 1000*1000*1000;
    isize MB = (isize) 1000*1000;
    isize KB = (isize) 1000;
    isize B = (isize) 1;

    const char* out = "";
    isize unit = 1;

    if(bytes > GB)
    {
        out = "GB";
        unit = GB;
    }
    else if(bytes > MB)
    {
        out = "MB";
        unit = MB;
    }
    else if(bytes > KB)
    {
        out = "KB";
        unit = KB;
    }
    else
    {
        out = "B";
        unit = B;
    }

    if(unit_or_null)
        *unit_or_null = unit;

    return out;
}

const char* format_memory_unit_ephemeral(isize bytes)
{
    isize unit = 1;
    static String_Builder formatted = {0};
    if(formatted.allocator == NULL)
        array_init(&formatted, allocator_get_static());

    const char* unit_text = get_memory_unit(bytes, &unit);
    f64 ratio = (f64) bytes / (f64) unit;
    format_into(&formatted, "%.1lf %s", ratio, unit_text);

    return formatted.data;
}

void log_allocator_stats(const char* log_module, Log_Type log_type, Allocator_Stats stats)
{
    String_Builder formatted = {0};
    array_init_backed(&formatted, NULL, 512);

    LOG(log_module, log_type, "bytes_allocated: %s", format_memory_unit_ephemeral(stats.bytes_allocated));
    LOG(log_module, log_type, "max_bytes_allocated: %s", format_memory_unit_ephemeral(stats.max_bytes_allocated));
    LOG(log_module, log_type, "allocation_count: %lli", (lli) stats.allocation_count);
    LOG(log_module, log_type, "deallocation_count: %lli", (lli) stats.deallocation_count);
    LOG(log_module, log_type, "reallocation_count: %lli", (lli) stats.reallocation_count);

    array_deinit(&formatted);
}
EXPORT void allocator_out_of_memory(
    Allocator* allocator, isize new_size, void* old_ptr, isize old_size, isize align, 
    Source_Info called_from, const char* format_string, ...)
{
    Allocator_Stats stats = {0};
    if(allocator != NULL && allocator->get_stats != NULL)
        stats = allocator_get_stats(allocator);
        
    if(stats.type_name == NULL)
        stats.type_name = "<no type name>";

    if(stats.name == NULL)
        stats.name = "<no name>";
    
    String_Builder user_message = {0};
    array_init_backed(&user_message, allocator_get_scratch(), 1024);
    
    va_list args;
    va_start(args, format_string);
    vformat_into(&user_message, format_string, args);
    va_end(args);

    LOG_FATAL("MEMORY", 
        "Allocator %s %s ran out of memory\n"
        "new_size:    %lli B\n"
        "old_ptr:     %p\n"
        "old_size:    %lli B\n"
        "align:       %lli B\n"
        "called from: " SOURCE_INFO_FMT "\n"
        "user message:\n%s",
        stats.type_name, stats.name, 
        (lli) new_size, 
        old_ptr,
        (lli) old_size,
        (lli) align,
        SOURCE_INFO_PRINT(called_from),
        cstring_from_builder(user_message)
    );
    
    LOG_FATAL("MEMORY", "Allocator_Stats:");
    log_group_push();
        log_allocator_stats("MEMORY", LOG_TYPE_FATAL, stats);
    log_group_pop();
    
    log_group_push();
        log_callstack("MEMORY", LOG_TYPE_FATAL, -1, 1);
    log_group_pop();

    log_flush_all();
    platform_trap(); 
    platform_abort();
}


void render_screen_quad()
{
    static GLuint quadVAO = 0;
    static GLuint quadVBO = 0;
	if (quadVAO == 0)
	{
		f32 quadVertices[] = {
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(f32), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(f32), (void*)(3 * sizeof(f32)));
	}

	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

void render_sci_texture(App_State* app, Compute_Texture texture, f32 min, f32 max)
{
    static Render_Shader sci_shader = {0};
    if(sci_shader.shader == 0)
    {
        Allocator_Set prev = allocator_set_default(allocator_get_static());
        Error error = render_shader_init_from_disk(&sci_shader, STRING("shaders/sci_color.frag_vert"));
        TEST_MSG(error_is_ok(error), "Error while loading shaders!");
        allocator_set(prev);
    }
    
    //platform_thread_sleep(1);
    compute_texture_bind(texture, GL_READ_ONLY, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    render_shader_set_f32(&sci_shader, "_min", min);
    render_shader_set_f32(&sci_shader, "_max", max);
    render_shader_use(&sci_shader);
	render_screen_quad();
            
    if(0)
        render_shader_deinit(&sci_shader);

    glfwSwapBuffers(app->window);
}

Compute_Texture compute_texture_make(isize width, isize heigth, GLuint type, GLuint channels)
{
    GLuint format = GL_RGBA;
    GLuint internal_format = 0;

    ASSERT(type == GL_FLOAT);
    switch(channels)
    {
        default:
        case 1: 
            format = GL_RED;
            internal_format = GL_R32F; 
        break;

        case 2: 
            format =  GL_RG;
            internal_format = GL_RG32F; 
        break;

        case 3: 
            format =  GL_RGB;
            internal_format = GL_RGB32F; 
        break;

        case 4: 
            format =  GL_RGBA;
            internal_format = GL_RGBA32F; 
        break;
    }

    Compute_Texture tex = {0};
	glGenTextures(1, &tex.id);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex.id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, internal_format, (GLuint) width, (GLuint) heigth, 0, format, type, NULL);
    
    tex.format = internal_format;
    tex.channels = channels;
    tex.pixel_type = type;

	glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

void compute_texture_bind(Compute_Texture texture, GLenum access, isize slot)
{
	glBindImageTexture((GLuint) slot, texture.id, 0, GL_FALSE, 0, access, texture.format);
    glBindTextureUnit((i32) slot, texture.id);
}

void compute_texture_deinit(Compute_Texture* texture)
{
    glDeleteTextures(1, &texture->id);
    memset(texture, 0, sizeof *texture);
}