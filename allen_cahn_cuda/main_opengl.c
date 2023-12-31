#define _CRT_SECURE_NO_WARNINGS
#define JOT_ALL_IMPL
//#define JOT_MEM_DEBUG

#include "lib/file.h"
#include "lib/log.h"
#include "lib/logger_file.h"
#include "lib/allocator_debug.h"
#include "lib/allocator_malloc.h"
#include "lib/error.h"
#include "lib/time.h"
#include "lib/image.h"
#include "lib/format_netbpm.h"

#include "gl_utils/gl.h"
#include "gl_utils/gl_debug_output.h"
#include "gl_utils/gl_shader_util.h"
#include "gl_utils/gl_pixel_format.h"

#include <GLFW/glfw3.h>

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
    GL_Pixel_Format format;

    i32 width;
    i32 heigth;
} Compute_Texture;

void render_sci_texture(App_State* app, Compute_Texture texture, f32 min, f32 max);
void compute_texture_bind(Compute_Texture texture, GLenum access, isize slot);
void compute_texture_deinit(Compute_Texture* texture);
GL_Pixel_Format gl_pixel_format_from_pixel_format(Image_Pixel_Format pixel_format, isize channels);
Image_Pixel_Format pixel_format_from_gl_pixel_format(GL_Pixel_Format gl_format, isize* channels);
Compute_Texture compute_texture_make_with(isize width, isize heigth, Image_Pixel_Format format, isize channels, const void* data);
Compute_Texture compute_texture_make(isize width, isize heigth, Image_Pixel_Format type, isize channels);

void compute_texture_set_pixels(Compute_Texture* texture, Image_Builder image);
void compute_texture_get_pixels(Image_Builder* into, Compute_Texture texture);

void compute_texture_set_pixels_converted(Compute_Texture* texture, Image_Builder image);
void compute_texture_get_pixels_converted(Image_Builder* into, Compute_Texture texture);

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

typedef f32 (*RK4_Func)(f32 t, const f32 x, void* context);

f32 runge_kutta4_var(f32 t_ini, f32 x_ini, f32 T, f32 tau_ini, f32 delta, RK4_Func f, void* context)
{
    (void) delta;
    f32 t = t_ini;
    f32 tau = tau_ini;
    f32 x = x_ini;

    f32 n = 1;
    for(;;)
    {
        bool last = false;
        if(fabsf(T - t) < fabs(tau))
        {
            tau = T - t;
            last = true;
        }
        else
        {
            last = false;
        }

        f32 k1 = f(t, x, context);
        f32 k2 = f(t + tau/3, x + tau/3*k1, context);
        f32 k3 = f(t + tau/3, x + tau/6*(k1 + k2), context);
        f32 k4 = f(t + tau/2, x + tau/8*(k1 + 3*k3), context);
        f32 k5 = f(t + tau/1, x + tau*(0.5f*k1 - 1.5f*k3 + 2*k4), context);
        
        f32 epsilon = 0;
        for(f32 i = 0; i < n; i++)
        {
            f32 curr = fabsf(0.2f*k1 - 0.9f*k3 + 0.8f*k4 - 0.1f*k5);
            epsilon = MAX(epsilon, curr);
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

f32 runge_kutta4(f32 t_ini, f32 x_ini, f32 T, f32 tau, RK4_Func f, void* context)
{
    f32 x = x_ini;
    for (f32 t = t_ini; t <= T; t += tau)
    {
        f32 k1 = tau*f(t, x, context);
        f32 k2 = tau*f(t + 0.5f*tau, x + 0.5f*k1, context);
        f32 k3 = tau*f(t + 0.5f*tau, x + 0.5f*k2, context);
        f32 k4 = tau*f(t + tau, x + k3, context);
  
        f32 x_next = x + (k1 + 2*k2 + 2*k3 + k4)/6;
        x = x_next;  
    }
  
    return x;
}

f32 euler(f32 t_ini, f32 x_ini, f32 T, f32 tau, RK4_Func f, void* context)
{
    f32 t = t_ini;
    f32 x = x_ini;

    for(; t <= T; )
    {
        f32 derx = f(t, x, context);
        f32 x_next = x + tau*derx;

        x = x_next;
        t += tau;
    }

    return x;
}


f32 semi_euler(f32 t_ini, f32 x_ini, f32 T, f32 tau, RK4_Func f, void* context)
{
    f32 t = t_ini;
    f32 x = x_ini;

    for(; t <= T; )
    {
        f32 derx = f(t + tau, x, context);
        f32 x_next = x + tau*derx;

        x = x_next;
        t += tau;
    }

    return x;
}

f32 rk4_func_x(f32 t, const f32 x, void* context)
{
    (void) context;
    (void) x;
    return t;
}

void compare_rk4()
{
    f32 t_ini = 0;
    f32 x_ini = 0;
    f32 tau = 0.001f;

    for(f32 T = 0; T < 5; T += 0.5f)
    {
        f32 euler_r = euler(t_ini, x_ini, T, tau, rk4_func_x, NULL);
        f32 semi_euler_r = semi_euler(t_ini, x_ini, T, tau, rk4_func_x, NULL);
        f32 rk4_var_r = runge_kutta4_var(t_ini, x_ini, T, tau, 0.01f, rk4_func_x, NULL);
        f32 rk4_r = runge_kutta4(t_ini, x_ini, T, tau, rk4_func_x, NULL);
        f32 exact_r = T*T/2;

        LOG_INFO("RK4", "T = %f", T);
        log_group_push();
            LOG_INFO("RK4", "exact:         %f", exact_r);
            LOG_INFO("RK4", "euler:         %f", euler_r);
            LOG_INFO("RK4", "semi_euler:    %f", semi_euler_r);
            LOG_INFO("RK4", "rk4_var_r:     %f", rk4_var_r);
            LOG_INFO("RK4", "rk4_r:         %f", rk4_r);
        log_group_pop();
    }
}

void run_func_allen_cahn(void* context)
{
    // compare_rk4();


    const i32 PAUSE_AFTER_SAVES = 3;
    const i32 SAVE_EVERY = -1;
    const char* const SAVE_FOLDER = "snapshots";
    const char* const SAVE_PREFIX = "v1";

    const i32 _SIZE_X = 1024;
    const i32 _SIZE_Y = _SIZE_X;
    
    const f32 _INITIAL_PHI = 1;
    const f32 _INITIAL_T = 0.5;
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

    Compute_Texture phi_map         = compute_texture_make(_SIZE_X, _SIZE_Y, PIXEL_FORMAT_F32, 1);
    Compute_Texture T_map           = compute_texture_make(_SIZE_X, _SIZE_Y, PIXEL_FORMAT_F32, 1);
    Compute_Texture next_phi_map    = compute_texture_make(_SIZE_X, _SIZE_Y, PIXEL_FORMAT_F32, 1);
    Compute_Texture next_T_map      = compute_texture_make(_SIZE_X, _SIZE_Y, PIXEL_FORMAT_F32, 1);
    Compute_Texture output_phi_map  = compute_texture_make(_SIZE_X, _SIZE_Y, PIXEL_FORMAT_F32, 1);
    Compute_Texture output_T_map    = compute_texture_make(_SIZE_X, _SIZE_Y, PIXEL_FORMAT_F32, 1);

    GLFWwindow* window = context;
    App_State* app = (App_State*) glfwGetWindowUserPointer(window); (void) app;

    platform_directory_create(string_make(SAVE_FOLDER), NULL);

    //Platform_Calendar_Time calendar_time = platform_epoch_time_to_calendar_time(platform_local_epoch_time());
    Platform_Calendar_Time calendar_time = platform_local_calendar_time_from_epoch_time(platform_epoch_time());

    String_Builder serialized_image = {0};

    Render_Shader compute_shader = {0};

    Error error = compute_shader_init_from_disk(&compute_shader, STRING("shaders/allen_cahn.comp"), WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, WORK_GROUP_SIZE_Z);
    TEST_MSG(error_is_ok(error), "Error while loading shaders!");

    i64 save_counter = 0;
    i64 frame_counter = 0;
    f64 frame_time_sum = 0;
    
    f64 fps_display_last_time_sum = 0;
    f64 fps_display_last_time = 0;
    
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

            compute_texture_bind(phi_map, GL_READ_ONLY, 0);
            compute_texture_bind(T_map, GL_READ_ONLY, 1);

            compute_texture_bind(next_phi_map, GL_WRITE_ONLY, 2);
            compute_texture_bind(next_T_map, GL_WRITE_ONLY, 3);
            
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
void glfw_resize_func(GLFWwindow* window, int width, int heigth);
void glfw_key_func(GLFWwindow* window, int key, int scancode, int action, int mods);

void run_func(void* context);
void error_func(void* context, Platform_Sandbox_Error error);

void func4() 
{
    // log_callstack("app", LOG_TYPE_TRACE, -1, 1);
    *(char*) NULL = 0;
    ASSERT(false);
}
void func3() {func4();}
void func2() {func3();}
void func1() {func2();}

void run_func(void* context)
{
    // log_callstack("app", LOG_TYPE_DEBUG, -1, 1);
    (void) context;
    for(int i = 0; i < 10; i++)
        func1();
}

int _main()
{
    platform_init();

    platform_exception_sandbox(
        run_func, NULL, 
        error_func, NULL);

    platform_deinit();
    return 0;
}

int main()
{
    platform_init();

    i64 epoch_time = platform_epoch_time();
    Platform_Calendar_Time now_local_ = platform_local_calendar_time_from_epoch_time(epoch_time);
    ASSERT(epoch_time == platform_epoch_time_from_local_calendar_time(now_local_));

    Malloc_Allocator static_allocator = {0};
    malloc_allocator_init(&static_allocator);
    allocator_set_static(&static_allocator.allocator);
    
    Malloc_Allocator malloc_allocator = {0};
    malloc_allocator_init_use(&malloc_allocator, 0);
    
    error_system_init(&static_allocator.allocator);
    file_logger_init_use(&global_logger, &malloc_allocator.allocator, &malloc_allocator.allocator);

    Debug_Allocator debug_alloc = {0};
    debug_allocator_init_use(&debug_alloc, allocator_get_default(), DEBUG_ALLOCATOR_DEINIT_LEAK_CHECK | DEBUG_ALLOCATOR_CAPTURE_CALLSTACK);

    // GLFWallocator allocator = {0};
    // allocator.allocate = glfw_malloc_func;
    // allocator.reallocate = glfw_realloc_func;
    // allocator.deallocate = glfw_free_func;
    // allocator.user = &malloc_allocator;
 
    // glfwInitAllocator(&allocator);
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
    return malloc_allocator_malloc((Malloc_Allocator*) user, (isize) size);
}

void* glfw_realloc_func(void* block, size_t size, void* user)
{
    return malloc_allocator_realloc((Malloc_Allocator*) user, block, (isize) size);
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

Compute_Texture compute_texture_make_with(isize width, isize heigth, Image_Pixel_Format format, isize channels, const void* data)
{
    GL_Pixel_Format pixel_format = gl_pixel_format_from_pixel_format(format, channels);
    ASSERT(pixel_format.unrepresentable == false);
    
    Compute_Texture tex = {0};
	glGenTextures(1, &tex.id);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex.id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, (GLint) pixel_format.internal_format, (GLsizei) width, (GLsizei) heigth, 0, pixel_format.format, pixel_format.type, data);

    tex.format = pixel_format;
    tex.width = (i32) width;
    tex.heigth = (i32) heigth;

	glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

Compute_Texture compute_texture_make(isize width, isize heigth, Image_Pixel_Format type, isize channels)
{
    return compute_texture_make_with(width, heigth, type, channels, NULL);
}

void compute_texture_bind(Compute_Texture texture, GLenum access, isize slot)
{
	glBindImageTexture((GLuint) slot, texture.id, 0, GL_FALSE, 0, access, texture.format.internal_format);
    glBindTextureUnit((GLuint) slot, texture.id);
}

void compute_texture_deinit(Compute_Texture* texture)
{
    glDeleteTextures(1, &texture->id);
    memset(texture, 0, sizeof *texture);
}

void compute_texture_set_pixels(Compute_Texture* texture, Image_Builder image)
{
    compute_texture_deinit(texture);
    *texture = compute_texture_make_with(image.width, image.height, image.pixel_format, image_builder_channel_count(image), image.pixels);
}

void compute_texture_get_pixels(Image_Builder* into, Compute_Texture texture)
{
    image_builder_init(into, into->allocator, texture.format.channels, texture.format.equivalent);
    image_builder_resize(into, (i32) texture.width, (i32) texture.heigth);

    glGetTextureImage(texture.id, 0, texture.format.format, texture.format.type, (GLsizei) image_builder_all_pixels_size(*into), into->pixels);
}

void compute_texture_set_pixels_converted(Compute_Texture* texture, Image_Builder image)
{
    if(texture->width != image.width || texture->heigth != image.height)
    {
        Image_Pixel_Format prev_pixe_format = texture->format.equivalent;
        isize prev_channel_count = texture->format.channels;

        compute_texture_deinit(texture);
        *texture = compute_texture_make(image.width, image.height, prev_pixe_format, prev_channel_count);
    }

    GL_Pixel_Format gl_format = gl_pixel_format_from_pixel_format(image.pixel_format, image_builder_channel_count(image));
    glTextureSubImage2D(texture->id, 0, 0, 0, image.width, image.height, gl_format.format, gl_format.type, image.pixels);
}

void compute_texture_get_pixels_converted(Image_Builder* into, Compute_Texture texture)
{
    image_builder_resize(into, (i32) texture.width, (i32) texture.heigth);
    GL_Pixel_Format gl_format = gl_pixel_format_from_pixel_format(into->pixel_format, image_builder_channel_count(*into));
    
    glGetTextureImage(texture.id, 0, gl_format.format, gl_format.type, (GLsizei) image_builder_all_pixels_size(*into), into->pixels);
}

#if PLATFORM_OS == PLATFORM_OS_WINDOWS
    #error WINDOWS?!
    #include "lib/platform_windows.c"
#elif PLATFORM_OS == PLATFORM_OS_UNIX
    #include "lib/platform_linux.c"
#else
    #error Provide support for this operating system or define PLATFORM_OS to one of the values in platform.h
#endif