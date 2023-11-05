
#define LIB_ALL_IMPL
//#define LIB_MEM_DEBUG
#define DEBUG
#define GLAD_GL_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION


#include "lib/log.h"
#include "lib/logger_file.h"
#include "lib/allocator_debug.h"
#include "lib/allocator_malloc.h"
#include "lib/error.h"

#include "gl.h"
#include "gl_debug_output.h"
#include "gl_shader_util.h"

void run_func(void* context);
void run_test_func(void* context);
void error_func(void* context, Platform_Sandox_Error error_code);

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
 
    GLFWwindow* window = glfwCreateWindow(1600, 900, "Render", NULL, NULL);
    TEST_MSG(window != NULL, "Failed to make glfw window");

    App_State app = {0};
    glfwSetWindowUserPointer(window, &app);
    glfwMakeContextCurrent(window);

    int version = gladLoadGL((GLADloadfunc) glfwGetProcAddress);
    TEST_MSG(version != 0, "Failed to load opengl with glad");

    gl_debug_output_enable();

    #ifndef RUN_TESTS
    platform_exception_sandbox(
        run_func, window, 
        error_func, window);
    #else
    platform_exception_sandbox(
        run_test_func, window, 
        error_func, window);
    #endif
    glfwDestroyWindow(window);
    glfwTerminate();

    debug_allocator_deinit(&debug_alloc);
    
    file_logger_deinit(&global_logger);
    error_system_deinit();

    //@TODO: fix
    ASSERT(malloc_allocator.bytes_allocated == 0);
    malloc_allocator_deinit(&malloc_allocator);
    platform_deinit();

    return 0;    
}


void error_func(void* context, Platform_Sandox_Error error_code)
{
    (void) context;
    const char* msg = platform_sandbox_error_to_cstring(error_code);
    
    LOG_ERROR("APP", "%s exception occured", msg);
    LOG_TRACE("APP", "printing trace:");
    log_group_push();
    log_callstack("APP", LOG_TYPE_ERROR, -1, 1);
    log_group_pop();
}