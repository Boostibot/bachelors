
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

#include "glfw/glfw3.h"

void run_func(void* context);
void error_func(void* context, Platform_Sandox_Error error_code);

typedef struct App_State {
    int a;
} App_State;

void run_func(void* context)
{
    App_State* app = (App_State*) context;
    LOG_INFO("APP", "Hello world %lli", (lli) app->a);
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

    platform_exception_sandbox(
        run_func, window, 
        error_func, window);

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