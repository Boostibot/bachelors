#ifndef JOT_LOG
#define JOT_LOG

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

enum {
    LOG_ENUM_MAX = 63,  //This is the maximum value log types are allowed to have without being ignored.
    LOG_FLUSH = 63,     //only flushes the log but doesnt log anything
    LOG_INFO = 0,       //Used to log general info.
    LOG_OKAY = 1,    //Used to log the opposites of errors
    LOG_WARN = 2,       //Used to log near error conditions
    LOG_ERROR = 3,      //Used to log errors
    LOG_FATAL = 4,      //Used to log errors just before giving up some important action
    LOG_DEBUG = 5,      //Used to log for debug purposes. Is only logged in debug builds
    LOG_TRACE = 6,      //Used to log for step debug purposes (prinf("HERE") and such). Is only logged in step debug builds
};

typedef int Log_Type;
typedef struct Logger Logger;
typedef void (*Vlog_Func)(Logger* logger, const char* module, Log_Type type, size_t indentation, int line, const char* file, const char* function, const char* format, va_list args);

typedef struct Logger {
    Vlog_Func log;
} Logger;

//Returns the default used logger
Logger* log_system_get_logger();
//Sets the default used logger. Returns a pointer to the previous logger so it can be restored later.
Logger* log_system_set_logger(Logger* logger);

void    log_group();   //Increases indentation of subsequent log messages
void    log_ungroup();    //Decreases indentation of subsequent log messages
size_t* log_group_depth(); //Returns the current indentation of messages

void log_message(const char* module, Log_Type type, int line, const char* file, const char* function, const char* format, ...);
void vlog_message(const char* module, Log_Type type, int line, const char* file, const char* function, const char* format, va_list args);
void log_flush();

const char* log_type_to_string(Log_Type type);
Logger def_logger_make();

//Logs a message. Does not get dissabled.
#define LOG(module, log_type, format, ...)   log_message(module, log_type, __LINE__, __FILE__, __FUNCTION__, format, ##__VA_ARGS__)
#define VLOG(module, log_type, format, args) vlog_message(module, log_type, __LINE__, __FILE__, __FUNCTION__, format, args)
#define LOG_HERE() LOG("here", LOG_TRACE, "> %s %s:%i", __FUNCTION__, __FILE__, __LINE__)

//Logs a message type into the provided module cstring.
#define LOG_INFO(module, format, ...)  LOG(module, LOG_INFO,  format, ##__VA_ARGS__)
#define LOG_OKAY(module, format, ...)  LOG(module, LOG_OKAY,  format, ##__VA_ARGS__)
#define LOG_WARN(module, format, ...)  LOG(module, LOG_WARN,  format, ##__VA_ARGS__)
#define LOG_ERROR(module, format, ...) LOG(module, LOG_ERROR, format, ##__VA_ARGS__)
#define LOG_FATAL(module, format, ...) LOG(module, LOG_FATAL, format, ##__VA_ARGS__)
#define LOG_DEBUG(module, format, ...) LOG(module, LOG_DEBUG, format, ##__VA_ARGS__)
#define LOG_TRACE(module, format, ...) LOG(module, LOG_TRACE, format, ##__VA_ARGS__)

typedef struct Memory_Format {
    const char* unit;
    size_t unit_value;
    double fraction;

    int whole;
    int remainder;
} Memory_Format;

Memory_Format get_memory_format(size_t bytes);

#define TIME_FMT "%02i:%02i:%02i %03i"
#define TIME_PRINT(c) (int)(c).hour, (int)(c).minute, (int)(c).second, (int)(c).millisecond

#define STRING_FMT "%.*s"
#define STRING_PRINT(string) (int) (string).size, (string).data

#define MEMORY_FMT "%.2lf%s"
#define MEMORY_PRINT(bytes) get_memory_format((bytes)).fraction, get_memory_format((bytes)).unit

#define DO_LOG
#endif

#if (defined(JOT_ALL_IMPL) || defined(JOT_LOG_IMPL)) && !defined(JOT_LOG_HAS_IMPL)
#define JOT_LOG_HAS_IMPL

static void def_logger_func(Logger* logger, const char* module, Log_Type type, size_t indentation, int line, const char* file, const char* function, const char* format, va_list args);

static Logger _global_def_logger = {def_logger_func};
static Logger* _global_logger = &_global_def_logger;
static size_t _global_log_group_depth = 0;

Logger* log_system_get_logger()
{
    return _global_logger;
}

Logger* log_system_set_logger(Logger* logger)
{
    Logger* before = _global_logger;
    _global_logger = logger;
    return before;
}

void log_group()
{
    _global_log_group_depth ++;
}
void log_ungroup()
{
    _global_log_group_depth --;
}
size_t* log_group_depth()
{
    return &_global_log_group_depth;
}

void vlog_message(const char* module, Log_Type type, int line, const char* file, const char* function, const char* format, va_list args)
{
    bool static_enabled = false;
    #ifdef DO_LOG
        static_enabled = true;
    #endif
    Logger* global_logger = _global_logger;
    if(static_enabled && global_logger)
    {
        size_t extra_indentation = 0;
        for(; module[extra_indentation] == '>'; extra_indentation++);

        global_logger->log(global_logger, module + extra_indentation, type, _global_log_group_depth + extra_indentation, line, file, function, format, args);
    }
}

void log_message(const char* module, Log_Type type, int line, const char* file, const char* function, const char* format, ...)
{
    va_list args;               
    va_start(args, format);     
    vlog_message(module, type, line, file, function, format, args);                    
    va_end(args);                
}

void log_flush()
{
    log_message("", LOG_FLUSH, __LINE__, __FILE__, __FUNCTION__, " ");
}

const char* log_type_to_string(Log_Type type)
{
    switch(type)
    {
        case LOG_FLUSH: return "FLUSH"; break;
        case LOG_INFO: return "INFO"; break;
        case LOG_OKAY: return "SUCC"; break;
        case LOG_WARN: return "WARN"; break;
        case LOG_ERROR: return "ERROR"; break;
        case LOG_FATAL: return "FATAL"; break;
        case LOG_DEBUG: return "DEBUG"; break;
        case LOG_TRACE: return "TRACE"; break;
        default: return "";
    }
}

Logger def_logger_make()
{
    Logger out = {def_logger_func};
    return out;
}

#include <ctime>
static void def_logger_func(Logger* logger, const char* module, Log_Type type, size_t indentation, int line, const char* file, const char* function, const char* format, va_list args)
{
    //Some of the ansi colors that can be used within logs. 
    //However their usage is not recommended since these will be written to log files and thus make their parsing more difficult.
    #define ANSI_COLOR_NORMAL       "\x1B[0m"
    #define ANSI_COLOR_RED          "\x1B[31m"
    #define ANSI_COLOR_BRIGHT_RED   "\x1B[91m"
    #define ANSI_COLOR_GREEN        "\x1B[32m"
    #define ANSI_COLOR_YELLOW       "\x1B[33m"
    #define ANSI_COLOR_BLUE         "\x1B[34m"
    #define ANSI_COLOR_MAGENTA      "\x1B[35m"
    #define ANSI_COLOR_CYAN         "\x1B[36m"
    #define ANSI_COLOR_WHITE        "\x1B[37m"
    #define ANSI_COLOR_GRAY         "\x1B[90m"

    (void) logger;
    (void) line;
    (void) file;
    (void) function;
    if(type == LOG_FLUSH)
        return;

    std::timespec ts = {0};
    (void) std::timespec_get(&ts, TIME_UTC);
    struct tm* now = std::gmtime(&ts.tv_sec);

    const char* color_mode = ANSI_COLOR_NORMAL;
    if(type == LOG_ERROR || type == LOG_FATAL)
        color_mode = ANSI_COLOR_BRIGHT_RED;
    else if(type == LOG_WARN)
        color_mode = ANSI_COLOR_YELLOW;
    else if(type == LOG_OKAY)
        color_mode = ANSI_COLOR_GREEN;
    else if(type == LOG_TRACE || type == LOG_DEBUG)
        color_mode = ANSI_COLOR_GRAY;

    typedef long long int lli;

    printf("%s%02i:%02i:%02lli %5s %6s: ", color_mode, now->tm_hour, now->tm_min, (lli) now->tm_sec, log_type_to_string(type), module);

    //We only do fancy stuff when there is something to print.
    //We also only print extra newline if there is none already.
    size_t fmt_size = strlen(format);
    bool print_newline = true;
    if(fmt_size > 0)
    {
        for(size_t i = 0; i < indentation; i++)
            printf("   ");

        vprintf(format, args);
        if(format[fmt_size - 1] == '\n')
            print_newline = false; 
    }

    if(print_newline)
        printf(ANSI_COLOR_NORMAL"\n");
    else
        printf(ANSI_COLOR_NORMAL);
        
}

void assertion_report(const char* expression, int line, const char* file, const char* function, const char* message, ...)
{
    log_message("assert", LOG_FATAL, line, file, function, "TEST(%s) TEST/ASSERT failed! (%s %s: %lli) ", expression, file, function, line);
    if(message != NULL && strlen(message) != 0)
    {
        va_list args;               
        va_start(args, message);     
        vlog_message(">assert", LOG_FATAL, line, file, function, message, args);
        va_end(args);  
    }

    log_flush();
}

Memory_Format get_memory_format(size_t bytes)
{
    size_t B  = (size_t) 1;
    size_t KB = (size_t) 1024;
    size_t MB = (size_t) 1024*1024;
    size_t GB = (size_t) 1024*1024*1024;
    size_t TB = (size_t) 1024*1024*1024*1024;

    Memory_Format out = {0};
    out.unit = "";
    out.unit_value = 1;
    if(bytes >= TB)
    {
        out.unit = "TB";
        out.unit_value = TB;
    }
    else if(bytes >= GB)
    {
        out.unit = "GB";
        out.unit_value = GB;
    }
    else if(bytes >= MB)
    {
        out.unit = "MB";
        out.unit_value = MB;
    }
    else if(bytes >= KB)
    {
        out.unit = "KB";
        out.unit_value = KB;
    }
    else
    {
        out.unit = "B";
        out.unit_value = B;
    }

    out.fraction = (double) bytes / (double) out.unit_value;
    out.whole = (int) (bytes / out.unit_value);
    out.remainder = (int) (bytes / out.unit_value);

    return out;
}


#endif
