#ifndef JOT_LOG
#define JOT_LOG

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifndef LOG_CUSTOM_SETTINGS
    #define DO_LOG          /* Disables all log types */   
    #define DO_LOG_INFO
    #define DO_LOG_SUCCESS
    #define DO_LOG_WARN 
    #define DO_LOG_ERROR
    #define DO_LOG_FATAL

    #ifndef NDEBUG
    #define DO_LOG_DEBUG
    #define DO_LOG_TRACE
    #endif
#endif

enum {
    LOG_ENUM_MAX = 63,  //This is the maximum value log types are allowed to have without being ignored.
    LOG_FLUSH = 63,     //only flushes the log but doesnt log anything
    LOG_INFO = 0,       //Used to log general info.
    LOG_SUCCESS = 1,    //Used to log the opposites of errors
    LOG_WARN = 2,       //Used to log near error conditions
    LOG_ERROR = 3,      //Used to log errors
    LOG_FATAL = 4,      //Used to log errors just before giving up some important action
    LOG_DEBUG = 5,      //Used to log for debug purposes. Is only logged in debug builds
    LOG_TRACE = 6,      //Used to log for step debug purposes (prinf("HERE") and such). Is only logged in step debug builds
};

#ifndef _MSC_VER
    #define __FUNCTION__ __func__
#endif


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

void  log_group_push();   //Increases indentation of subsequent log messages
void  log_group_pop();    //Decreases indentation of subsequent log messages
size_t log_group_depth(); //Returns the current indentation of messages

void log_message(const char* module, Log_Type type, int line, const char* file, const char* function, const char* format, ...);
void vlog_message(const char* module, Log_Type type, int line, const char* file, const char* function, const char* format, va_list args);
void log_flush();

const char* log_type_to_string(Log_Type type);
Logger def_logger_make();
void def_logger_func(Logger* logger, const char* module, Log_Type type, size_t indentation, int line, const char* file, const char* function, const char* format, va_list args);

//Default logging facility. Logs a message into the provided module cstring with log_type type (info, warn, error...)
#define LOG(module, log_type, format, ...)      PP_IF(DO_LOG, LOG_ALWAYS)(module, log_type, format, ##__VA_ARGS__)
#define VLOG(module, log_type, format, args)    PP_IF(DO_LOG, LOG_ALWAYS)(module, log_type, format, args)

//Logs a message type into the provided module cstring.
#define LOG_INFO(module, format, ...)           PP_IF(DO_LOG_INFO, LOG)(module, LOG_INFO, format, ##__VA_ARGS__)
#define LOG_SUCCESS(module, format, ...)        PP_IF(DO_LOG_SUCCESS, LOG)(module, LOG_SUCCESS, format, ##__VA_ARGS__)
#define LOG_WARN(module, format, ...)           PP_IF(DO_LOG_WARN, LOG)(module, LOG_WARN, format, ##__VA_ARGS__)
#define LOG_ERROR(module, format, ...)          PP_IF(DO_LOG_ERROR, LOG)(module, LOG_ERROR, format, ##__VA_ARGS__)
#define LOG_FATAL(module, format, ...)          PP_IF(DO_LOG_FATAL, LOG)(module, LOG_FATAL, format, ##__VA_ARGS__)
#define LOG_DEBUG(module, format, ...)          PP_IF(DO_LOG_DEBUG, LOG)(module, LOG_DEBUG, format, ##__VA_ARGS__)
#define LOG_TRACE(module, format, ...)          PP_IF(DO_LOG_TRACE, LOG)(module, LOG_TRACE, format, ##__VA_ARGS__)

//Logs a message. Does not get dissabled.
#define LOG_ALWAYS(module, log_type, format, ...)   log_message(module, log_type, __LINE__, __FILE__, __FUNCTION__, format, ##__VA_ARGS__)
#define VLOG_ALWAYS(module, log_type, format, args) vlog_message(module, log_type, __LINE__, __FILE__, __FUNCTION__, format, args)
//Does not do anything (failed condition) but type checks the arguments
#define LOG_NEVER(module, log_type, format, ...)  ((module && false) ? log_message(module, log_type, __LINE__, __FILE__, __FUNCTION__, format, ##__VA_ARGS__) : (void) 0)

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

//Gets expanded when the particular type is dissabled.
#define _IF_NOT_DO_LOG(ignore)              LOG_NEVER
#define _IF_NOT_DO_LOG_INFO(ignore)         LOG_NEVER
#define _IF_NOT_DO_LOG_SUCCESS(ignore)      LOG_NEVER
#define _IF_NOT_DO_LOG_WARN(ignore)         LOG_NEVER
#define _IF_NOT_DO_LOG_ERROR(ignore)        LOG_NEVER
#define _IF_NOT_DO_LOG_FATAL(ignore)        LOG_NEVER
#define _IF_NOT_DO_LOG_DEBUG(ignore)        LOG_NEVER
#define _IF_NOT_DO_LOG_TRACE(ignore)        LOG_NEVER


#define TIME_FMT "%02i:%02i:%02i %03i"
#define TIME_PRINT(c) (int)(c).hour, (int)(c).minute, (int)(c).second, (int)(c).millisecond

#define STRING_FMT "%.*s"
#define STRING_PRINT(string) (int) (string).size, (string).data

#define SOURCE_INFO_FMT "( %s : %i )"
#define SOURCE_INFO_PRINT(source_info) (source_info).file, (int) (source_info).line

//Pre-Processor (PP) utils
#define PP_STRINGIFY_(x)        #x
#define PP_CONCAT2(a, b)        a ## b
#define PP_CONCAT3(a, b, c)     PP_CONCAT2(PP_CONCAT2(a, b), c)
#define PP_CONCAT4(a, b, c, d)  PP_CONCAT2(PP_CONCAT3(a, b, c), d)
#define PP_CONCAT(a, b)         PP_CONCAT2(a, b)
#define PP_STRINGIFY(...)       PP_STRINGIFY_(__VA_ARGS__)
#define PP_ID(x)                x

//if CONDITION_DEFINE is defined: expands to x, 
//else: expands to _IF_NOT_##CONDITION_DEFINE(x). See above how to use this.
//The reason for its use is that simply all other things I have tried either didnt
// work or failed to compose for obscure reasons
#define PP_IF(CONDITION_DEFINE, x)         PP_CONCAT(_IF_NOT_, CONDITION_DEFINE)(x)
#define _IF_NOT_(x) x

#endif

#if (defined(JOT_ALL_IMPL) || defined(JOT_LOG_IMPL)) && !defined(JOT_LOG_HAS_IMPL)
#define JOT_LOG_HAS_IMPL

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

void log_group_push()
{
    _global_log_group_depth ++;
}
void log_group_pop()
{
    _global_log_group_depth --;
}
size_t log_group_depth()
{
    return _global_log_group_depth;
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
        case LOG_SUCCESS: return "SUCC"; break;
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
void def_logger_func(Logger* logger, const char* module, Log_Type type, size_t indentation, int line, const char* file, const char* function, const char* format, va_list args)
{
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
    else if(type == LOG_SUCCESS)
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
#endif
