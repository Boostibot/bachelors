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

//Logs a message. Does not get dissabled.
#define LOG(module, log_type, format, ...)   log_message(module, log_type, __LINE__, __FILE__, __FUNCTION__, format, ##__VA_ARGS__)
#define VLOG(module, log_type, format, args) vlog_message(module, log_type, __LINE__, __FILE__, __FUNCTION__, format, args)
#define _LOG_HERE(fmt_extra, ...) LOG("here", LOG_TRACE, "> %s %s:%i %s" fmt_extra, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define LOG_HERE(...) _LOG_HERE("", ##__VA_ARGS__)

//Logs a message type into the provided module cstring.
#define LOG_INFO(module, format, ...)  LOG(module, LOG_INFO,  format, ##__VA_ARGS__)
#define LOG_OKAY(module, format, ...)  LOG(module, LOG_OKAY,  format, ##__VA_ARGS__)
#define LOG_WARN(module, format, ...)  LOG(module, LOG_WARN,  format, ##__VA_ARGS__)
#define LOG_ERROR(module, format, ...) LOG(module, LOG_ERROR, format, ##__VA_ARGS__)
#define LOG_FATAL(module, format, ...) LOG(module, LOG_FATAL, format, ##__VA_ARGS__)
#define LOG_DEBUG(module, format, ...) LOG(module, LOG_DEBUG, format, ##__VA_ARGS__)
#define LOG_TRACE(module, format, ...) LOG(module, LOG_TRACE, format, ##__VA_ARGS__)

typedef struct Str_Buffer32 {
    char str[32];
} Str_Buffer32;

typedef struct Str_Buffer16 {
    char str[16];
} Str_Buffer16;

Str_Buffer16 format_bytes(size_t bytes);

#define TIME_FMT "%02i:%02i:%02i %03i"
#define TIME_PRINT(c) (int)(c).hour, (int)(c).minute, (int)(c).second, (int)(c).millisecond

#define STRING_FMT "%.*s"
#define STRING_PRINT(string) (int) (string).size, (string).data

#define DO_LOG

#include <stdio.h>
#include <string.h>

enum {
    FILE_LOGGER_FILE_PATH = 1,
    FILE_LOGGER_FILE_APPEND = 2,
    FILE_LOGGER_NO_CONSOLE_PRINT = 4,
    FILE_LOGGER_NO_CONSOLE_COLORS = 8,
};
typedef struct File_Logger {
    Logger logger;
    FILE* file;
    char* path;

    int flags;
} File_Logger;

void file_logger_log(Logger* gen_logger, const char* module, Log_Type type, size_t indentation, int line, const char* file, const char* function, const char* format, va_list args);
bool file_logger_init(File_Logger* logger, const char* path, int flags);
void file_logger_deinit(File_Logger* logger);
Logger* console_logger();
#endif

// #define JOT_ALL_IMPL
#if (defined(JOT_ALL_IMPL) || defined(JOT_LOG_IMPL)) && !defined(JOT_LOG_HAS_IMPL)
#define JOT_LOG_HAS_IMPL
static File_Logger _console_logger = {file_logger_log, NULL, NULL, 0};
static Logger* _global_logger = &_console_logger.logger;
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

char* malloc_vfmt(size_t* count_or_null, char* backing, size_t backing_size, const char* fmt, va_list args)
{
    va_list copy;
    va_copy(copy, args);

    int count = 0;
    char* out = NULL;
    if(backing_size >= 0) 
    {
        count = vsnprintf(backing, backing_size, fmt, copy);
        out = backing;
    }
    if(out == NULL || (size_t) count >= backing_size)
    {
        va_list copy2;
        va_copy(copy2, args);
        count = vsnprintf(backing, backing_size, fmt, copy2);

        out = (char*) malloc((size_t) count + 1);
        vsnprintf(out, (size_t) count + 1, fmt, args);
        out[count] = '\0';
    }
    if(count_or_null)
        *count_or_null = (size_t) count;
    return out;
}

char* malloc_fmt_custom(size_t* count_or_null, char* backing, size_t backing_size, const char* fmt, ...)
{
    va_list args;               
    va_start(args, fmt);    
    char* out =  malloc_vfmt(count_or_null, backing, backing_size, fmt, args);
    va_end(args);  
    return out;
}

#include <string.h>
#include <ctype.h>
#include <time.h>
	
void file_logger_log(Logger* gen_logger, const char* module, Log_Type type, size_t indentation, int line, const char* file, const char* function, const char* format, va_list args)
{
    File_Logger* logger = (File_Logger*) (void*) gen_logger;
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

    (void) line;
    (void) file;
    (void) function;
    if(type == LOG_FLUSH)
    {
        if(logger->file)
            fflush(logger->file);
    }
    else
    {
        timespec ts = {0};
        (void) timespec_get(&ts, TIME_UTC);
        struct tm* now = gmtime(&ts.tv_sec);

        size_t user_max_len = 0;
        char user_backing[512]; (void) user_backing;
        char* user = malloc_vfmt(&user_max_len, user_backing, sizeof user_backing, format, args);
        size_t user_len = user_max_len;

        //trim trailing whitespace
        for(; user_len > 0; user_len--)
            if(!isspace(user[user_len - 1]))
                break;

        size_t complete_len = 0;
        char complete_backing[512]; (void) complete_backing;
         char* complete = malloc_fmt_custom(&complete_len, complete_backing, sizeof complete_backing, 
            "%02i:%02i:%02i %5s %6s: %.*s%.*s", 
            now->tm_hour, now->tm_min, now->tm_sec, log_type_to_string(type), module, (int) indentation, "", (int) user_len, user);

        if((logger->flags & FILE_LOGGER_NO_CONSOLE_PRINT) == 0)
        {
            if(logger->flags & FILE_LOGGER_NO_CONSOLE_COLORS)
                puts(complete);
            else
            {
                const char* line_begin = "";
                const char* line_end = "";
                if(type == LOG_ERROR || type == LOG_FATAL)
                    line_begin = ANSI_COLOR_BRIGHT_RED;
                else if(type == LOG_WARN)
                    line_begin = ANSI_COLOR_YELLOW;
                else if(type == LOG_OKAY)
                    line_begin = ANSI_COLOR_GREEN;
                else if(type == LOG_TRACE || type == LOG_DEBUG)
                    line_begin = ANSI_COLOR_GRAY;
                else    
                    line_begin = ANSI_COLOR_NORMAL;

                line_end = ANSI_COLOR_NORMAL;
                printf("%s%s%s\n", line_begin, complete, line_end);
            }
        }
        
        if(logger->file)
            fprintf(logger->file, "%s\n", complete);

        if(user != user_backing)
            free(user);

        if(complete != complete_backing)
            free(complete);
    }
}

void file_logger_deinit(File_Logger* logger)
{
    if(logger->file)
        fclose(logger->file);
    free(logger->path);
    memset(logger, 0, sizeof *logger);
}

bool file_logger_init(File_Logger* logger, const char* path, int flags)
{
    file_logger_deinit(logger);

    const char* open_mode = flags & FILE_LOGGER_FILE_APPEND ? "ab" : "wb"; 
    char* filename = NULL;
    if((flags & FILE_LOGGER_FILE_PATH) && path)
        filename = malloc_fmt_custom(NULL, NULL, 0, "%s", path);
    else
    {
        timespec ts = {0};
        (void) timespec_get(&ts, TIME_UTC);
        struct tm* now = localtime(&ts.tv_sec);
        filename = malloc_fmt_custom(NULL, NULL, 0, 
            "%s%02i-%02i-%02i__%02i-%02i-%02i.log", 
            path ? path : "logs/", now->tm_year, now->tm_mon, now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
    }

    FILE* file = fopen(filename, open_mode);
    
    logger->file = file;
    logger->path = filename;
    logger->flags = flags;
    logger->logger.log = file_logger_log;

    return file != NULL;
}

Logger* console_logger()
{
    return &_console_logger.logger;
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

Str_Buffer16 format_bytes(size_t bytes)
{
    size_t B  = (size_t) 1;
    size_t KB = (size_t) 1024;
    size_t MB = (size_t) 1024*1024;
    size_t GB = (size_t) 1024*1024*1024;
    size_t TB = (size_t) 1024*1024*1024*1024;

    const char* unit = "";
    size_t unit_value = 1;
    if(bytes >= TB)
    {
        unit = "TB";
        unit_value = TB;
    }
    else if(bytes >= GB)
    {
        unit = "GB";
        unit_value = GB;
    }
    else if(bytes >= MB)
    {
        unit = "MB";
        unit_value = MB;
    }
    else if(bytes >= KB)
    {
        unit = "KB";
        unit_value = KB;
    }
    else
    {
        unit = "B";
        unit_value = B;
    }

    Str_Buffer16 out = {0};
    double value = (double) bytes / (double) unit_value;
    snprintf(out.str, sizeof(out.str), "%.2lf%s", value, unit);
    return out;
}

#endif
