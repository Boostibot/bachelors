#ifndef JOT_ASSERT
#define JOT_ASSERT

#include "log.h"
#include <stdlib.h>

#if defined(_MSC_VER)
    #define platform_debug_break() __debugbreak() 
#elif defined(__GNUC__) || defined(__clang__)
    #ifndef _GNU_SOURCE
    #define _GNU_SOURCE
    #endif
    #include <signal.h>
    // #define platform_debug_break() __builtin_trap() /* bad looks like a fault in program! */
    #define platform_debug_break() raise(SIGTRAP)
#else
    #error unsupported compiler. Please add your own definition
#endif

#if !defined(ASSERT_CUSTOM_SETTINGS) && !defined(NDEBUG)
    //Locally enables/disables asserts. If we wish to disable for part of
    // code we simply undefine them then redefine them after.
    #define DO_ASSERTS       /* enables assertions */
    #define DO_ASSERTS_SLOW  /* enables slow assertions - expensive assertions or once that change the time complexity of an algorhitm */
    #define DO_BOUNDS_CHECKS /* checks bounds prior to lookup */
#endif

//If x evaluates to false executes assertion_report() without any message. 
#define TEST(x)                 TEST_MSG(x, "")              /* executes always (even in release) */
#define ASSERT(x)               ASSERT_MSG(x, "")            /* is enabled by DO_ASSERTS */
#define ASSERT_SLOW(x)          ASSERT_SLOW_MSG(x, "")       /* is enabled by DO_ASSERTS_SLOW */

//If x evaluates to false executes assertion_report() with the specified message. 
#define TEST_MSG(x, msg, ...)               (!(x) ? (assertion_report(#x, __LINE__, __FILE__, __FUNCTION__, (msg), ##__VA_ARGS__), (platform_debug_break()), abort()) : (void) 0)
#define ASSERT_MSG(x, msg, ...)             PP_IF(DO_ASSERTS,       TEST_MSG)(x, msg, ##__VA_ARGS__)
#define ASSERT_SLOW_MSG(x, msg, ...)        PP_IF(DO_ASSERTS_SLOW,  TEST_MSG)(x, msg, ##__VA_ARGS__)

//Gets called when assertion fails. 
//Does not have to terminate process since that is done at call site by the assert macro itself.
//if ASSERT_CUSTOM_REPORT is defined is left unimplemented
void assertion_report(const char* expression, int line, const char* file, const char* function, const char* message, ...);

void default_assertion_report(const char* expression, int line, const char* file, const char* function, const char* message, va_list args);

//==================== IMPLEMENTATION =======================

    //Doesnt do anything (failed branch) but still properly expands x and msg so it can be type checked.
    //Dissabled asserts expand to this.
    #define DISSABLED_TEST_MSG(x, msg, ...)           (0 ? ((void) (x), assertion_report("", __LINE__, __FILE__, __FUNCTION__, (msg), ##__VA_ARGS__)) : (void) 0)

    //If dissabled expand to this
    #define _IF_NOT_DO_ASSERTS(ignore)         DISSABLED_TEST_MSG
    #define _IF_NOT_DO_ASSERTS_SLOW(ignore)    DISSABLED_TEST_MSG

#endif

#if (defined(JOT_ALL_IMPL) || defined(JOT_ASSERT_IMPL)) && !defined(JOT_ASSERT_HAS_IMPL)
#define JOT_ASSERT_HAS_IMPL

    #ifndef ASSERT_CUSTOM_REPORT
        void assertion_report(const char* expression, int line, const char* file, const char* function, const char* message, ...)
        {
            va_list args;               
            va_start(args, message);     
            default_assertion_report(expression, line, file, function, message, args);                    
            va_end(args);  
        }
    #endif

    void default_assertion_report(const char* expression, int line, const char* file, const char* function, const char* message, va_list args)
    {
        log_message("assert", LOG_FATAL, line, file, function, "TEST(%s) TEST/ASSERT failed! (%s %s: %lli) ", expression, file, function, line);
        if(message != NULL && strlen(message) != 0)
        {
            log_message(">assert", LOG_FATAL, line, file, function, "message:");
                vlog_message(">>assert", LOG_FATAL, line, file, function, message, args);
        }
    }

#endif