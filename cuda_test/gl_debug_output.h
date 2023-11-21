#pragma once

#include "gl.h"
#include "../allen_cahn/lib/log.h"

#define DEBUG_OUTPUT_CHANEL "RENDER"


GLenum _gl_check_error(const char *file, int line)
{
    GLenum errorCode = 0;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        const char* error = "";
        switch (errorCode)
        {
            case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
            case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
            case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
            default:                               error = "UNKNOWN_ERROR"; break;
        }

        LOG_ERROR(DEBUG_OUTPUT_CHANEL, "GL error %s | %s (%d)", error, file, line);
    }
    return errorCode;
}

#define gl_check_error() _gl_check_error(__FILE__, __LINE__) 

void gl_debug_output_func(GLenum source, 
                            GLenum type, 
                            unsigned int id, 
                            GLenum severity, 
                            GLsizei length, 
                            const char *message, 
                            const void *userParam)
{
    // ignore non-significant error/warning codes
    if(id == 131169 || id == 131185 || id == 131218 || id == 131204) return; 

    (void) length;
    (void) userParam;
    
    Log_Type log_type = LOG_TYPE_INFO;
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         log_type = LOG_TYPE_FATAL; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       log_type = LOG_TYPE_ERROR; break;
        case GL_DEBUG_SEVERITY_LOW:          log_type = LOG_TYPE_WARN;  break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: log_type = LOG_TYPE_INFO;  break;
    };

    LOG(DEBUG_OUTPUT_CHANEL, log_type, "GL error (%d): %s", (int) id, message);

    log_group_push();
    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             LOG(DEBUG_OUTPUT_CHANEL, log_type, "Source: API"); break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   LOG(DEBUG_OUTPUT_CHANEL, log_type, "Source: Window System"); break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: LOG(DEBUG_OUTPUT_CHANEL, log_type, "Source: Shader Compiler"); break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     LOG(DEBUG_OUTPUT_CHANEL, log_type, "Source: Third Party"); break;
        case GL_DEBUG_SOURCE_APPLICATION:     LOG(DEBUG_OUTPUT_CHANEL, log_type, "Source: Application"); break;
        case GL_DEBUG_SOURCE_OTHER:           LOG(DEBUG_OUTPUT_CHANEL, log_type, "Source: Other"); break;
    };

    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               LOG(DEBUG_OUTPUT_CHANEL, log_type, "Type: Error"); break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: LOG(DEBUG_OUTPUT_CHANEL, log_type, "Type: Deprecated Behaviour"); break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  LOG(DEBUG_OUTPUT_CHANEL, log_type, "Type: Undefined Behaviour"); break; 
        case GL_DEBUG_TYPE_PORTABILITY:         LOG(DEBUG_OUTPUT_CHANEL, log_type, "Type: Portability"); break;
        case GL_DEBUG_TYPE_PERFORMANCE:         LOG(DEBUG_OUTPUT_CHANEL, log_type, "Type: Performance"); break;
        case GL_DEBUG_TYPE_MARKER:              LOG(DEBUG_OUTPUT_CHANEL, log_type, "Type: Marker"); break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          LOG(DEBUG_OUTPUT_CHANEL, log_type, "Type: Push Group"); break;
        case GL_DEBUG_TYPE_POP_GROUP:           LOG(DEBUG_OUTPUT_CHANEL, log_type, "Type: Pop Group"); break;
        case GL_DEBUG_TYPE_OTHER:               LOG(DEBUG_OUTPUT_CHANEL, log_type, "Type: Other"); break;
    };
    log_group_pop();
}


void gl_debug_output_enable()
{
    int flags = 0; 
    glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
    {
        LOG_INFO(DEBUG_OUTPUT_CHANEL, "Debug info enabled");
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS); 
        glDebugMessageCallback(gl_debug_output_func, NULL);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
    } 
}
