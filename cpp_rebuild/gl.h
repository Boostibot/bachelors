#pragma once

#include "log.h"
#include "assert.h"

typedef enum Pixel_Type {
    PIXEL_TYPE_U8 ,
    PIXEL_TYPE_U16,
    PIXEL_TYPE_U24,
    PIXEL_TYPE_U32,
    PIXEL_TYPE_U64,

    PIXEL_TYPE_I8 ,
    PIXEL_TYPE_I16,
    PIXEL_TYPE_I24,
    PIXEL_TYPE_I32,
    PIXEL_TYPE_I64,
    
    PIXEL_TYPE_F8 ,
    PIXEL_TYPE_F16,
    PIXEL_TYPE_F32,
    PIXEL_TYPE_F64,

    PIXEL_TYPE_INVALID = INT_MIN
} Pixel_Type;

typedef struct GL_Pixel_Format {
    unsigned channel_type;     //GL_FLOAT, GL_UNSIGNED_BYTE, ...
    unsigned access_format;    //GL_RGBA, GL_RED, ...
    unsigned internal_format;  //GL_RGBA16I, GL_R8, GL_RGBA16F...
} GL_Pixel_Format;

typedef struct GL_Texture {
    unsigned handle;    
    GL_Pixel_Format format;
    Pixel_Type pixel_type;
    int pixel_size;

    int width;
    int heigth;
} GL_Texture;

void gl_init(void* load_function);
GL_Texture gl_texture_make(size_t width, size_t heigth, Pixel_Type format, size_t channels, const void* data_or_null);
void draw_sci_texture(GL_Texture texture, float min, float max);
void draw_sci_cuda_memory(GL_Texture texture, float min, float max, const float* cuda_memory);

#ifdef NO_GL

void draw_sci_texture(GL_Texture texture, float min, float max)
{
    (void) texture;
    (void) min;
    (void) max;
}
GL_Texture gl_texture_make(size_t width, size_t heigth, Pixel_Type format, size_t channels, const void* data_or_null)
{
    GL_Texture tex = {0};
    tex.width = width;
    tex.heigth = heigth;

    (void) format;
    (void) channels;
    (void) data_or_null;

    return tex;
}


void draw_sci_cuda_memory(GL_Texture texture, float min, float max, const float* cuda_memory)
{
    (void) texture;
    (void) min;
    (void) max;
    (void) cuda_memory;
}

#else


//@TODO: remove this pixel type madness!

#define GLAD_GL_IMPLEMENTATION
#include "cuda_util.h"
#include "extrenal/include/glad2/gl.h"

int pixel_type_size(Pixel_Type pixel_type)
{
    switch(pixel_type)
    {
        case PIXEL_TYPE_U8:  return 1;
        case PIXEL_TYPE_U16: return 2;
        case PIXEL_TYPE_U24: return 3;
        case PIXEL_TYPE_U32: return 4;
        case PIXEL_TYPE_U64: return 8;
        
        case PIXEL_TYPE_I8:  return 1;
        case PIXEL_TYPE_I16: return 2;
        case PIXEL_TYPE_I24: return 3;
        case PIXEL_TYPE_I32: return 4;
        case PIXEL_TYPE_I64: return 8;

        case PIXEL_TYPE_F8:  return 1;
        case PIXEL_TYPE_F16: return 2;
        case PIXEL_TYPE_F32: return 4;
        case PIXEL_TYPE_F64: return 8;
        
        case PIXEL_TYPE_INVALID: 
        default: {
            if(pixel_type > 0)
                return (int) pixel_type;
            else
                return 1;
        }
    }
}

//If fails sets all members to 0
GL_Pixel_Format gl_pixel_format_from_pixel_type(Pixel_Type pixel_format, size_t channels)
{
    GL_Pixel_Format error_format = {0};
    
    GL_Pixel_Format out = {0};
    int channel_size = pixel_type_size(pixel_format);
    
    switch(channels)
    {
        case 1: out.access_format = GL_RED; break;
        case 2: out.access_format = GL_RG; break;
        case 3: out.access_format = GL_RGB; break;
        case 4: out.access_format = GL_RGBA; break;
        
        default: return error_format;
    }

    #define PIXEL_TYPE_CASE(PIXEL_TYPE, GL_CHANNEL_TYPE, POSTFIX) \
        case PIXEL_TYPE: { \
            out.channel_type = GL_CHANNEL_TYPE; \
            switch(channels) \
            { \
                case 1: out.internal_format = GL_R ## POSTFIX; break; \
                case 2: out.internal_format = GL_RG ## POSTFIX; break; \
                case 3: out.internal_format = GL_RGB ## POSTFIX; break; \
                case 4: out.internal_format = GL_RGBA ## POSTFIX; break; \
            } \
        } break \


    switch(pixel_format)
    {
        PIXEL_TYPE_CASE(PIXEL_TYPE_U8, GL_UNSIGNED_BYTE, 8);
        PIXEL_TYPE_CASE(PIXEL_TYPE_U16, GL_UNSIGNED_SHORT, 16);
        PIXEL_TYPE_CASE(PIXEL_TYPE_U32, GL_UNSIGNED_INT, 32UI);
        case PIXEL_TYPE_U24: return error_format;
        case PIXEL_TYPE_U64: return error_format;

        PIXEL_TYPE_CASE(PIXEL_TYPE_I8, GL_BYTE, 8I);
        PIXEL_TYPE_CASE(PIXEL_TYPE_I16, GL_SHORT, 16I);
        PIXEL_TYPE_CASE(PIXEL_TYPE_I32, GL_INT, 32I);
        case PIXEL_TYPE_I24: return error_format;
        case PIXEL_TYPE_I64: return error_format;
        
        PIXEL_TYPE_CASE(PIXEL_TYPE_F16, GL_HALF_FLOAT, 16F);
        PIXEL_TYPE_CASE(PIXEL_TYPE_F32, GL_FLOAT, 32F);
        case PIXEL_TYPE_F8: return error_format;
        case PIXEL_TYPE_F64: return error_format;
        
        case PIXEL_TYPE_INVALID: 
        default: return error_format;
    }

    #undef PIXEL_TYPE_CASE
    return out;
}

GL_Pixel_Format gl_pixel_format_from_pixel_type_size(Pixel_Type pixel_format, size_t pixel_size)
{
    return gl_pixel_format_from_pixel_type(pixel_format, pixel_size / pixel_type_size(pixel_format));
}

//returns PIXEL_TYPE_INVALID and sets channels to 0 if couldnt match
Pixel_Type pixel_type_from_gl_internal_format(unsigned internal_format, int* channels)
{
    ASSERT(channels != 0);
    #define GL_TYPE_CASE1(P1, PIXEL_TYPE) \
        case GL_R ## P1: \
            *channels = 1; return PIXEL_TYPE; \
        case GL_RG ## P1: \
            *channels = 2; return PIXEL_TYPE; \
        case GL_RGB ## P1: \
            *channels = 3; return PIXEL_TYPE; \
        case GL_RGBA ## P1: \
            *channels = 4; return PIXEL_TYPE; \

    #define GL_TYPE_CASE2(P1, P2, PIXEL_TYPE) \
        case GL_R ## P1: \
        case GL_R ## P2: \
            *channels = 1; return PIXEL_TYPE; \
        case GL_RG ## P1: \
        case GL_RG ## P2: \
            *channels = 2; return PIXEL_TYPE; \
        case GL_RGB ## P1: \
        case GL_RGB ## P2: \
            *channels = 3; return PIXEL_TYPE; \
        case GL_RGBA ## P1: \
        case GL_RGBA ## P2: \
            *channels = 4; return PIXEL_TYPE; \

    switch(internal_format)
    {
        GL_TYPE_CASE2(8, 8UI, PIXEL_TYPE_U8);
        GL_TYPE_CASE1(8I, PIXEL_TYPE_U8);
        
        GL_TYPE_CASE2(16, 16UI, PIXEL_TYPE_U16);
        GL_TYPE_CASE1(16I, PIXEL_TYPE_U16);

        GL_TYPE_CASE1(32UI, PIXEL_TYPE_U32);
        GL_TYPE_CASE1(32I, PIXEL_TYPE_I32);

        GL_TYPE_CASE1(16F, PIXEL_TYPE_F16);
        GL_TYPE_CASE1(32F, PIXEL_TYPE_F32);

        default: 
            *channels = 0;
            return PIXEL_TYPE_INVALID;
        break;
    }

    #undef GL_TYPE_CASE1
    #undef GL_TYPE_CASE2
}

//returns PIXEL_TYPE_INVALID on failiure
Pixel_Type pixel_type_from_gl_access_format(GL_Pixel_Format gl_format)
{
    switch(gl_format.channel_type)
    {
        case GL_UNSIGNED_BYTE: return PIXEL_TYPE_U8;
        case GL_UNSIGNED_SHORT: return PIXEL_TYPE_U16;
        case GL_UNSIGNED_INT: return PIXEL_TYPE_U32;

        case GL_BYTE: return PIXEL_TYPE_I8;
        case GL_SHORT: return PIXEL_TYPE_I16;
        case GL_INT: return PIXEL_TYPE_I32;

        case GL_HALF_FLOAT: return PIXEL_TYPE_F16;
        case GL_FLOAT: return PIXEL_TYPE_F32;

        default: return PIXEL_TYPE_INVALID;
    }
}

//Outputs 0 to channels if couldnt match
Pixel_Type pixel_type_from_gl_pixel_format(GL_Pixel_Format gl_format, int* channels)
{
    return pixel_type_from_gl_internal_format(gl_format.internal_format, channels);
}


const char* gl_translate_error(GLenum code)
{
    switch (code)
    {
        case GL_INVALID_ENUM:                  return "INVALID_ENUM";
        case GL_INVALID_VALUE:                 return "INVALID_VALUE";
        case GL_INVALID_OPERATION:             return "INVALID_OPERATION";
        case GL_STACK_OVERFLOW:                return "STACK_OVERFLOW";
        case GL_STACK_UNDERFLOW:               return "STACK_UNDERFLOW";
        case GL_OUT_OF_MEMORY:                 return "OUT_OF_MEMORY";
        case GL_INVALID_FRAMEBUFFER_OPERATION: return "INVALID_FRAMEBUFFER_OPERATION";
        default:                               return "UNKNOWN_ERROR";
    }
}

GLenum _gl_check_error(const char *file, int line)
{
    GLenum errorCode = 0;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        const char* error = gl_translate_error(errorCode);
        LOG_ERROR("opengl", "GL error %s | %s (%d)", error, file, line);
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
    
    Log_Type log_type = LOG_INFO;
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         log_type = LOG_FATAL; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       log_type = LOG_ERROR; break;
        case GL_DEBUG_SEVERITY_LOW:          log_type = LOG_WARN;  break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: log_type = LOG_INFO;  break;
    };

    LOG("opengl", log_type, "GL error (%d): %s", (int) id, message);

    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             LOG(">opengl", log_type, "Source: API"); break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   LOG(">opengl", log_type, "Source: Window System"); break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: LOG(">opengl", log_type, "Source: Shader Compiler"); break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     LOG(">opengl", log_type, "Source: Third Party"); break;
        case GL_DEBUG_SOURCE_APPLICATION:     LOG(">opengl", log_type, "Source: Application"); break;
        case GL_DEBUG_SOURCE_OTHER:           LOG(">opengl", log_type, "Source: Other"); break;
    };

    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               LOG(">opengl", log_type, "Type: Error"); break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: LOG(">opengl", log_type, "Type: Deprecated Behaviour"); break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  LOG(">opengl", log_type, "Type: Undefined Behaviour"); break; 
        case GL_DEBUG_TYPE_PORTABILITY:         LOG(">opengl", log_type, "Type: Portability"); break;
        case GL_DEBUG_TYPE_PERFORMANCE:         LOG(">opengl", log_type, "Type: Performance"); break;
        case GL_DEBUG_TYPE_MARKER:              LOG(">opengl", log_type, "Type: Marker"); break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          LOG(">opengl", log_type, "Type: Push Group"); break;
        case GL_DEBUG_TYPE_POP_GROUP:           LOG(">opengl", log_type, "Type: Pop Group"); break;
        case GL_DEBUG_TYPE_OTHER:               LOG(">opengl", log_type, "Type: Other"); break;
    };
}


static void gl_post_call_gl_callback(void *ret, const char *name, GLADapiproc apiproc, int len_args, ...) {
    GLenum error_code;

    (void) ret;
    (void) apiproc;
    (void) len_args;

    error_code = glad_glGetError();

    if (error_code != GL_NO_ERROR) 
        LOG_ERROR("opengl", "error %s in %s!", gl_translate_error(error_code), name);
}

void gl_debug_output_enable()
{
    int flags = 0; 
    glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
    {
        LOG_INFO("opengl", "Debug info enabled");
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS); 
        glDebugMessageCallback(gl_debug_output_func, NULL);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
    } 

    gladSetGLPostCallback(gl_post_call_gl_callback);
    gladInstallGLDebug();
}

unsigned compile_shader(const char* vertex_shader_source, const char* frag_shader_source)
{
    unsigned vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertexShader);

    unsigned fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &frag_shader_source, NULL);
    glCompileShader(fragmentShader);

    unsigned shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    int vertex_success = false; 
    int fragment_success = false;
    int link_success = false;
    char error_msg[512] = {0};

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vertex_success);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &fragment_success);
    if(!vertex_success)
    {
        glGetShaderInfoLog(vertexShader, sizeof error_msg, NULL, error_msg);
        LOG_ERROR("opengl", "Error compiling vertex shader:");
        LOG_ERROR(">opengl", "%s", error_msg);
    }
       
    if(!fragment_success)
    {
        glGetShaderInfoLog(fragmentShader, sizeof error_msg, NULL, error_msg);
        LOG_ERROR("opengl", "Error compiling fragment shader:");
        LOG_ERROR(">opengl", "%s", error_msg);
    }

    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &link_success);
    if(!link_success)
    {
        glGetProgramInfoLog(shaderProgram, sizeof error_msg, NULL, error_msg);
        LOG_ERROR("opengl", "Error linkin shader program:");
        LOG_ERROR(">opengl", "%s", error_msg);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    if(!vertex_success || !fragment_success || !link_success)
    {
        glDeleteProgram(shaderProgram);
        return 0;
    }
    else
        return shaderProgram;
}

void render_screen_quad()
{
    static unsigned quadVAO = 0;
    static unsigned quadVBO = 0;
	if (quadVAO == 0)
	{
		float quadVertices[] = {
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
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	}

	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

GL_Texture gl_texture_make(size_t width, size_t heigth, Pixel_Type format, size_t channels, const void* data_or_null)
{
    GL_Pixel_Format pixel_format = gl_pixel_format_from_pixel_type(format, channels);
    ASSERT(pixel_format.access_format != 0);
    
    GL_Texture tex = {0};
	glGenTextures(1, &tex.handle);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex.handle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, pixel_format.internal_format, (unsigned) width, (unsigned) heigth, 0, pixel_format.access_format, pixel_format.channel_type, data_or_null);
    glGenerateMipmap(GL_TEXTURE_2D);

    tex.format = pixel_format;
    tex.width = (int) width;
    tex.heigth = (int) heigth;
    tex.pixel_type = format;
    tex.pixel_size = (int) channels * pixel_type_size(format);

	glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

void draw_sci_texture(GL_Texture texture, float min, float max)
{
    #define MIN_LOCATION 1
    #define MAX_LOCATION 2
    #define TEX_LOCATION 3
    #define TEXTURE_BINDING 0

    static bool shader_error = false;
    static unsigned sci_shader = 0;
    if(sci_shader == 0 && shader_error == false)
    {
        const char* frag_shader_source = R"SHADER(
            #version 430 core
            #extension GL_ARB_explicit_uniform_location : require

            layout (location = 1) uniform float _min; //MIN_LOCATION
            layout (location = 2) uniform float _max; //MAX_LOCATION
            layout (location = 3) uniform sampler2D tex; //TEX_LOCATION
            
            out vec4 color;
            in vec2 uv;

            #define PI 3.14159265359

            void main()
            {
                float minVal = _min;
                float maxVal = _max;

                vec2 reverse_uv = vec2(uv.x, uv.y);
                vec3 texCol = texture(tex, reverse_uv).rgb;      
                float val = texCol.r;
                if(val < minVal)
                {
                    float display = (1 - atan(minVal - val)/PI*2)/2;
                    color = vec4(display, display, display, 1.0);
                    color = vec4(0, 0, 0, 1.0);
                }
                else if(val > maxVal)
                {
                    float display = (atan(val - minVal)/PI*2)/2 + 0.5;
                    color = vec4(display, display, display, 1.0);
                    color = vec4(1, 1, 1, 1.0);
                }
                else
                {
                    val = min(max(val, minVal), maxVal- 0.0001);
                    float d = maxVal - minVal;
                    val = d == 0.0 ? 0.5 : (val - minVal) / d;
                    float m = 0.25;
                    float num = floor(val / m);
                    float s = (val - num * m) / m;
                    float r = 0, g = 0, b = 0;

                    switch (int(num)) {
                        case 0 : r = 0.0; g = s; b = 1.0; break;
                        case 1 : r = 0.0; g = 1.0; b = 1.0-s; break;
                        case 2 : r = s; g = 1.0; b = 0.0; break;
                        case 3 : r = 1.0; g = 1.0 - s; b = 0.0; break;
                    }

                    //color = vec4(val, val, val, 1.0);

                    color = vec4(r, g, b, 1.0);
                }
            }
        )SHADER";

        const char* vertex_shader_source = R"SHADER(
            #version 430 core

            layout (location = 0) in vec3 a_pos;
            layout (location = 1) in vec2 a_uv;

            out vec2 uv;

            void main()
            {
                uv = a_uv;
                gl_Position = vec4(a_pos, 1.0);
            }
        )SHADER";

        sci_shader = compile_shader(vertex_shader_source, frag_shader_source);
        shader_error = sci_shader == 0;
    }
    
    if(shader_error == false)
    {
	    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0 + TEXTURE_BINDING);
        glBindTexture(GL_TEXTURE_2D, texture.handle);
    
        glUseProgram(sci_shader);
        glUniform1f(MIN_LOCATION, min);
        glUniform1f(MAX_LOCATION, max);
        glUniform1i(TEX_LOCATION, TEXTURE_BINDING);

	    render_screen_quad();
    }

    
    #undef MIN_LOCATION
    #undef MAX_LOCATION
    #undef TEX_LOCATION
    #undef TEXTURE_BINDING
}


#include "cuda_util.h"
#include <cuda_gl_interop.h>
void draw_sci_cuda_memory(GL_Texture texture, float min, float max, const float* cuda_memory)
{
    #define MAX_CUDA_GRAPHIC_RESOURCES 16
    static struct cudaGraphicsResource *cuda_resources[MAX_CUDA_GRAPHIC_RESOURCES] = {0}; 
    static unsigned used_resource_handles[MAX_CUDA_GRAPHIC_RESOURCES] = {0}; 
    static int used_resource_count = 0;

    size_t resource_index = -1;
    for(size_t i = 0; i < used_resource_count; i++)
    {
        if(used_resource_handles[i] == texture.handle)
        {
            resource_index = i;
            break;
        }
    }

    if(resource_index == -1)
    {
        if(used_resource_count >= MAX_CUDA_GRAPHIC_RESOURCES)
        {
            LOG_ERROR("opengl", "too many curently managed cuda resources!");
            return;
        }

        resource_index = used_resource_count++;
        CUDA_TEST(cudaGraphicsGLRegisterImage(&cuda_resources[resource_index], texture.handle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
        used_resource_handles[resource_index] = texture.handle;
    }
            
    struct cudaGraphicsResource *cuda_resource = cuda_resources[resource_index]; 

    CUDA_TEST(cudaGraphicsMapResources(1, &cuda_resource, 0));
    
    #if 1
        //for texture objects
        cudaArray_t mapped_array = {0};
        size_t texture_size = texture.width * texture.heigth * texture.pixel_size;
        void* dev_ptr = NULL;
        size_t mapped_size = 0;
        //CUDA_TEST(cudaGraphicsResourceGetMappedPointer(&dev_ptr, &mapped_size, cuda_resource));
        //ASSERT(mapped_size >= texture_size);
        //CUDA_TEST(cudaMemcpy(dev_ptr, cuda_memory, texture_size, cudaMemcpyDeviceToDevice));

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
    draw_sci_texture(texture, min, max);
}


void gl_init(void* load_function)
{
    int version = gladLoadGL((GLADloadfunc) load_function);
    TEST_MSG(version != 0, "Failed to load opengl with glad");
    LOG_INFO("opengl", "initialized opengl");
    
    gl_debug_output_enable();
}

#endif