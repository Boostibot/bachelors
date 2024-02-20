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
void draw_sci_texture(unsigned texture, float min, float max);
void draw_sci_cuda_memory(const char* name, int width, int height, float min, float max, bool linear_filtering, const float* cuda_memory);

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

#define GLAD_GL_IMPLEMENTATION
#include "cuda_util.h"
#include "extrenal/glad2/gl.h"

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

void draw_sci_texture(unsigned texture, float min, float max)
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
                if(isnan(val))
                {
                    color = vec4(1, 0, 1, 1); //Bright purple
                }
                else if(val < minVal)
                {
                    float display = (1 - atan(minVal - val)/PI*2)*0.3;
                    color = vec4(display, display, display, 1.0);
                    //Shades from dark gray to black

                }
                else if(val > maxVal)
                {
                    float display = (atan(val - minVal)/PI*2*0.3 + 0.7);
                    color = vec4(display, display, display, 1.0);
                    //Shades from bright gray to white
                }
                else
                {
                    //Spectreum blue -> cyan -> green -> yellow -> red

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
        glBindTexture(GL_TEXTURE_2D, texture);
    
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
void draw_sci_cuda_memory(const char* name, int width, int height, float min, float max, bool linear_filtering, const float* cuda_memory)
{
    enum { 
        MAX_CUDA_GRAPHIC_RESOURCES = 16,
        PIXEL_SIZE = 3
    };
    typedef struct {
        int width;
        int height;
        const char* name;
        GLuint handle;

        void* cpu_memory;
    } Texture;

    static Texture textures[MAX_CUDA_GRAPHIC_RESOURCES] = {0}; 
    static int used_texture_count = 0;

    int resource_index = -1;
    for(int i = 0; i < used_texture_count; i++)
    {
        if(textures[i].width == width && textures[i].height == height && strcmp(textures[i].name, name) == 0)
        {
            resource_index = i;
            break;
        }
    }

    if(resource_index == -1)
    {
        if(used_texture_count >= MAX_CUDA_GRAPHIC_RESOURCES)
        {
            LOG_ERROR("opengl", "too many curently managed cuda resources!");
            return;
        }
        else
        {
            //Create a new texture and register it as cuda resource
            Texture texture = {width, height, name};

            glGenTextures(1, &texture.handle);
            glBindTexture(GL_TEXTURE_2D, texture.handle);

            glTexImage2D(
                GL_TEXTURE_2D,
                0, 
                GL_R32F, // internal format
                width, 
                height, 
                0, 
                GL_RED, // acess format
                GL_FLOAT, //data type
                NULL);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linear_filtering ? GL_LINEAR : GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linear_filtering ? GL_LINEAR : GL_NEAREST);
            glGenerateMipmap(GL_TEXTURE_2D);

            texture.cpu_memory = malloc((size_t) width * (size_t) height * sizeof(float));

            resource_index = used_texture_count++;
            textures[resource_index] = texture;
            // CUDA_TEST(cudaGraphicsGLRegisterImage(&cuda_resources[resource_index], texture.handle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone)); 
        }
    }
            
    //cudaGraphicsGLRegisterImage kept failing with "unknown error" on MX450 on ubuntu. 
    // Because of this we do incredibly inefficient copy to host memory and back to device through opengl call.
    Texture texture = textures[resource_index];
    CUDA_TEST(cudaMemcpy(texture.cpu_memory, cuda_memory, (size_t) width * (size_t) height * sizeof(float), cudaMemcpyDeviceToHost));

    glBindTexture(GL_TEXTURE_2D, texture.handle);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, texture.cpu_memory);
    draw_sci_texture((unsigned) texture.handle, min, max);
}

void gl_init(void* load_function)
{
    int version = gladLoadGL((GLADloadfunc) load_function);
    TEST_MSG(version != 0, "Failed to load opengl with glad");
    LOG_INFO("opengl", "initialized opengl");
    
    gl_debug_output_enable();
}

#endif