#pragma once

#include "lib/file.h"
#include "math.h"
#include "gl.h"

#define SHADER_UTIL_CHANEL "ASSET"

String_Builder render_shader_source_prepend(String data, String prepend, Allocator* allocator)
{
    String_Builder composed = {allocator};
    if(prepend.size == 0)
    {
        builder_append(&composed, data);
    }
    else
    {
        isize version_i = string_find_first(data, STRING("#version"), 0);
        isize after_version = 0;
        //if it contains no version directive start at start
        if(version_i == -1)
            after_version = 0;
        //if it does start just after the next line break
        else
        {
            ASSERT(version_i <= data.size);
            after_version = string_find_first_char(data, '\n', version_i);
            if(after_version == -1)
                after_version = data.size - 1;

            after_version += 1;
        }
            
        ASSERT(after_version <= data.size);

        String before_insertion = string_head(data, after_version);
        String after_insertion = string_tail(data, after_version);
        builder_append(&composed, before_insertion);
        builder_append(&composed, prepend);
        builder_append(&composed, STRING("\n"));
        builder_append(&composed, after_insertion);
    }

    return composed;
}

typedef struct Render_Shader
{
    GLuint shader;
    String_Builder name;
} Render_Shader;

void render_shader_deinit(Render_Shader* shader)
{
    if(shader->shader != 0)
    {
        glUseProgram(0);
        glDeleteProgram(shader->shader);
    }
    array_deinit(&shader->name);
    memset(shader, 0, sizeof(*shader));
}

String translate_shader_error(u32 code, void* context)
{
    (void) context;
    if(code == 1)
        return STRING("shader compilation failed");
    else
        return STRING("<unexpected shader error; this is a result of a bug>");
}

Error render_shader_init(Render_Shader* shader, const char* vertex, const char* fragment, const char* geometry, String name, String_Builder* error, isize* error_at_stage)
{
    PERF_COUNTER_START(c);
    static u32 shader_error_module = 0;
    if(shader_error_module == 0)
        shader_error_module = error_system_register_module(translate_shader_error, STRING("shader_util.h"), NULL);

    render_shader_deinit(shader);
    ASSERT(vertex != NULL && fragment != NULL);

    GLuint shader_program = 0;
    GLuint shader_types[3] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_GEOMETRY_SHADER};
    GLuint shaders[3] = {0};
    const char* shader_sources[3] = {vertex, fragment, geometry};
    int  success = true;
    bool has_geometry = geometry != NULL && strlen(geometry) != 0;

    for(isize i = 0; i < 3 && success; i++)
    {
        //geometry shader doesnt have to exist
        if(shader_types[i] == GL_GEOMETRY_SHADER && has_geometry == false)
            break;

        shaders[i] = glCreateShader(shader_types[i]);
        glShaderSource(shaders[i], 1, &shader_sources[i], NULL);
        glCompileShader(shaders[i]);

        //Check for errors
        glGetShaderiv(shaders[i], GL_COMPILE_STATUS, &success);
        if(!success)
        {
            if(error != NULL)
            {
                array_resize(error, 1024);
                glGetShaderInfoLog(shaders[i], (GLsizei) error->size, NULL, error->data);
                array_resize(error, strlen(error->data));
            }
            if(error_at_stage)
                *error_at_stage = i + 1;
            break;
        }
    }
        
    if(success)
    {
        shader_program = glCreateProgram();
        glAttachShader(shader_program, shaders[0]);
        glAttachShader(shader_program, shaders[1]);
        if(has_geometry)
            glAttachShader(shader_program, shaders[2]);
            
        glLinkProgram(shader_program);
        glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
        if(!success) 
        {
            if(error != NULL)
            {
                array_resize(error, 1024);
                glGetProgramInfoLog(shader_program, (GLsizei) error->size, NULL, error->data);
                array_resize(error, strlen(error->data));
            }
            if(error_at_stage)
                *error_at_stage = 4;
        }
    }

    glDeleteShader(shaders[0]);
    glDeleteShader(shaders[1]);
    glDeleteShader(shaders[2]);

    if(success == false)
    {
        glDeleteProgram(shader_program);
        shader_program = 0;
    }

    shader->shader = shader_program;
    shader->name = builder_from_string(name, NULL);
    
    PERF_COUNTER_END(c);

    if(success == false)
        return error_make(shader_error_module, 1);
    else
        return error_make(shader_error_module, 0);
}

Error render_shader_init_from_disk_custom(Render_Shader* shader, String vertex_path, String fragment_path, String geometry_path, String prepend, String name, String_Builder* error)
{
    PERF_COUNTER_START(c);
    Allocator* scratch = allocator_get_scratch();
    String_Builder temp = {scratch};

    Error vertex_error = file_read_entire(vertex_path, &temp);
    String_Builder vertex_source = render_shader_source_prepend(string_from_builder(temp), prepend, scratch);

    Error fragment_error = file_read_entire(fragment_path, &temp);
    String_Builder fragment_source = render_shader_source_prepend(string_from_builder(temp), prepend, scratch);

    String_Builder geometry_source = {0};
    Error geomtery_error = {0};
    if(geometry_path.size > 0)
    {
        geomtery_error = file_read_entire(geometry_path, &temp);
        geometry_source = render_shader_source_prepend(string_from_builder(temp), prepend, scratch);
    }

    Error compile_error = ERROR_OR(vertex_error) ERROR_OR(fragment_error) geomtery_error;
    isize error_at_compilation_stage = 0; 
    if(error_is_ok(compile_error))
    {
        //Else tries to build the program
        // if an error ocurs here it gets written to error
        // and shader_program = 0. This means we dont have to do anything
        compile_error = render_shader_init(shader,
            cstring_from_builder(vertex_source),
            cstring_from_builder(fragment_source),
            cstring_from_builder(geometry_source),
            name,
            &temp, &error_at_compilation_stage);
    }
        
    //Error reporting ... that is longer than the function itself :/
    if(!error_is_ok(compile_error))
    {
        #if defined(DO_LOG) && defined(DO_LOG_ERROR)
            String_Builder vertex_error_string = {scratch};
            String_Builder fragment_error_string = {scratch};
            String_Builder geometry_error_string = {scratch};

            if(!error_is_ok(vertex_error))
                error_code_into(&vertex_error_string, vertex_error);
            if(!error_is_ok(fragment_error))
                error_code_into(&fragment_error_string, fragment_error);
            if(!error_is_ok(geomtery_error))
                error_code_into(&geometry_error_string, geomtery_error);

            //if has error print the translated version
            LOG_ERROR(SHADER_UTIL_CHANEL, "shader_load() failed! files:\n");
            log_group_push();
                LOG_ERROR(SHADER_UTIL_CHANEL, "vertex:   \"" STRING_FMT "\"", STRING_PRINT(vertex_path));
                    log_group_push();
                    if(vertex_error_string.size > 0)
                        LOG_ERROR(SHADER_UTIL_CHANEL, STRING_FMT, STRING_PRINT(vertex_error_string));
                    if(error_at_compilation_stage == 1)
                        LOG_ERROR(SHADER_UTIL_CHANEL, STRING_FMT, STRING_PRINT(temp));
                log_group_pop();

                LOG_ERROR(SHADER_UTIL_CHANEL, "fragment: \"" STRING_FMT "\"", STRING_PRINT(fragment_path));
                log_group_push();
                    if(fragment_error_string.size > 0)
                        LOG_ERROR(SHADER_UTIL_CHANEL, STRING_FMT, STRING_PRINT(fragment_error_string));
                    if(error_at_compilation_stage == 2)
                        LOG_ERROR(SHADER_UTIL_CHANEL, STRING_FMT, STRING_PRINT(temp));
                log_group_pop();

                if(geometry_path.size > 0)
                {
                    LOG_ERROR(SHADER_UTIL_CHANEL, "geometry: \"" STRING_FMT "\"", STRING_PRINT(geometry_path));
                    log_group_push();
                        if(geometry_error_string.size > 0)
                            LOG_ERROR(SHADER_UTIL_CHANEL, STRING_FMT, STRING_PRINT(geometry_error_string));
                        if(error_at_compilation_stage == 3)
                            LOG_ERROR(SHADER_UTIL_CHANEL, STRING_FMT, STRING_PRINT(temp));
                    log_group_pop();
                }
            
                if(prepend.size > 0)
                {
                    LOG_ERROR(SHADER_UTIL_CHANEL, "prepend:");
                    log_group_push();
                        LOG_ERROR(SHADER_UTIL_CHANEL, STRING_FMT, STRING_PRINT(prepend));
                    log_group_pop();
                }

                if(error_at_compilation_stage == 4)
                {
                    LOG_ERROR(SHADER_UTIL_CHANEL, "link:");
                    log_group_push();
                        LOG_ERROR(SHADER_UTIL_CHANEL, STRING_FMT, STRING_PRINT(temp));
                    log_group_pop();
                }
            log_group_pop();

            if(error)
            {
                if(!error_is_ok(vertex_error))
                    array_copy(error, vertex_error_string);
                else if(!error_is_ok(fragment_error))
                    array_copy(error, fragment_error_string);
                else if(!error_is_ok(geomtery_error))
                    array_copy(error, geometry_error_string);
                else if(!error_is_ok(compile_error))
                    array_copy(error, temp);
            }

            array_deinit(&vertex_error_string);
            array_deinit(&fragment_error_string);
            array_deinit(&geometry_error_string);
        #endif
    }

    array_deinit(&vertex_source);
    array_deinit(&fragment_source);
    array_deinit(&geometry_source);
    array_deinit(&geometry_source);
    array_deinit(&temp);
    
    PERF_COUNTER_END(c);
    return compile_error;
}
    
Error render_shader_init_from_disk(Render_Shader* shader, String vertex_path, String fragment_path, String name)
{
    //String_Builder name = {0};
    //array_init_backed(&name, allocator_get_scratch(), 256);
    //format_into(&name, "vert: " STRING_FMT " frag: " STRING_FMT, STRING_PRINT(vertex_path), STRING_PRINT(fragment_path));

    return render_shader_init_from_disk_custom(shader, vertex_path, fragment_path, STRING(""), STRING(""), name, NULL);
}

void render_shader_use(const Render_Shader* shader)
{
    ASSERT(shader->shader != 0);
    glUseProgram(shader->shader);
}

void render_shader_unuse(const Render_Shader* shader)
{
    (void) shader;
    glUseProgram(0);
}

GLint render_shader_get_uniform_location(const Render_Shader* shader, const char* uniform)
{
    PERF_COUNTER_START(c);
    GLint location = glGetUniformLocation(shader->shader, uniform);
    if(location == -1)
    {
        //if the shader uniform is not found we log only 
        //each 2^n th error message from this shader and uniform
        //(unless shader name, uniform, and shader program have a hash colision
        // but thats astronomically improbable)

        String uniform_str = string_make(uniform);
        u64 name_hash = hash64_murmur(shader->name.data, shader->name.size, 0);
        u64 uniform_hash = hash64_murmur(uniform_str.data, uniform_str.size, 0);
        u64 shader_hash = hash64(shader->shader);
        u64 final_hash = name_hash ^ uniform_hash ^ shader_hash;

        static Hash_Index64 error_hash_index = {0};
        if(error_hash_index.allocator == NULL)
        {
            hash_index64_init(&error_hash_index, allocator_get_static());
            hash_index64_reserve(&error_hash_index, 32);
        }

        isize found = hash_index64_find_or_insert(&error_hash_index, final_hash, 0);
        u64* error_count = &error_hash_index.entries[found].value;
        if(is_power_of_two_or_zero((*error_count)++))
            LOG_ERROR("RENDER", "failed to find uniform %-25s shader: " STRING_FMT, uniform, STRING_PRINT(shader->name));
    }
    PERF_COUNTER_END(c);

    return location;
}

bool render_shader_set_i32(const Render_Shader* shader, const char* name, i32 val)
{
    GLint location = render_shader_get_uniform_location(shader, name);
    if(location == -1) 
        return false;

    glUniform1i(location, (GLint) val);
    return true;
}
    
bool render_shader_set_f32(const Render_Shader* shader, const char* name, f32 val)
{
    GLint location = render_shader_get_uniform_location(shader, name);
    if(location == -1)
        return false;

    glUniform1f(location, (GLfloat) val);
    return true;
}

bool render_shader_set_vec3(const Render_Shader* shader, const char* name, Vec3 val)
{
    GLint location = render_shader_get_uniform_location(shader, name);
    if(location == -1)
        return false;

    glUniform3fv(location, 1, AS_FLOATS(val));
    return true;
}

bool render_shader_set_mat4(const Render_Shader* shader, const char* name, Mat4 val)
{
    GLint location = render_shader_get_uniform_location(shader, name);
    if(location == -1)
        return false;

    glUniformMatrix4fv(location, 1, GL_FALSE, AS_FLOATS(val));
    return true;
}
    
bool render_shader_set_mat3(const Render_Shader* shader, const char* name, Mat3 val)
{
    GLint location = render_shader_get_uniform_location(shader, name);
    if(location == -1)
        return false;

    glUniformMatrix3fv(location, 1, GL_FALSE, AS_FLOATS(val));
    return true;
}
