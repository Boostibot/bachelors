#pragma once

#include "lib/defines.h"
#include "lib/string.h"
#include "lib/math.h"
#include "lib/serialize.h"
#include "lib/file.h"

typedef struct Allen_Cahn_Params{
    i32 mesh_size_x;
    i32 mesh_size_y;
    f32 sym_size;

    f32 dt;
    f32 L;
    f32 xi;
    f32 a;
    f32 b;
    f32 alpha;
    f32 beta;
    f32 Tm;
    f32 Tinit;
} Allen_Cahn_Params;

typedef struct Allen_Cahn_Initial_Conditions{
    f32 inside_phi;
    f32 inside_T;

    f32 outside_phi;
    f32 outside_T;

    Vec2 circle_center;
    f32 circle_radius;

    Vec2 square_from;
    Vec2 square_to;

    String_Builder start_snapshot;
} Allen_Cahn_Initial_Conditions;

typedef struct Allen_Cahn_Snapshots{
    String_Builder folder;
    String_Builder prefix;
    f32 every;
    f32 sym_time;
} Allen_Cahn_Snapshots;

typedef struct Allen_Cahn_Config{
    Allocator* allocator;
    String_Builder config_name;
    Allen_Cahn_Params params;
    Allen_Cahn_Snapshots snapshots;
    Allen_Cahn_Initial_Conditions initial_conditions;
} Allen_Cahn_Config;

typedef struct Allen_Cahn_Scale {
    f32 L0;
    f32 Tm;
    f32 Tini;
    f32 c;
    f32 rho;
    f32 lambda;
} Allen_Cahn_Scale;


EXPORT bool serialize_allen_cahn_params(Lpf_Dyn_Entry* entry, Allen_Cahn_Params* val, Read_Or_Write action);
EXPORT bool serialize_allen_cahn_initial_conditions(Lpf_Dyn_Entry* entry, Allen_Cahn_Initial_Conditions* val, Read_Or_Write action);
EXPORT bool serialize_allen_cahn_snapshots(Lpf_Dyn_Entry* entry, Allen_Cahn_Snapshots* val, Read_Or_Write action);
EXPORT bool serialize_allen_cahn_config(Lpf_Dyn_Entry* entry, Allen_Cahn_Config* val, Read_Or_Write action);
EXPORT bool allen_cahn_read_file_config(Allen_Cahn_Config* out_config, const char* config_file_name);

//@TODO: lpf_write vs lpf_write_as_root!
String ephemeral_format_lpf_dyn_entry(Lpf_Dyn_Entry entry)
{
    Lpf_Dyn_Entry root = {0};
    return STRING("");
}

EXPORT bool serialize_allen_cahn_params(Lpf_Dyn_Entry* entry, Allen_Cahn_Params* val, Read_Or_Write action)
{
    bool state = true;

    if(serialize_scope(entry, STRING("Allen_Cahn_Params"), LPF_FLAG_ALIGN_MEMBERS, action) == false)
    {
        state = false; 
        LOG_ERROR("config", "badly formatted Allen_Cahn_Params! The entry is not a scope!");
    }
    else
    {
        const char* generic_message = "Mallformed or missing entry %s of type: %s";
        Vec2 sym_mesh = {(f32) val->mesh_size_x, (f32) val->mesh_size_y};
        if(serialize_vec2(serialize_locate(entry, "sym_mesh", action), &sym_mesh, vec2_of(0), action) == false)
        {
            state = false; 
            LOG_ERROR("config", generic_message, "sym_mesh", "2f (vec2)");
        }
        
        #define SERIALIZE_PARAM_FIELD(type, name)   \
            if(serialize_##type(serialize_locate(entry, #name, action),   &val->name, 0, action) == false) \
            {   \
                state = false; \
                LOG_ERROR("config", generic_message, #name, #type); \
            } \

        //@TODO: serialize flags for this: 1 log errors 2 commented
        //       or maybe have some sort of serialize global state which we could change to set these params
        SERIALIZE_PARAM_FIELD(f32, sym_size);
        serialize_blank(entry, 1, action);
        SERIALIZE_PARAM_FIELD(f32, dt);
        SERIALIZE_PARAM_FIELD(f32, L);
        SERIALIZE_PARAM_FIELD(f32, xi);
        SERIALIZE_PARAM_FIELD(f32, a);
        SERIALIZE_PARAM_FIELD(f32, b);
        SERIALIZE_PARAM_FIELD(f32, beta);
        SERIALIZE_PARAM_FIELD(f32, Tm);

        if(action == SERIALIZE_READ && state)
        {
            val->mesh_size_x = (i32) sym_mesh.x;
            val->mesh_size_y = (i32) sym_mesh.y;
        }
    }
    return state;
}

EXPORT bool serialize_allen_cahn_initial_conditions(Lpf_Dyn_Entry* entry, Allen_Cahn_Initial_Conditions* val, f32 sym_size, bool completeley_optional, Read_Or_Write action)
{
    bool state = serialize_scope(entry, STRING("Allen_Cahn_Params"), LPF_FLAG_ALIGN_MEMBERS, action);
    if(state || completeley_optional)
    {
        serialize_f32(serialize_locate(entry, "inside_phi", action), &val->inside_phi, 0, action);
        serialize_f32(serialize_locate(entry, "inside_T", action),   &val->inside_T, 0, action);
        serialize_f32(serialize_locate(entry, "outside_phi", action), &val->outside_phi, 0, action);
        serialize_f32(serialize_locate(entry, "outside_T", action),   &val->outside_T, 0, action);
    
        serialize_blank(entry, 1, action);
        serialize_vec2(serialize_locate(entry, "circle_center", action), &val->circle_center, vec2(sym_size/2, sym_size/2), action);
        serialize_f32(serialize_locate(entry, "circle_radius", action),  &val->circle_radius, sym_size/8, action);
    
        serialize_blank(entry, 1, action);
        serialize_vec2(serialize_locate(entry, "square_from", action), &val->square_from, vec2_of(0), action);
        serialize_vec2(serialize_locate(entry, "square_to", action), &val->square_to, vec2_of(0), action);
    
        serialize_blank(entry, 1, action);
        if(action == SERIALIZE_READ || val->start_snapshot.size > 0)
            serialize_string(serialize_locate(entry, "start_snapshot", action), &val->start_snapshot, STRING(""), action);
        else
        {
            serialize_comment(entry, STRING("can start from a snapshot"), action);
            serialize_comment(entry, STRING(" start_snapshot : path/to/snapshot.snap"), action);
        }
    }

    return state;
}

EXPORT bool serialize_allen_cahn_snapshots(Lpf_Dyn_Entry* entry, Allen_Cahn_Snapshots* val, f32 dt, String config_name, bool completeley_optional, Read_Or_Write action)
{
    bool state = true;
    state = state && serialize_scope(entry, STRING("Allen_Cahn_Snapshots"), LPF_FLAG_ALIGN_MEMBERS, action);
    if(state || completeley_optional)
    {
        serialize_blank(entry, 1, action);
        serialize_comment(entry, STRING("defaults to snapshots"), action);
        serialize_string(serialize_locate(entry, "folder", action), &val->folder, STRING("snapshots"), action);

        serialize_blank(entry, 1, action);
        serialize_comment(entry, STRING("defaults to name of config scope"), action);
        serialize_string(serialize_locate(entry, "prefix", action), &val->prefix, config_name, action);
        
        if(action == SERIALIZE_WRITE)
        {
            serialize_blank(entry, 1, action);
            serialize_comment(entry, STRING("either sym_time or sym_iters to pause the symulation once the criteria is met"), action);
            serialize_comment(entry, STRING("negative values indicate the symulation should never pause"), action);
            serialize_comment(entry, STRING("if both are defined stops at whichever is sooner"), action);
            serialize_f32(serialize_locate(entry, "sym_time", action), &val->sym_time, 0, action);
            serialize_comment(entry, STRING("sym_iters: 10000"), action);
            
            serialize_blank(entry, 1, action);
            serialize_comment(entry, STRING("either number or every to capture a fixed number of snapshots or with every seecond interval"), action);
            serialize_comment(entry, STRING("negative values indicate no shapshots should be taken"), action);
            serialize_comment(entry, STRING("if both are defined uses whichever is smaller (after conversion)"), action);
            serialize_f32(serialize_locate(entry, "every", action), &val->every, 0, action);
            serialize_comment(entry, STRING("number: 10"), action);
        }
        else
        {
            f32 sym_time = INFINITY;
            i32 sym_iters = INT_MAX;
        
            f32 every = INFINITY;
            i32 number = INT_MAX;

            bool sym_iters_or_time = false
                || serialize_f32(serialize_locate(entry, "sym_time", action), &sym_time, 0, action)
                || serialize_i32(serialize_locate(entry, "sym_iters", action), &sym_iters, 0, action);

            if(sym_iters_or_time)
            {
                f32 sym_iters_converted = sym_iters * dt;
                val->sym_time = MIN(sym_time, sym_iters_converted);
            }
            if(sym_iters_or_time || val->sym_time <= 0)
            {
                val->sym_time = -1;
            }

            bool numver_or_every = false
                || serialize_f32(serialize_locate(entry, "every", action), &every, 0, action)
                || serialize_i32(serialize_locate(entry, "number", action), &number, 0, action);

            if(numver_or_every)
            {
                f32 number_converted = val->sym_time / number;
                val->every = MIN(sym_time, number_converted);
            }
            if(numver_or_every || val->every <= 0)
            {
                val->every = -1;
            }
        }
    }
    return state;
}

EXPORT bool serialize_allen_cahn_config(Lpf_Dyn_Entry* entry, Allen_Cahn_Config* val, String config_name, Read_Or_Write action)
{
    bool state = serialize_allen_cahn_params(serialize_locate(entry, "params", action), &val->params, action);
    if(state)
    {
        serialize_allen_cahn_initial_conditions(serialize_locate(entry, "initial_conditions", action), &val->initial_conditions, val->params.sym_size, true, action);
        serialize_allen_cahn_snapshots(serialize_locate(entry, "snapshots", action), &val->snapshots, val->params.dt, config_name, true, action);
    }

    return state;
}

INTERNAL bool _allen_cahn_read_file_config_recursive(Allen_Cahn_Config* out_config, const char* config_file_name, isize depth)
{
    if(depth > 10)
    {
        LOG_ERROR("config", "Error recursion too deep!");
        return false;
    }

    LOG_INFO("config", "reading config file '%s'", config_file_name);
    log_group_push();
    Allocator* alloc = allocator_get_scratch();
    Lpf_Dyn_Entry root = {0};
    String_Builder select = {alloc};
    String_Builder config_data = {alloc};
    root.allocator = alloc;

    Error error = file_read_entire(string_make(config_file_name), &config_data);

    bool state = false;
    if(error_is_ok(error) == false)
    {
        //TODO: errors as cstrings!
        LOG_ERROR("config", "Error %s while reading config file %s", error_code(error).data, config_file_name);
    }
    else
    {
        Lpf_Format_Options options = lpf_make_default_format_options();
        options.log_errors = true;
        options.skip_inline_comments = true;
        options.skip_comments = true;
        options.skip_blanks = true;
        options.skip_scope_ends = true;

        Lpf_Error lpf_error = lpf_read_custom(string_from_builder(config_data), &root, &options);

        if(lpf_error != LPF_ERROR_NONE)
            LOG_ERROR("config", "Error %s while parsing config file %s", lpf_error_to_string(lpf_error), config_file_name);
        else
        {
            Read_Or_Write r = SERIALIZE_READ;
            
            if(serialize_string(serialize_locate(&root, "select", r), &select, STRING(""), r))
            {
                Lpf_Dyn_Entry* selected = serialize_locate(&root, cstring_escape(select.data), r);
                if(selected == NULL)
                {
                    LOG_INFO("config", "select not found within file %s. Looking for %s as outside file", config_file_name, select.data);
                    state = _allen_cahn_read_file_config_recursive(out_config, select.data, depth + 1);
                }
                else
                {
                    state = serialize_allen_cahn_config(selected, out_config, string_from_builder(select), r);
                }
            }

            if(state == false)
            {
                LOG_INFO("config", "select not found or was invalid. Treating file %s as Allen_Cahn_Config scope", config_file_name);
                state = serialize_allen_cahn_config(&root, out_config, STRING("default"), r);
            }

            if(state == false)
            {
                LOG_ERROR("config", "could not match all config file entries inside %s", config_file_name);
            }
        }

        if(state == false)
        {
            LOG_DEBUG("config", "%s", config_data.data);
        }
    }
    
    if(state)
        LOG_INFO("config", "reading config file '%s' successfull", config_file_name);

    log_group_pop();
    lpf_dyn_entry_deinit(&root);
    array_deinit(&select);
    array_deinit(&config_data);

    return state;
}

EXPORT bool allen_cahn_read_file_config(Allen_Cahn_Config* out_config, const char* config_file_name)
{
    return _allen_cahn_read_file_config_recursive(out_config, config_file_name, 0);
}

f32 allen_cahn_scale_heat(f32 T, Allen_Cahn_Scale scale)
{
    return 1 + (T - scale.Tm)/(scale.Tm - scale.Tini);
}

f32 allen_cahn_scale_latent_heat(f32 L, Allen_Cahn_Scale scale)
{
    return L * scale.rho * scale.c/(scale.Tm - scale.Tini);
}

f32 allen_cahn_scale_pos(f32 x, Allen_Cahn_Scale scale)
{
    return x / scale.L0;
}

f32 allen_cahn_scale_xi(f32 xi, Allen_Cahn_Scale scale)
{
    (void) scale;
    //return xi;
    return xi / scale.L0;
}

f32 allen_cahn_scale_time(f32 t, Allen_Cahn_Scale scale)
{
    const f32 _t0 = (scale.rho*scale.c/scale.lambda)*scale.L0*scale.L0;
    return t / _t0;
}

f32 allen_cahn_scale_beta(f32 beta, Allen_Cahn_Scale scale)
{
    return beta * scale.L0 * (scale.Tm - scale.Tini);
}

f32 allen_cahn_scale_alpha(f32 alpha, Allen_Cahn_Scale scale)
{
    return alpha * scale.lambda / (scale.rho * scale.c);
}