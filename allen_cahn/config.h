#pragma once
#include "kernel.h"
#include <string>

typedef struct Vec2{
    Real x;
    Real y;
} Vec2;

typedef struct Allen_Cahn_Initial_Conditions{
    Real inside_phi;
    Real inside_T;

    Real outside_phi;
    Real outside_T;

    Vec2 circle_center;
    Real circle_inner_radius;
    Real circle_outer_radius;

    Vec2 square_from;
    Vec2 square_to;
} Allen_Cahn_Initial_Conditions;

typedef struct Allen_Cahn_Snapshots{
    double snapshot_every;
    int    snapshot_times;
    bool snapshot_initial_conditions;
    bool has_every;
    bool has_times;

    std::string folder;
    std::string prefix;
    std::string postfix;
} Allen_Cahn_Snapshots;

typedef struct Allen_Cahn_Config{
    Allen_Cahn_Params params;
    Allen_Cahn_Snapshots snapshots;
    Allen_Cahn_Initial_Conditions initial_conditions;

    double stop_after;
    bool interactive_mode;
    bool linear_filtering;
    Real display_min;
    Real display_max;

    Solver_Type solver;
} Allen_Cahn_Config;

enum Skip_Options {
    SKIP_NORMAL,
    SKIP_INVERSE,
};

size_t skip_set(size_t from, std::string_view source, std::string_view char_set, Skip_Options options = SKIP_NORMAL)
{
    for(size_t i = from; i < source.size(); i++)
    {
        char c = source[i];
        bool is_in_set = char_set.find(c, 0) != std::string::npos;
        if(is_in_set && options == SKIP_INVERSE)
            return i;
        if(is_in_set == false && options == SKIP_NORMAL)
            return i;
    }

    return source.size();
}

size_t skip_set_reverse(size_t from, size_t to, std::string_view source, std::string_view char_set, Skip_Options options = SKIP_NORMAL)
{
    for(size_t i = from; i-- > to;)
    {
        char c = source[i];
        bool is_in_set = char_set.find(c, 0) != std::string::npos;
        if(is_in_set && options == SKIP_INVERSE)
            return i;
        if(is_in_set == false && options == SKIP_NORMAL)
            return i;
    }

    return to;
}

std::string_view get_line(size_t* positon, std::string_view source)
{
    if(*positon >= source.size())
        return "";

    size_t line_start = *positon;
    size_t line_end = source.find('\n', *positon);
    if(line_end == std::string::npos)
        line_end = source.size();

    std::string_view out = source.substr(line_start, line_end - line_start);
    *positon = line_end + 1;
    return out;
}

#include "log.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

bool file_read_entire(const char* path, std::string* into)
{
    std::ifstream t(path);
    std::stringstream buffer;
    buffer << t.rdbuf();
    if(into)
        *into = buffer.str();

    return !t.bad();
}

#define WHITESPACE " \t\n\r\v\f"
#define MARKERS    "=:,"

enum Key_Value_Type {
    KEY_VALUE_BLANK,
    KEY_VALUE_COMMENT,
    KEY_VALUE_SECTION,
    KEY_VALUE_ENTRY,
    KEY_VALUE_ERROR,
};

Key_Value_Type match_key_value_pair(std::string_view line, std::string* key, std::string* value)
{
    size_t key_from = skip_set(0, line, WHITESPACE);
    size_t key_to = skip_set(key_from, line, WHITESPACE MARKERS, SKIP_INVERSE);
    
    if(key_from == line.size())
        return KEY_VALUE_BLANK;

    size_t marker_from = skip_set(key_to, line, WHITESPACE);
    size_t marker_to = skip_set(marker_from, line, MARKERS);

    size_t value_from = skip_set(marker_to, line, WHITESPACE);
    size_t value_to = skip_set_reverse(line.size(), value_from, line, WHITESPACE) + 1;

    std::string_view found_key = line.substr(key_from, key_to - key_from);
    std::string_view found_marker = line.substr(marker_from, marker_to - marker_from);
    std::string_view found_value = line.substr(value_from, value_to - value_from);

    if(found_key.size() > 0 && (found_key[0] == '#' || found_key[0] == ';'))
        return KEY_VALUE_COMMENT;

    if(found_marker == "=" || found_marker == ":" || found_marker == ",")
    {
        *key = found_key;
        *value = found_value;
        return KEY_VALUE_ENTRY;
    }
    else
    {
        return KEY_VALUE_ERROR;
    }
}

using Key_Value = std::unordered_map<std::string, std::string>;

Key_Value to_key_value(std::string_view source)
{
    Key_Value key_value;
    int line_number = 1;
    for(size_t i = 0; i < source.size(); line_number++)
    {
        std::string_view line = get_line(&i, source);
        std::string key;
        std::string value;
        
        Key_Value_Type type = match_key_value_pair(line, &key, &value);
        if(type == KEY_VALUE_ERROR)
            LOG_ERROR("to_key_value", "invalid syntax on line %i '%s'. Ignoring.", line_number, std::string(line).c_str());
        else if(type == KEY_VALUE_ENTRY)
            key_value[key] = value;
    }

    return key_value;
}

bool key_value_get_any(const Key_Value& map, void* out, const char* str, const char* type_fmt, const char* type)
{
    auto found = map.find(str);
    bool state = found != map.end();
    if(state == false)
        LOG_ERROR("config", "couldnt find %s '%s'", type, str);
    else
    {
        const char* val = found->second.c_str();  
        state = sscanf(val, type_fmt, out) == 1;
        if(state == false)
            LOG_ERROR("config", "Couldnt match %s. Got: '%s'. While parsing value '%s'", type, val, str);
    }
    return state;
}

bool key_value_get_vec2(const Key_Value& map, Vec2* out, const char* str)
{
    auto found = map.find(str);
    bool state = found != map.end();
    if(state == false)
        LOG_ERROR("config", "couldnt find Vec2 '%s'", str);
    else
    {
        const char* val = found->second.c_str();  
        state = sscanf(val, REAL_FMT " " REAL_FMT, &out->x, &out->y) == 2;
        if(state == false)
            LOG_ERROR("config", "Couldnt match Vec2. Got: '%s'. While parsing value '%s'", val, str);
    }
    return state;
}

bool key_value_get_int(const Key_Value& map, int* out, const char* str)
{
    return key_value_get_any(map, out, str, "%i", "int");
}

bool key_value_get_real(const Key_Value& map, Real* out, const char* str)
{
    return key_value_get_any(map, out, str, REAL_FMT, "Real");
}

bool key_value_get_float(const Key_Value& map, float* out, const char* str)
{
    return key_value_get_any(map, out, str, "%f", "float");
}

bool key_value_get_double(const Key_Value& map, double* out, const char* str)
{
    return key_value_get_any(map, out, str, "%lf", "double");
}

bool key_value_get_string(const Key_Value& map, std::string* out, const char* str)
{
    auto found = map.find(str);
    bool state = found != map.end();
    if(state)
        *out = found->second;
    else
        LOG_ERROR("config", "couldnt find or match string '%s'", str);
    return state;
}

bool key_value_get_bool(const Key_Value& map, bool* out, const char* str)
{
    auto found = map.find(str);
    bool state = found != map.end();
    if(state)
    {
        if(found->second == "true" || found->second == "1")
            *out = true;
        else if(found->second == "false" || found->second == "0")
            *out = false;
        else
        {
            LOG_ERROR("config", "couldnt find or match bool '%s = %s' ", str, found->second.c_str());
            state = false;
        }
    }
    else
        LOG_ERROR("config", "couldnt find bool '%s'", str);

    return state;
}

#include <stdarg.h>
std::string format_string(const char* format, ...)
{
    std::string out;
    va_list args;
    va_start(args, format);

    va_list args_copy;
    va_copy(args_copy, args);

    int size = vsnprintf(NULL, 0, format, args_copy);

    out.resize((size_t) size+5);
    size = vsnprintf(&out[0], out.size(), format, args);
    out.resize((size_t) size);
    va_end(args);

    return out;
}

#include <filesystem>
bool allen_cahn_read_config(const char* path, Allen_Cahn_Config* config)
{
    Allen_Cahn_Config null_config = {};
    *config = null_config; 
    Allen_Cahn_Initial_Conditions* initial = &config->initial_conditions;
    Allen_Cahn_Snapshots* snaps = &config->snapshots;
    Allen_Cahn_Params* params = &config->params;

    std::string source;
    bool state = file_read_entire(path, &source);
    if(state == false)
        LOG_ERROR("config", "coudlnt read config file '%s'. Current working directory '%s'", path, std::filesystem::current_path().string().c_str());
    else
    {
        Key_Value pairs = to_key_value(source);

        uint8_t matched_initial = true
            & (uint8_t) key_value_get_real(pairs, &initial->inside_phi, "inside_phi")
            & (uint8_t) key_value_get_real(pairs, &initial->inside_T, "inside_T")
            & (uint8_t) key_value_get_real(pairs, &initial->outside_phi, "outside_phi")
            & (uint8_t) key_value_get_real(pairs, &initial->outside_T, "outside_T")
            & (uint8_t) key_value_get_vec2(pairs, &initial->circle_center, "circle_center")
            & (uint8_t) key_value_get_real(pairs, &initial->circle_inner_radius, "circle_inner_radius")
            & (uint8_t) key_value_get_real(pairs, &initial->circle_outer_radius, "circle_outer_radius")
            & (uint8_t) key_value_get_vec2(pairs, &initial->square_from, "square_from")
            & (uint8_t) key_value_get_vec2(pairs, &initial->square_to, "square_to");

        uint8_t matched_snaps = true
            & (uint8_t) key_value_get_double(pairs, &snaps->snapshot_every, "every")
            & (uint8_t) key_value_get_int(pairs, &snaps->snapshot_times, "times")
            & (uint8_t) key_value_get_bool(pairs, &snaps->snapshot_initial_conditions, "snapshot_initial_conditions")
            & (uint8_t) key_value_get_string(pairs, &snaps->folder, "folder")
            & (uint8_t) key_value_get_string(pairs, &snaps->prefix, "prefix")
            & (uint8_t) key_value_get_string(pairs, &snaps->postfix, "postfix");
            

        const char* solver_strings[] = {"explicit", "semi-implicit", "semi-implicit-coupled"};
        Solver_Type solver_types[] = {SOLVER_TYPE_EXPLICIT, SOLVER_TYPE_SEMI_IMPLICIT, SOLVER_TYPE_SEMI_IMPLICIT_COUPLED};

        Solver_Type solver_type = SOLVER_TYPE_NONE;
        std::string solver_string;
        bool matched_solver = key_value_get_string(pairs, &solver_string, "solver");
        if(matched_solver)
        {
            for(size_t i = 0; i < sizeof(solver_strings) / sizeof(*solver_strings); i++)
            {
                if(solver_string == solver_strings[i])
                    solver_type = solver_types[i];
            }

            if(solver_type == SOLVER_TYPE_NONE)
            {
                LOG_ERROR("config", "invalid value '%s' for solver! Expecting one of:", solver_string.data());
                for(size_t i = 0; i < sizeof(solver_strings) / sizeof(*solver_strings); i++)
                    LOG_ERROR("config", "> %s", solver_strings[i]);

                matched_solver = false;
            }
            else
                config->solver = solver_type;
        }

        uint8_t matched_program = true
            & (uint8_t) matched_solver
            & (uint8_t) key_value_get_double(pairs, &config->stop_after, "stop_after")
            & (uint8_t) key_value_get_bool(pairs, &config->interactive_mode, "interactive")
            & (uint8_t) key_value_get_bool(pairs, &config->linear_filtering, "linear_filtering")
            & (uint8_t) key_value_get_real(pairs, &config->display_min, "display_min")
            & (uint8_t) key_value_get_real(pairs, &config->display_max, "display_max");
            
        uint8_t matched_params = true
            & (uint8_t) key_value_get_int(pairs, &params->m, "mesh_size_x")
            & (uint8_t) key_value_get_int(pairs, &params->n, "mesh_size_y")
            & (uint8_t) key_value_get_real(pairs, &params->L0, "L0")
            & (uint8_t) key_value_get_real(pairs, &params->dt, "dt")
            & (uint8_t) key_value_get_real(pairs, &params->L, "L")
            & (uint8_t) key_value_get_real(pairs, &params->xi, "xi")
            & (uint8_t) key_value_get_real(pairs, &params->a, "a")
            & (uint8_t) key_value_get_real(pairs, &params->b, "b")
            & (uint8_t) key_value_get_real(pairs, &params->alpha, "alpha")
            & (uint8_t) key_value_get_real(pairs, &params->beta, "beta")
            & (uint8_t) key_value_get_real(pairs, &params->Tm, "Tm")
            & (uint8_t) key_value_get_real(pairs, &params->Tinit, "Tini")
            & (uint8_t) key_value_get_real(pairs, &params->S, "S")
            & (uint8_t) key_value_get_real(pairs, &params->m0, "m")
            & (uint8_t) key_value_get_real(pairs, &params->theta0, "theta0")
            & (uint8_t) key_value_get_bool(pairs, &params->do_anisotropy, "do_anisotropy")
            ;

        state = matched_initial && matched_snaps && matched_params && matched_program;
        if(state == false)
            LOG_ERROR("config", "couldnt find or parse some config entries. Config is only partially loaded!");
    }

    return state;
}
