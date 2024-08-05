#pragma once
#include "simulation.h"
#include <string>

typedef struct Vec2{
    double x;
    double y;
} Vec2;

typedef struct Sim_Config{
    std::string entire_config_file;

    Sim_Params params;

    //Simulation parameters
    double simul_stop_time;

    //initial nonditions
    std::string init_path;
    double init_inside_phi;
    double init_inside_T;

    double init_outside_phi;
    double init_outside_T;

    Vec2   init_circle_center;
    double init_circle_radius;
    double init_circle_fade;

    Vec2   init_square_from;
    Vec2   init_square_to;

    //snapshots
    double snapshot_every;
    int    snapshot_times;
    bool   snapshot_initial_conditions;

    std::string snapshot_folder;
    std::string snapshot_prefix;
    std::string snapshot_postfix;

    //App settings
    bool app_run_simulation;
    bool app_run_tests;
    bool app_run_benchmarks;
    bool app_collect_stats;
    bool app_collect_step_residuals;
    double app_collect_stats_every;

    bool   app_interactive_mode;
    bool   app_linear_filtering;
    double app_display_min;
    double app_display_max;

} Sim_Config;

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
            return i + 1;
        if(is_in_set == false && options == SKIP_NORMAL)
            return i + 1;
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

bool file_write_entire(const char* path, std::string_view content)
{
    std::ofstream t;
    t.open(path);
    t << content;
    return !t.bad();
}

#define WHITESPACE " \t\n\r\v\f"
#define MARKERS    "=:"

enum Key_Value_Type {
    KEY_VALUE_BLANK,
    KEY_VALUE_COMMENT,
    KEY_VALUE_SECTION,
    KEY_VALUE_ENTRY,
    KEY_VALUE_ERROR,
};

Key_Value_Type match_key_value_pair(std::string_view line, std::string* key, std::string* value, std::string* section)
{
    size_t key_from = skip_set(0, line, WHITESPACE);
    size_t key_to = skip_set(key_from, line, WHITESPACE MARKERS, SKIP_INVERSE);
    
    if(key_from == line.size())
        return KEY_VALUE_BLANK;

    size_t marker_from = skip_set(key_to, line, WHITESPACE);
    size_t marker_to = skip_set(marker_from, line, MARKERS);

    size_t value_from = skip_set(marker_to, line, WHITESPACE);
    size_t value_to = skip_set_reverse(line.size(), value_from, line, WHITESPACE);

    std::string_view found_key = line.substr(key_from, key_to - key_from);
    std::string_view found_marker = line.substr(marker_from, marker_to - marker_from);
    std::string_view found_value = line.substr(value_from, value_to - value_from);

    //If is comment
    if(found_key.size() > 0 && (found_key[0] == '#' || found_key[0] == ';'))
        return KEY_VALUE_COMMENT;

    //If is section
    if(found_key.size() > 0 && found_key.front() == '[' && found_key.back() == ']')
    {
        *section = found_key.substr(1, found_key.size() - 2);
        return KEY_VALUE_SECTION;
    }

    if(found_marker == "=" || found_marker == ":" || found_marker == ",")
    {
        //Remove comment from value
        size_t comment_index1 = found_value.find(';');
        size_t comment_index2 = found_value.find('#');
        if(comment_index1 == (size_t) -1)
            comment_index1 = found_value.size();
        if(comment_index2 == (size_t) -1)
            comment_index2 = found_value.size();

        size_t comment_index = comment_index1 < comment_index2 ? comment_index1 : comment_index2;
        size_t comment_removed_value_to = skip_set_reverse(comment_index, 0, found_value, WHITESPACE);

        std::string_view comment_removed_value = found_value.substr(0, comment_removed_value_to);

        *key = found_key;
        *value = comment_removed_value;
        return KEY_VALUE_ENTRY;
    }
    else
    {
        return KEY_VALUE_ERROR;
    }
}

#define SECTION_MARKER "~!~!~!~" //A very unlikely set of characters

using Key_Value = std::unordered_map<std::string, std::string>;

Key_Value to_key_value(std::string_view source)
{
    Key_Value key_value;
    int line_number = 1;
    std::string key;
    std::string value;
    std::string current_section;
    for(size_t i = 0; i < source.size(); line_number++)
    {
        std::string_view line = get_line(&i, source);
        key.clear();
        value.clear();
        
        Key_Value_Type type = match_key_value_pair(line, &key, &value, &current_section);
        if(type == KEY_VALUE_ERROR)
            LOG_ERROR("to_key_value", "invalid syntax on line %i '%s'. Ignoring.", line_number, std::string(line).c_str());
        else if(type == KEY_VALUE_ENTRY)
            key_value[current_section + SECTION_MARKER + key] = value;
    }

    return key_value;
}

std::string key_value_key(const char* section, const char* str)
{
    return std::string(section) + SECTION_MARKER + str;
}

bool key_value_get_any(const Key_Value& map, void* out, const char* section, const char* str, const char* type_fmt, const char* type)
{
    std::string key = key_value_key(section, str);
    auto found = map.find(key);
    bool state = found != map.end();
    if(state == false)
        LOG_ERROR("config", "couldnt find %s '%s' in section [%s]", type, str, section);
    else
    {
        const char* val = found->second.c_str();  
        state = sscanf(val, type_fmt, out) == 1;
        if(state == false)
            LOG_ERROR("config", "Couldnt match %s. Got: '%s'. While parsing value '%s'", type, val, str);
    }
    return state;
}

bool key_value_get_vec2(const Key_Value& map, Vec2* out, const char* section, const char* str)
{
    std::string key = key_value_key(section, str);
    auto found = map.find(key);
    bool state = found != map.end();
    if(state == false)
        LOG_ERROR("config", "couldnt find Vec2 '%s' in section [%s]", str, section);
    else
    {
        const char* val = found->second.c_str();  
        state = sscanf(val, "%lf %lf", &out->x, &out->y) == 2;
        if(state == false)
            LOG_ERROR("config", "Couldnt match Vec2. Got: '%s'. While parsing value '%s'", val, str);
    }
    return state;
}

bool key_value_get_int(const Key_Value& map, int* out, const char* section, const char* str)
{
    return key_value_get_any(map, out, section, str, "%i", "int");
}

bool key_value_get_float(const Key_Value& map, float* out, const char* section, const char* str)
{
    return key_value_get_any(map, out, section, str, "%f", "float");
}

bool key_value_get_double(const Key_Value& map, double* out, const char* section, const char* str)
{
    return key_value_get_any(map, out, section, str, "%lf", "double");
}

bool key_value_get_string(const Key_Value& map, std::string* out, const char* section, const char* str)
{
    std::string key = key_value_key(section, str);
    auto found = map.find(key);
    bool state = found != map.end();
    if(state)
        *out = found->second;
    else
        LOG_ERROR("config", "couldnt find or match string '%s' in section [%s]", str, section);
    return state;
}

bool key_value_get_bool(const Key_Value& map, bool* out, const char* section, const char* str)
{
    std::string key = key_value_key(section, str);
    auto found = map.find(key);
    bool state = found != map.end();
    if(state == false)
        LOG_ERROR("config", "couldnt find bool '%s' in section [%s]", str, section);
    else
    {
        if(found->second == "true" || found->second == "1")
            *out = true;
        else if(found->second == "false" || found->second == "0")
            *out = false;
        else
        {
            LOG_ERROR("config", "couldnt match bool '%s = %s' ", str, found->second.c_str());
            state = false;
        }
    }

    return state;
}

bool key_value_get_solver_type(const Key_Value& map, Sim_Solver_Type* out, const char* section, const char* str)
{
    Sim_Solver_Type solver_type = SOLVER_TYPE_NONE;
    std::string solver_string;
    bool matched_solver = key_value_get_string(map, &solver_string, section, str);
    if(matched_solver)
    {
        for(size_t i = 0; i < SOLVER_TYPE_ENUM_COUNT; i++)
        {
            Sim_Solver_Type type = (Sim_Solver_Type) i;
            const char* name = solver_type_to_cstring(type);
            if(solver_string == name)
                solver_type = type;
        }

        if(solver_type == SOLVER_TYPE_NONE)
        {
            LOG_ERROR("config", "invalid value '%s' for solver! Expecting one of:", solver_string.data());
            for(size_t i = SOLVER_TYPE_NONE + 1; i < SOLVER_TYPE_ENUM_COUNT; i++)
                LOG_ERROR(">config", "%i. '%s'", i, solver_type_to_cstring((Sim_Solver_Type) i));

            matched_solver = false;
        }
        else
            *out = solver_type;
    }

    return matched_solver;
}

bool key_value_get_boundary_type(const Key_Value& map, Sim_Boundary_Type* out, const char* section, const char* str)
{
    Sim_Boundary_Type solver_type = BOUNDARY_ENUM_COUNT;
    std::string solver_string;
    bool matched_solver = key_value_get_string(map, &solver_string, section, str);
    if(matched_solver)
    {
        for(size_t i = 0; i < BOUNDARY_ENUM_COUNT; i++)
        {
            Sim_Boundary_Type type = (Sim_Boundary_Type) i;
            const char* name = boundary_type_to_cstring(type);
            if(solver_string == name)
                solver_type = type;
        }

        if(solver_type == BOUNDARY_ENUM_COUNT)
        {
            LOG_ERROR("config", "invalid value '%s' for boundary! Expecting one of:", solver_string.data());
            for(size_t i = BOUNDARY_PERIODIC; i < BOUNDARY_ENUM_COUNT; i++)
                LOG_ERROR(">config", "%i. '%s'", i, boundary_type_to_cstring((Sim_Boundary_Type) i));

            matched_solver = false;
        }
        else
            *out = solver_type;
    }

    return matched_solver;
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
bool allen_cahn_read_config(const char* path, Sim_Config* config)
{
    Sim_Config null_config = {};
    *config = null_config; 
    Sim_Params* params = &config->params;

    bool state = file_read_entire(path, &config->entire_config_file);
    if(state == false)
        LOG_ERROR("config", "coudlnt read config file '%s'. Current working directory '%s'", path, std::filesystem::current_path().string().c_str());
    else
    {
        Key_Value pairs = to_key_value(config->entire_config_file);

        uint8_t matched_simulation = true
            & (uint8_t) key_value_get_double(pairs, &params->dt, "simulation", "dt")
            & (uint8_t) key_value_get_double(pairs, &params->L0, "simulation", "L0")
            & (uint8_t) key_value_get_double(pairs, &params->L, "simulation", "L")
            & (uint8_t) key_value_get_double(pairs, &params->xi, "simulation", "xi")
            & (uint8_t) key_value_get_double(pairs, &params->a, "simulation", "a")
            & (uint8_t) key_value_get_double(pairs, &params->b, "simulation", "b")
            & (uint8_t) key_value_get_double(pairs, &params->alpha, "simulation", "alpha")
            & (uint8_t) key_value_get_double(pairs, &params->beta, "simulation", "beta")
            & (uint8_t) key_value_get_double(pairs, &params->Tm, "simulation", "Tm")
            & (uint8_t) key_value_get_double(pairs, &params->S, "simulation", "S")
            & (uint8_t) key_value_get_double(pairs, &params->m0, "simulation", "m")
            & (uint8_t) key_value_get_double(pairs, &params->theta0, "simulation", "theta0")
            & (uint8_t) key_value_get_double(pairs, &params->gamma, "simulation", "gamma")
            & (uint8_t) key_value_get_bool(pairs, &params->do_exact, "simulation", "do_exact")
            & (uint8_t) key_value_get_solver_type(pairs, &params->solver, "simulation", "solver")
            & (uint8_t) key_value_get_boundary_type(pairs, &params->Phi_boundary, "simulation", "Phi_boundary")
            & (uint8_t) key_value_get_boundary_type(pairs, &params->T_boundary, "simulation", "T_boundary")
            & (uint8_t) key_value_get_double(pairs, &config->simul_stop_time, "simulation", "stop_after")
            & (uint8_t) key_value_get_int(pairs, &params->nx, "simulation", "mesh_size_x")
            & (uint8_t) key_value_get_int(pairs, &params->ny, "simulation", "mesh_size_y")
            & (uint8_t) key_value_get_double(pairs, &params->T_tolerance, "simulation", "T_tolerance")
            & (uint8_t) key_value_get_double(pairs, &params->Phi_tolerance, "simulation", "Phi_tolerance")
            & (uint8_t) key_value_get_double(pairs, &params->corrector_tolerance, "simulation", "corrector_tolerance")
            & (uint8_t) key_value_get_int(pairs, &params->T_max_iters, "simulation", "T_max_iters")
            & (uint8_t) key_value_get_int(pairs, &params->Phi_max_iters, "simulation", "Phi_max_iters")
            & (uint8_t) key_value_get_int(pairs, &params->corrector_max_iters, "simulation", "corrector_max_iters")
            & (uint8_t) key_value_get_bool(pairs, &params->do_corrector_loop, "simulation", "do_corrector_loop")
            & (uint8_t) key_value_get_bool(pairs, &params->do_corrector_guess, "simulation", "do_corrector_guess")
            ;
        //optional
        (uint8_t) key_value_get_double(pairs, &params->min_dt, "simulation", "min_dt");
            
        uint8_t matched_initial = true
            & (uint8_t) key_value_get_double(pairs, &config->init_inside_phi, "initial", "inside_phi")
            & (uint8_t) key_value_get_double(pairs, &config->init_inside_T, "initial", "inside_T")
            & (uint8_t) key_value_get_double(pairs, &config->init_outside_phi, "initial", "outside_phi")
            & (uint8_t) key_value_get_double(pairs, &config->init_outside_T, "initial", "outside_T")
            & (uint8_t) key_value_get_vec2(pairs, &config->init_circle_center, "initial", "circle_center")
            & (uint8_t) key_value_get_double(pairs, &config->init_circle_radius, "initial", "circle_radius")
            & (uint8_t) key_value_get_double(pairs, &config->init_circle_fade, "initial", "circle_fade")
            & (uint8_t) key_value_get_vec2(pairs, &config->init_square_from, "initial", "square_from")
            & (uint8_t) key_value_get_vec2(pairs, &config->init_square_to, "initial", "square_to");

        uint8_t matched_snaps = true
            & (uint8_t) key_value_get_double(pairs, &config->snapshot_every, "snapshot", "every")
            & (uint8_t) key_value_get_int(pairs, &config->snapshot_times, "snapshot", "times")
            & (uint8_t) key_value_get_bool(pairs, &config->snapshot_initial_conditions, "snapshot", "snapshot_initial_conditions")
            & (uint8_t) key_value_get_string(pairs, &config->snapshot_folder, "snapshot", "folder")
            & (uint8_t) key_value_get_string(pairs, &config->snapshot_prefix, "snapshot", "prefix")
            & (uint8_t) key_value_get_string(pairs, &config->snapshot_postfix, "snapshot", "postfix")
            ;

        uint8_t matched_program = true
            & (uint8_t) key_value_get_bool(pairs, &config->app_run_simulation, "program", "run_simulation")
            & (uint8_t) key_value_get_bool(pairs, &config->app_run_tests, "program", "run_tests")
            & (uint8_t) key_value_get_bool(pairs, &config->app_run_benchmarks, "program", "run_benchmarks")
            & (uint8_t) key_value_get_bool(pairs, &config->app_interactive_mode, "program", "interactive")
            & (uint8_t) key_value_get_bool(pairs, &config->app_linear_filtering, "program", "linear_filtering")
            & (uint8_t) key_value_get_bool(pairs, &config->app_collect_stats, "program", "collect_stats")
            & (uint8_t) key_value_get_bool(pairs, &config->app_collect_step_residuals, "program", "collect_step_residual")
            & (uint8_t) key_value_get_double(pairs, &config->app_collect_stats_every, "program", "collect_stats_every")
            & (uint8_t) key_value_get_double(pairs, &config->app_display_min, "program", "display_min")
            & (uint8_t) key_value_get_double(pairs, &config->app_display_max, "program", "display_max")
            ;
            
        //TODO: this should be more proper
        if(params->do_exact)
        {
            double A = 1.0/16;
            double h = MAX(params->L0/params->nx, params->L0/params->ny);
            params->Tm = 0;
            params->L = 1;
            if(params->solver != SOLVER_TYPE_EXACT)
                params->dt = A/4 * h*h;
            LOG_WARN("config", "dt used: %e", params->dt);
            params->a = 1;
            params->b = 1;
            params->alpha = 1;
            params->beta = 1/0.001;
            params->S = 0;
            params->xi = params->L0/params->nx * 11/10;
            config->init_circle_radius = 0.25;
        }
            
        state = matched_initial && matched_snaps && matched_simulation && matched_program;
        if(state == false)
            LOG_ERROR("config", "couldnt find or parse some config entries. Config is only partially loaded!");
        else
            LOG_OKAY("config", "config successfully read!");
    }

    return state;
}
