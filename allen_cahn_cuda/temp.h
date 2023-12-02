#include "platform.h"

#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif
#ifndef UNICODE
    #define UNICODE
#endif 

#include <locale.h>
#include <assert.h>
#include <windows.h>

#include <memoryapi.h>
#include <processthreadsapi.h>
#include <winnt.h>
#include <profileapi.h>
#include <hidusage.h>
#include <windowsx.h>
#include <intrin.h>
#include <direct.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")

#undef near
#undef far

enum { WINDOWS_PLATFORM_FILE_TYPE_PIPE = PLATFORM_FILE_TYPE_PIPE };
#undef PLATFORM_FILE_TYPE_PIPE


#define _TRANSLATED_ERRORS_SIMULATANEOUS 8
#define _EPHEMERAL_STRING_SIMULTANEOUS 4

typedef struct Platform_State {
    Platform_Allocator allocator;

    int64_t perf_counter_base;
    int64_t perf_counter_freq;
    int64_t startup_epoch_time;

    wchar_t* ephemeral_strings[_EPHEMERAL_STRING_SIMULTANEOUS];
    int64_t ephemeral_strings_slot;

    char* _translated_errors[_TRANSLATED_ERRORS_SIMULATANEOUS];
    int64_t _translated_error_slot;

    char* directory_current_working_cached;
    int64_t current_working_cached_generation;
    int64_t current_working_generation;
    char* directory_executable_cached;

    bool   stack_trace_init;
    HANDLE stack_trace_process;
    DWORD  stack_trace_error;
} Platform_State;

Platform_State gp_state = {0};

//=========================================
// Virtual memory
//=========================================
void* platform_virtual_reallocate(void* adress, int64_t bytes, Platform_Virtual_Allocation action, Platform_Memory_Protection protection)
{
    //CreateThread();
    if(action == PLATFORM_VIRTUAL_ALLOC_RELEASE)
    {
        (void) VirtualFree(adress, 0, MEM_RELEASE);  
        return NULL;
    }

    if(action == PLATFORM_VIRTUAL_ALLOC_DECOMMIT)
    {
        (void) VirtualFree(adress, bytes, MEM_DECOMMIT);  
        return NULL;
    }
 
    int prot = 0;
    if(protection == PLATFORM_MEMORY_PROT_NO_ACCESS)
        prot = PAGE_NOACCESS;
    if(protection == PLATFORM_MEMORY_PROT_READ)
        prot = PAGE_READONLY;
    else
        prot = PAGE_READWRITE;

    if(action == PLATFORM_VIRTUAL_ALLOC_RESERVE)
        return VirtualAlloc(adress, bytes, MEM_RESERVE, prot);
    else
        return VirtualAlloc(adress, bytes, MEM_COMMIT, prot);
}

void* platform_heap_reallocate(int64_t new_size, void* old_ptr, int64_t old_size, int64_t align)
{
    assert(align > 0 && new_size >= 0 && old_size >= 0);
    
    (void) old_size;
    #ifndef NDEBUG
    if(old_ptr != NULL && old_size != 0)
    {
        //@TODO: Remove this. Actually refactor the entire way we do allocations
        int64_t correct_size = platform_heap_get_block_size(old_ptr, align);
        assert(old_size == correct_size && "incorrect old_size passed to platform_heap_reallocate!");
    }
    #endif

    if(new_size == 0)
    {
        _aligned_free(old_ptr);
        return NULL;
    }
    return _aligned_realloc(old_ptr, (size_t) new_size, (size_t) align);
}


int64_t platform_heap_get_block_size(void* old_ptr, int64_t align)
{
    int64_t size = _aligned_msize(old_ptr, (size_t) align, 0);
    return size;
}

//=========================================
// Threading
//=========================================
int64_t platform_thread_get_proccessor_count()
{
    return GetCurrentProcessorNumber();
}

#pragma warning(disable : 4057)
//typedef DWORD (WINAPI *PTHREAD_START_ROUTINE)(
//    LPVOID lpThreadParameter
//    );
Platform_Thread platform_thread_create(int (*func)(void*), void* context, int64_t stack_size)
{
    Platform_Thread thread = {0};
    PTHREAD_START_ROUTINE cast_func = (PTHREAD_START_ROUTINE) (void*) func;
    if(stack_size <= 0)
        stack_size = 0;
    thread.handle = CreateThread(0, stack_size, cast_func, context, 0, (LPDWORD) &thread.id);
    return thread;
}
void platform_thread_destroy(Platform_Thread* thread)
{
    Platform_Thread null = {0};
    CloseHandle(thread->handle);
    *thread = null;
}

int32_t platform_thread_get_id()
{
    return GetCurrentThreadId();
}
void platform_thread_yield()
{
    SwitchToThread();
}
void platform_thread_sleep(int64_t ms)
{
    Sleep((DWORD) ms);
}

void platform_thread_exit(int code)
{
    ExitThread(code);
}

int platform_thread_join(Platform_Thread thread)
{
    WaitForSingleObject(thread.handle, INFINITE);
    DWORD code = 0;
    GetExitCodeThread(thread.handle, &code);
    return code;
}

Platform_Error _error_code(bool state)
{
    if(state)
        return PLATFORM_ERROR_OK;
    else
        return (Platform_Error) GetLastError();
}

Platform_Error  platform_mutex_create(Platform_Mutex* mutex)
{
    platform_mutex_destroy(mutex);
    Platform_Mutex out = {0};
    out.handle = (void*) CreateMutexW(NULL, FALSE, L"platform_mutex_create");
    *mutex = out;

    return _error_code(true);
}

void platform_mutex_destroy(Platform_Mutex* mutex)
{
    CloseHandle((HANDLE) mutex->handle);
}

Platform_Error platform_mutex_acquire(Platform_Mutex* mutex)
{
    DWORD dwWaitResult = WaitForSingleObject( 
            (HANDLE) mutex->handle, 
            INFINITE);  // no time-out interval
   
    return _error_code(dwWaitResult == WAIT_OBJECT_0);
}

void platform_mutex_release(Platform_Mutex* mutex)
{
    bool return_val = ReleaseMutex((HANDLE) mutex->handle);
    assert(return_val && "there is little we can do about errors appart from hoping they dont occur");
}


//=========================================
// Timings
//=========================================
int64_t platform_perf_counter()
{
    LARGE_INTEGER ticks;
    ticks.QuadPart = 0;
    (void) QueryPerformanceCounter(&ticks);
    return ticks.QuadPart;
}

int64_t platform_perf_counter_startup()
{
    if(gp_state.perf_counter_base == 0)
        gp_state.perf_counter_base = platform_perf_counter();
    return gp_state.perf_counter_base;
}

int64_t platform_perf_counter_frequency()
{
    if(gp_state.perf_counter_freq == 0)
    {
        LARGE_INTEGER ticks;
        ticks.QuadPart = 0;
        (void) QueryPerformanceFrequency(&ticks);
        gp_state.perf_counter_freq = ticks.QuadPart;
    }
    return gp_state.perf_counter_freq;
}

//time -> filetime
//(ts * 10000000LL) + 116444736000000000LL = t
//(ts * 10LL) + 116444736000LL = tu = t/1 0000 000
    
//filetime -> time
//ts = (t - 116444736000000000LL) / 10000000;
//t / 10000000 - 11644473600LL = ts;
//t / 10 - 11644473600 000 000LL = tu = ts*1 0000 000;
static int64_t _filetime_to_epoch_time(FILETIME t)  
{    
    ULARGE_INTEGER ull;    
    ull.LowPart = t.dwLowDateTime;    
    ull.HighPart = t.dwHighDateTime;
    int64_t tu = ull.QuadPart / 10 - 11644473600000000LL;
    return tu;
}

static FILETIME _epoch_time_to_filetime(int64_t tu)  
{    
    ULARGE_INTEGER time_value;
    FILETIME time;
    time_value.QuadPart = (tu + 11644473600000000LL)*10;

    time.dwLowDateTime = time_value.LowPart;
    time.dwHighDateTime = time_value.HighPart;
    return time;
}
static time_t _filetime_to_time_t(FILETIME ft)  
{    
    ULARGE_INTEGER ull;    
    ull.LowPart = ft.dwLowDateTime;    
    ull.HighPart = ft.dwHighDateTime;    
    return (time_t) (ull.QuadPart / 10000000ULL - 11644473600ULL);  
}

int64_t platform_universal_epoch_time()
{
    FILETIME filetime;
    GetSystemTimeAsFileTime(&filetime);
    int64_t epoch_time = _filetime_to_epoch_time(filetime);
    return epoch_time;
}

int64_t platform_startup_epoch_time()
{
    if(gp_state.startup_epoch_time == 0)
        gp_state.startup_epoch_time = platform_universal_epoch_time();

    return gp_state.startup_epoch_time;
}

int64_t platform_local_epoch_time()
{
    FILETIME filetime;
    GetSystemTimeAsFileTime(&filetime);
    FILETIME local_filetime = {0};
    bool okay = FileTimeToLocalFileTime(&filetime, &local_filetime);
    assert(okay); (void) okay;
    int64_t epoch_time = _filetime_to_epoch_time(local_filetime);
    return epoch_time;
}

Platform_Calendar_Time platform_epoch_time_to_calendar_time(int64_t epoch_time_usec)
{
    const int64_t _EPOCH_YEAR              = (int64_t) 1970;
    const int64_t _MILLISECOND_MICROSECOND = (int64_t) 1000;
    const int64_t _SECOND_MICROSECONDS     = (int64_t) 1000000;
    const int64_t _DAY_SECONDS             = (int64_t) 86400;
    const int64_t _YEAR_SECONDS            = (int64_t) 31556952;
    const int64_t _DAY_MICROSECONDS        = (_DAY_SECONDS * _SECOND_MICROSECONDS);
    const int64_t _YEAR_MICROSECONDS       = (_YEAR_SECONDS * _SECOND_MICROSECONDS);

    (void) _DAY_SECONDS;
    (void) _DAY_MICROSECONDS;

    SYSTEMTIME systime = {0};
    FILETIME filetime = _epoch_time_to_filetime(epoch_time_usec);
    bool okay = FileTimeToSystemTime(&filetime, &systime);
    assert(okay);

    Platform_Calendar_Time time = {0};
    time.day = (int8_t) systime.wDay;
    time.day_of_week = (int8_t) systime.wDayOfWeek;
    time.hour = (int8_t) systime.wHour;
    time.millisecond = (int16_t) systime.wMilliseconds;
    time.minute = (int8_t) systime.wMinute;
    time.month = (int8_t) systime.wMonth - 1;
    time.second = (int8_t) systime.wSecond;
    time.year = (int32_t) systime.wYear;

    int64_t years_since_epoch = (int64_t) time.year - _EPOCH_YEAR;
    int64_t microsec_diff = epoch_time_usec - years_since_epoch*_YEAR_MICROSECONDS;
    //int64_t microsec_remainder = microsec_diff % YEAR_MICROSECONDS;
    //int64_t day_of_year = (microsec_remainder / DAY_MICROSECONDS);
    //int64_t microsecond = (microsec_remainder % MILLISECOND_MICROSECOND);
    int64_t microsecond = (microsec_diff % _MILLISECOND_MICROSECOND);

    //time.day_of_year = (int16_t) day_of_year;
    time.microsecond = (int16_t) microsecond;

    //int64_t times = time.day_of_year;
    assert(0 <= time.month && time.month < 12);
    assert(0 <= time.day && time.day < 31);
    assert(0 <= time.hour && time.hour < 24);
    assert(0 <= time.minute && time.minute < 60);
    assert(0 <= time.second && time.second < 60);
    assert(0 <= time.millisecond && time.millisecond < 1000);
    assert(0 <= time.day_of_week && time.day_of_week < 7);
    //assert(0 <= time.day_of_year && time.day_of_year <= 365);
    return time;
}

int64_t platform_calendar_time_to_epoch_time(Platform_Calendar_Time calendar_time)
{
    SYSTEMTIME systime = {0};
    systime.wDay = calendar_time.day;
    systime.wDayOfWeek = calendar_time.day_of_week;
    systime.wHour = calendar_time.hour;
    systime.wMilliseconds = calendar_time.millisecond;
    systime.wMinute = calendar_time.minute;
    systime.wMonth = calendar_time.month;
    systime.wSecond = calendar_time.second;
    systime.wYear = (WORD) calendar_time.year;

    FILETIME filetime;
    bool okay = SystemTimeToFileTime(&systime, &filetime);
    assert(okay);

    int64_t epoch_time = _filetime_to_epoch_time(filetime);
    epoch_time += calendar_time.microsecond;

    #ifndef NDEBUG
    Platform_Calendar_Time roundtrip = platform_epoch_time_to_calendar_time(epoch_time);
    assert(memcmp(&roundtrip, &calendar_time, sizeof calendar_time) == 0 && "roundtrip must be correct");
    #endif // !NDEBUG

    return epoch_time;
}


//=========================================
// Filesystem
//=========================================


//typedef struct _Wchar_String
//{
//    const wchar_t* str
//}

#define IO_LOCAL_BUFFER_SIZE 1024
#define IO_NORMALIZE_LINUX 0
#define IO_NORMALIZE_DIRECTORY 0
#define IO_NORMALIZE_FILE 0


typedef struct Buffer_Header {
    int64_t size;
    int64_t capacity;
    int32_t item_size;
    int16_t is_alloced;
    int16_t init_at_line;
    Platform_Allocator* alloc;
} Buffer_Header;

Buffer_Header* buffer_header(void* buffer)
{
    assert(data);
    buffer_check_invariant(buffer);
    return (Buffer_Header*) buffer - 1; 
}

void buffer_check_invariant(void* buffer)
{
    if(buffer == NULL)
        return;

    Buffer_Header* header = (Buffer_Header*) buffer - 1;
    assert(header->item_size > 0);
    assert(header->capacity > 0);
    assert(header->size < header->capacity);
    assert(header->is_alloced == false || header->is_alloced == true);
    assert(header->init_at_line >= 0);
    assert(header->alloc != NULL);
}

void buffer_deinit(void** buffer)
{
    buffer_check_invariant(*buffer);
    if(*buffer)
    {
        Buffer_Header* header = buffer_header(*buffer);
        if(header->is_alloced)
        {
            int64_t total_size = header->capacity*header->item_size + sizeof *header;
            header->alloc->reallocate(header->alloc->context, 0, header, total_size);
        }
    }
}

Platform_Allocator* platform_alloc()
{
    return NULL;
}

void* buffer_init(Platform_Allocator* alloc, int64_t initial_capacity, int64_t item_size)
{
    assert(item_size > 0);
    assert(initial_capacity >= 0);
    assert(alloc != NULL);

    if(initial_capacity < 7)
        initial_capacity = 7;
    
    initial_capacity += 1;
    int64_t total_size = initial_capacity*item_size + sizeof(Buffer_Header);
    Buffer_Header* header = (Buffer_Header*) alloc->reallocate(alloc->context, total_size, NULL, 0);
    assert(header);

    memset(header, 0, total_size);
    
    header->alloc = alloc;
    header->capacity = initial_capacity;
    header->item_size = item_size;
    header->size = 0;
    header->is_alloced = true;
    buffer_check_invariant(header + 1);
    return header + 1;
}

void* buffer_init_backed(Platform_Allocator* alloc, void* backing, int64_t backing_size, int64_t item_size)
{
    assert(item_size > 0);
    int64_t num_items = (backing_size - sizeof(Buffer_Header) + item_size - 1) / item_size;
    if(backing == NULL || num_items <= 0)
        return buffer_init(alloc, -1, item_size);

    Buffer_Header* header = (Buffer_Header*) backing;
    memset(header, 0, backing_size);
    
    header->alloc = alloc;
    header->capacity = initial_capacity;
    header->item_size = item_size;
    header->size = 0;
    header->is_alloced = false;
    buffer_check_invariant(header + 1);
    return header + 1;
}

void buffer_resize(void** buffer_ptr, int64_t new_size)
{
    void* buffer = *buffer_ptr;
    if(buffer == NULL)
        return;

    Buffer_Header* header = buffer_header(buffer);
    if(header->capacity <= new_size)
    {
        int64_t new_capacity = header->capacity;
        while(new_capacity <= new_size)
            new_capacity *= 2;

        *buffer_ptr = buffer_init(header->alloc, new_capacity, header->item_size);
        header = buffer_header(buffer);
        memcpy(*buffer_ptr, buffer, header->item_size*header->size);
    }

    header->item_size = new_size;
    buffer_deinit(buffer);
}

void buffer_append(void** buffer_ptr, const void* data, int64_t data_count, int64_t data_item_size)
{
    void* buffer = *buffer_ptr;
    if(buffer == NULL)
        return;
    
    Buffer_Header* header = buffer_header(buffer);
    assert(data_item_size == header->item_size); (void) data_item_size;
    int64_t size_before = header->size;

    buffer_resize(&buffer, size_before + data_item_size);
    memcpy((char*) buffer + size_before*data_item_size, data, data_count*data_item_size);
    *buffer_ptr = buffer;
}

static int64_t _wcslen(const wchar_t* str)
{
    if(str == NULL)
        return 0;
    else
        return (int64_t) wcslen(str);
}
static int64_t _strlen(const char* str)
{
    if(str == NULL)
        return 0;
    else
        return (int64_t) strlen(str);
}

char* buffer_from_string(const char* string)
{
    int64_t len = _strlen(string);
    void* buffer = buffer_init(platform_alloc(), len, sizeof *string);
    buffer_append(&buffer, string, len, sizeof *string);
    return (char*) buffer;
}

wchar_t* buffer_from_wstring(const wchar_t* string)
{
    int64_t len = _wcslen(string);
    void* buffer = buffer_init(platform_alloc(), len, sizeof *string);
    buffer_append(&buffer, string, len, sizeof *string);
    return (wchar_t*) buffer;
}

void buffer_wconcat(void** buffer, const wchar_t* a, const wchar_t* b, const wchar_t* c)
{
    int64_t a_size = _wcslen(a);
    int64_t b_size = _wcslen(b);
    int64_t c_size = _wcslen(c);
    int64_t composite_size = a_size + b_size + c_size;
        
    buffer_resize(buffer, composite_size);
    memmove((wchar_t*) *buffer,                   a, sizeof(wchar_t) * a_size);
    memmove((wchar_t*) *buffer + a_size,          b, sizeof(wchar_t) * b_size);
    memmove((wchar_t*) *buffer + a_size + b_size, c, sizeof(wchar_t) * c_size);
}

void* _platform_allocator_reallocate(int64_t new_size, void* old_ptr, int64_t old_size)
{
    if(new_size == 0)
    {
        if(old_ptr == NULL || old_size == 0)
            return NULL;
    }

    if(gp_state.allocator.reallocate == NULL)
        return platform_heap_reallocate(new_size, old_ptr, old_size, sizeof(size_t));
    else
        return gp_state.allocator.reallocate(gp_state.allocator.context, new_size, old_ptr, old_size);
}


static int64_t _utf16_to_utf8(const wchar_t* utf16, int64_t utf16len, char** append_to) 
{
    if (utf16 == NULL || utf16len == 0) 
    {
        utf16 = L"";
        utf16len = 0;
    }

    int utf8len = WideCharToMultiByte(CP_UTF8, 0, utf16, (int) utf16len, NULL, 0, NULL, NULL);
    int64_t size_before = append_to->size;
    buffer_resize(append_to, size_before + utf8len);
    WideCharToMultiByte(CP_UTF8, 0, utf16, (int) utf16len, *append_to + size_before, (int) utf8len, 0, 0);
    return utf8len;
}


static int64_t _utf8_to_utf16(const char* utf8, int64_t utf8len, wchar_t** append_to) 
{
    if (utf8 == NULL || utf8len == 0) 
    {
        utf8 = "";
        utf8len = 0;
    }

    int utf16len = MultiByteToWideChar(CP_UTF8, 0, utf8, (int) utf8len, NULL, 0);
    int64_t size_before = append_to->size;
    buffer_resize(append_to, size_before + utf8len);
    MultiByteToWideChar(CP_UTF8, 0, utf8, (int) utf8len, *append_to + size_before, (int) utf16len);
    return utf16len;
}


wchar_t* _ephemeral_wstring_convert(Platform_String path, bool normalize_path)
{
    enum {RESET_EVERY = 16, MAX_SIZE = 512, MIN_SIZE = 128};
    if(path.data == NULL || path.size == 0)
    {
        path.data = "";
        path.size = 0;
    }

    wchar_t** curr = &gp_state.ephemeral_strings[gp_state.ephemeral_strings_slot % _EPHEMERAL_STRING_SIMULTANEOUS];
    gp_state.ephemeral_strings_slot = (gp_state.ephemeral_strings_slot + 1) % _EPHEMERAL_STRING_SIMULTANEOUS;
    int64_t utf16len = _utf8_to_utf16(path.data, path.size, curr);
    wchar_t* converted = *curr;
    
    //@TODO: shrinking
    if(normalize_path)
    {
        for(int64_t i = 0; i < utf16len; i++)
        {
            if(converted[i] == '\\')
                converted[i] = '/';
        }
    }

    return converted;
}


void _ephemeral_wstring_deinit_all()
{
    for(int64_t i = 0; i < _EPHEMERAL_STRING_SIMULTANEOUS; i++)
        buffer_deinit(&gp_state.ephemeral_strings[i]);
}

const wchar_t* _ephemeral_path(Platform_String path)
{
    return _ephemeral_wstring_convert(path, true);
}

#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)

#define _BUFFER_INIT_BACKED(buffer, type, backing_size) \
    char CONCAT(__backing__, __LINE__)[(backing_size) * sizeof(type)] = {0}; \
    _buffer_init_backed(buffer, CONCAT(__backing__, __LINE__), backing_size, sizeof(type))

char* _convert_to_utf8_normalize_path(int normalize_flag, const wchar_t* path, int64_t path_size)
{
    char* buffer = NULL;
    int64_t utf8len = _utf16_to_utf8(path, path_size, &buffer);
    for(int64_t i = 0; i < utf8len; i++)
    {
        if(buffer[i] == '\\')
            buffer[i] = '/';
    }

    return buffer;
}

const char* platform_translate_error(Platform_Error error)
{
    char* trasnlated = NULL;
    int64_t length = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | 
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        (DWORD) error,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR) &trasnlated,
        0, NULL );
    
    (void) length;
    LocalFree(gp_state._translated_errors[gp_state._translated_error_slot]);
    gp_state._translated_errors[gp_state._translated_error_slot] = trasnlated;
    gp_state._translated_error_slot = (gp_state._translated_error_slot + 1) % _TRANSLATED_ERRORS_SIMULATANEOUS;

    //Strips annoying trailing whitespace
    for(int64_t i = length; i-- > 0; )
    {
        if(isspace(trasnlated[i]))
            trasnlated[i] = '\0';
        else
            break;
    }

    return trasnlated;
}

void _translated_deinit_all()
{
    for(int64_t i = 0; i < _TRANSLATED_ERRORS_SIMULATANEOUS; i++)
    {
        LocalFree((HLOCAL) gp_state._translated_errors[i]);
        gp_state._translated_errors[i] = NULL;
    }
}

Platform_Error platform_file_create(Platform_String file_path, bool* was_just_created)
{
    bool state = true;

    const wchar_t* path = _ephemeral_path(file_path);
    HANDLE handle = CreateFileW(path, 0, 0, NULL, OPEN_ALWAYS, 0, NULL);
    state = handle != INVALID_HANDLE_VALUE;

    if(was_just_created != NULL)
    {
        DWORD last_error = GetLastError();
        if(last_error == ERROR_ALREADY_EXISTS)
            *was_just_created = false;
        else
            *was_just_created = true;
    }

    CloseHandle(handle);

    return _error_code(state);
}
Platform_Error platform_file_remove(Platform_String file_path, bool* was_deleted_deleted)
{
    const wchar_t* path = _ephemeral_path(file_path);
    SetFileAttributesW(path, FILE_ATTRIBUTE_NORMAL);
    bool state = !!DeleteFileW(path);
        
    if(was_deleted_deleted != NULL)
        *was_deleted_deleted = state;

    DWORD last_error = GetLastError();
    if(last_error == ERROR_FILE_NOT_FOUND)
    {
        state = true;
        if(was_deleted_deleted != NULL)
            *was_deleted_deleted = false;
    }

    return _error_code(state);
}

Platform_Error platform_file_move(Platform_String new_path, Platform_String old_path)
{       
    const wchar_t* new_path_norm = _ephemeral_path(new_path);
    const wchar_t* old_path_norm = _ephemeral_path(old_path);
    bool state = !!MoveFileExW(old_path_norm, new_path_norm, MOVEFILE_COPY_ALLOWED);
    
    return _error_code(state);
}

Platform_Error platform_file_copy(Platform_String new_path, Platform_String old_path)
{
    const wchar_t* new_path_norm = _ephemeral_path(new_path);
    const wchar_t* old_path_norm = _ephemeral_path(old_path);
    bool state = !!CopyFileW(old_path_norm, new_path_norm, true);
    
    return _error_code(state);
}

void platform_file_memory_unmap(Platform_Memory_Mapping* mapping)
{
    if(mapping == NULL)
        return;

    HANDLE hFile = (HANDLE) mapping->state[0];
    HANDLE hMap = (HANDLE) mapping->state[1];
    LPVOID lpBasePtr = mapping->address;

    if(lpBasePtr != NULL)
        UnmapViewOfFile(lpBasePtr);
    if(hMap != NULL && hMap != INVALID_HANDLE_VALUE)
        CloseHandle(hMap);
    if(hFile != NULL && hFile != INVALID_HANDLE_VALUE)
        CloseHandle(hFile);
        
    memset(mapping, 0, sizeof *mapping);
}

Platform_Error platform_file_memory_map(Platform_String file_path, int64_t desired_size_or_zero, Platform_Memory_Mapping* mapping)
{
    memset(mapping, 0, sizeof *mapping);

    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap = INVALID_HANDLE_VALUE;
    LPVOID lpBasePtr = NULL;
    LARGE_INTEGER liFileSize;
    liFileSize.QuadPart = 0;

    DWORD disposition = 0;
    if(desired_size_or_zero == 0)
        disposition = OPEN_EXISTING;
    else
        disposition = OPEN_ALWAYS;
        
    const wchar_t* path = _ephemeral_path(file_path);
    hFile = CreateFileW(
        path,                            // lpFileName
        GENERIC_READ | GENERIC_WRITE,          // dwDesiredAccess
        FILE_SHARE_READ | FILE_SHARE_WRITE,    // dwShareMode
        NULL,                                  // lpSecurityAttributes
        disposition,                           // dwCreationDisposition
        FILE_ATTRIBUTE_NORMAL,                 // dwFlagsAndAttributes
        0);                                    // hTemplateFile

    if (hFile == INVALID_HANDLE_VALUE)
        goto error;

    if (!GetFileSizeEx(hFile, &liFileSize)) 
        goto error;

    //If the file is completely empty 
    // we dont perform any more operations
    // return a valid pointer and size of 0
    if (liFileSize.QuadPart == 0 && desired_size_or_zero == 0) 
    {
        CloseHandle(hFile);
        mapping->size = 0;
        mapping->address = NULL;
        return PLATFORM_ERROR_OK;
    }

    {
        LARGE_INTEGER desired_size = {0};
        if(desired_size_or_zero == 0)
            desired_size.QuadPart = liFileSize.QuadPart;
        if(desired_size_or_zero > 0)
        {
            desired_size.QuadPart = desired_size_or_zero;

            //if is desired smaller shrinks the file
            if(desired_size_or_zero < liFileSize.QuadPart)
            {
                DWORD dwPtrLow = SetFilePointer(hFile, desired_size.LowPart, &desired_size.HighPart,  FILE_BEGIN); 
                if(dwPtrLow == INVALID_SET_FILE_POINTER)
                    goto error;

                if(SetEndOfFile(hFile) == FALSE)
                    goto error;
            }
        }
        if(desired_size_or_zero < 0)
            desired_size.QuadPart = -desired_size_or_zero + liFileSize.QuadPart;

        hMap = CreateFileMappingW(
            hFile,
            NULL,                          // Mapping attributes
            PAGE_READWRITE ,               // Protection flags
            desired_size.HighPart,         // MaximumSizeHigh
            desired_size.LowPart,          // MaximumSizeLow
            NULL);                         // Name
        if (hMap == 0) 
            goto error;

        lpBasePtr = MapViewOfFile(
            hMap,
            FILE_MAP_ALL_ACCESS,   // dwDesiredAccess
            0,                     // dwFileOffsetHigh
            0,                     // dwFileOffsetLow
            0);                    // dwNumberOfBytesToMap
        if (lpBasePtr == NULL) 
            goto error;

        mapping->size = desired_size.QuadPart;
        mapping->address = lpBasePtr;
        mapping->state[0] = (uint64_t) hFile;
        mapping->state[1] = (uint64_t) hMap;
        return PLATFORM_ERROR_OK;
    }

    error: {
        DWORD err = GetLastError();
        if(hMap != INVALID_HANDLE_VALUE && hMap != 0)
            CloseHandle(hMap);
        if(hFile != INVALID_HANDLE_VALUE)
            CloseHandle(hFile);

        return err;
    }
}

static bool _is_file_link(const wchar_t* directory_path)
{
    HANDLE file = CreateFileW(directory_path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    size_t requiredSize = GetFinalPathNameByHandleW(file, NULL, 0, FILE_NAME_NORMALIZED);
    CloseHandle(file);

    return requiredSize == 0;
}

Platform_Error platform_file_info(Platform_String file_path, Platform_File_Info* out_info)
{    
    WIN32_FILE_ATTRIBUTE_DATA native_info = {0};
    Platform_File_Info info = {0};
    
    const wchar_t* path = _ephemeral_path(file_path);
    bool state = !!GetFileAttributesExW(path, GetFileExInfoStandard, &native_info);
        
        if(native_info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)
            info->is_link = _is_file_link(path);
    if(!state)
        return _error_code(state);
            
    info->created_epoch_time = _filetime_to_epoch_time(native_info.ftCreationTime);
    info->last_access_epoch_time = _filetime_to_epoch_time(native_info.ftLastAccessTime);
    info->last_write_epoch_time = _filetime_to_epoch_time(native_info.ftLastWriteTime);
    info->size = ((int64_t) native_info.nFileSizeHigh << 32) | ((int64_t) native_info.nFileSizeLow);
        
    if(native_info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        info->type = PLATFORM_FILE_TYPE_DIRECTORY;
    else
        info->type = PLATFORM_FILE_TYPE_FILE;

    if(out_info)
        *out_info = info;
    return _error_code(state);
}
    
//@TODO: create a cyclically overriden buffers that will get used for utf8 conversions
// this ensures that they will not ever have to get freed while periodally freeing them
//We have lets say 4 allocations slots. We allocate into the first then the second then the third and so on.
// once we run out of slots we allocate only if we need to grow. If we use a slot some number of times (2x lets say)
// we reallocate it to ensure that the space shrinks once not needed.

Platform_Error platform_directory_create(Platform_String dir_path)
{
    const wchar_t* path = _ephemeral_path(dir_path);
    bool state = !!CreateDirectoryW(path, NULL);
    return _error_code(state);
}
    
Platform_Error platform_directory_remove(Platform_String dir_path)
{
    const wchar_t* path = _ephemeral_path(dir_path);
    bool state = !!RemoveDirectoryW(path);
    return _error_code(state);
}

static char* _alloc_full_path(int normalize_flag, const wchar_t* local_path)
{
    (void) normalize_flag;
    wchar_t* full_path = NULL;

    int64_t needed_size = GetFullPathNameW(local_path, 0, NULL, NULL);
    if(needed_size > full_path.size)
    {
        buffer_resize(&full_path, needed_size);
        needed_size = GetFullPathNameW(local_path, (DWORD) needed_size, full_path, NULL);
    }
    
    char* out = _convert_to_utf8_normalize_path(normalize_flag, full_path, needed_size, NULL);
    buffer_deinit(&full_path);
    return out;
}

typedef struct Directory_Visitor
{
    WIN32_FIND_DATAW current_entry;
    HANDLE first_found;
    bool failed;
} Directory_Visitor;

#define WIO_FILE_MASK_ALL L"\\*.*"

static Directory_Visitor _directory_iterate_init(const wchar_t* dir_path, const wchar_t* file_mask)
{
    wchar_t* built_path = NULL;
    buffer_wconcat(&built_path, dir_path, file_mask, NULL);

    Directory_Visitor visitor = {0};
    assert(built_path.data != NULL);
    visitor.first_found = FindFirstFileW(built_path, &visitor.current_entry);
    while(visitor.failed == false && visitor.first_found != INVALID_HANDLE_VALUE)
    {
        if(wcscmp(visitor.current_entry.cFileName, L".") == 0
            || wcscmp(visitor.current_entry.cFileName, L"..") == 0)
            visitor.failed = !FindNextFileW(visitor.first_found, &visitor.current_entry);
        else
            break;
    }

    buffer_deinit(&built_path);
    return visitor;
}

static Platform_Error _directory_iterate_has(const Directory_Visitor* visitor)
{
    return visitor->first_found != INVALID_HANDLE_VALUE && visitor->failed == false;
}

static void _directory_iterate_next(Directory_Visitor* visitor)
{
    visitor->failed = visitor->failed || !FindNextFileW(visitor->first_found, &visitor->current_entry);
}

static void _directory_iterate_deinit(Directory_Visitor* visitor)
{
    FindClose(visitor->first_found);
}

void _platform_allocator_dealloc_string(char* str)
{
    if(str)
    {
        int64_t size = strlen(str);
        _platform_allocator_reallocate(0, str, size + 1);
    }
}

static Platform_Error _directory_list_contents_alloc(const wchar_t* directory_path, _Buffer* entries, int64_t max_depth)
{
    enum { MAX_RECURSION = 10000 };

    typedef struct Dir_Context {
        Directory_Visitor visitor;
        const wchar_t* path;    
        int64_t depth;          
        int64_t index;          
    } Dir_Context;

    Dir_Context first = {0};
    first.visitor = _directory_iterate_init(directory_path, WIO_FILE_MASK_ALL);
    first.path = directory_path;

    Platform_Error out_error = {0};
    if(_directory_iterate_has(&first.visitor) == false)
    {
        out_error = _error_code(false);
    }
    else
    {
        Dir_Context* stack = NULL;
        wchar_t* built_path = NULL;

        buffer_resize(&stack, )

        _buffer_push(&stack, &first, sizeof(first));

        int64_t i = 0;
        //While the queue is not empty iterate
        for(int64_t reading_from = 0; reading_from < stack.size; reading_from++)
        {
            for(;;)
            {
                Dir_Context* dir_context = (Dir_Context*) _buffer_at(&stack, reading_from);
                Directory_Visitor* visitor = &dir_context->visitor;
                if(_directory_iterate_has(visitor) == false)
                    break;

                //Build up our file path using the passed in 
                //  [path] and the file/foldername we just found: 
                _w_concat(dir_context->path, L"\\", visitor->current_entry.cFileName, &built_path);
                assert(built_path.data != 0);
        
                Platform_File_Info info = {0};
                info.created_epoch_time = _filetime_to_epoch_time(visitor->current_entry.ftCreationTime);
                info.last_access_epoch_time = _filetime_to_epoch_time(visitor->current_entry.ftLastAccessTime);
                info.last_write_epoch_time = _filetime_to_epoch_time(visitor->current_entry.ftLastWriteTime);
                info.size = ((int64_t) visitor->current_entry.nFileSizeHigh << 32) | ((int64_t) visitor->current_entry.nFileSizeLow);
        
                info.type = PLATFORM_FILE_TYPE_FILE;
                if(visitor->current_entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
                    info.type = PLATFORM_FILE_TYPE_DIRECTORY;
                else
                    info.type = PLATFORM_FILE_TYPE_FILE;

                if(info.is_link)
                    info.is_link = _is_file_link(_wstring(built_path));  

                int flag = IO_NORMALIZE_LINUX;
                if(info.type == PLATFORM_FILE_TYPE_DIRECTORY)
                    flag |= IO_NORMALIZE_DIRECTORY;
                else
                    flag |= IO_NORMALIZE_FILE;

                Platform_Directory_Entry entry = {0};
                entry.info = info;
                entry.path = _alloc_full_path(flag, _wstring(built_path));
                entry.directory_depth = dir_context->depth;
                _buffer_push(entries, &entry, sizeof(entry));

                bool recursion = dir_context->depth + 1 < max_depth || max_depth <= 0;
                if(info.type == PLATFORM_FILE_TYPE_DIRECTORY && info.is_link == false && recursion)
                {
                    _Buffer next_path = _buffer_make(sizeof(wchar_t));
                    _w_concat(_wstring(built_path), NULL, NULL, &next_path);

                    Dir_Context next = {0};
                    next.visitor = _directory_iterate_init(_wstring(next_path), WIO_FILE_MASK_ALL);
                    next.depth = dir_context->depth + 1;
                    next.path = _wstring(next_path);
                    assert(next.depth < MAX_RECURSION && "must not get stuck in an infinite loop");
                    _buffer_push(&stack, &next, sizeof(next));

                    //In case we reallocated
                    dir_context = (Dir_Context*) _buffer_at(&stack, reading_from);
                    visitor = &dir_context->visitor;
                }

                _directory_iterate_next(visitor); 
                dir_context->index++;
            }
        }

        for(int64_t i = 0; i < stack.size; i++)
        {
            Dir_Context* dir_context = (Dir_Context*) _buffer_at(&stack, i);
            if(dir_context->path != directory_path)
                _buffer_deinit(&dir_context->path);
            _directory_iterate_deinit(&dir_context->visitor);
        }
    
        //Null terminate the entries
        Platform_Directory_Entry terminator = {0};
        _buffer_push(entries, &terminator, sizeof(terminator));

        _buffer_deinit(&stack);
        _buffer_deinit(&built_path);
    
    }
    return _error_code(true);
}

Platform_Error platform_directory_list_contents_alloc(Platform_String directory_path, Platform_Directory_Entry** entries, int64_t* entries_count, int64_t max_depth)
{
    assert(entries != NULL && entries_count != NULL);
    _Buffer entries_stack = _buffer_make(sizeof(Platform_Directory_Entry));

    const wchar_t* path = _ephemeral_path(directory_path);
    Platform_Error error = _directory_list_contents_alloc(path, &entries_stack, max_depth);

    *entries = (Platform_Directory_Entry*) entries_stack.data;
    *entries_count = entries_stack.size > 0 ? entries_stack.size - 1 : 0;
    return error;
}

void platform_directory_list_contents_free(Platform_Directory_Entry* entries)
{
    if(entries == NULL)
        return;

    int64_t i = 0;
    for(; entries[i].path != NULL; i++)
        _platform_allocator_dealloc_string(entries[i].path);
          
    int64_t buffer_size = 1;
    while(buffer_size < i)
        buffer_size *= 2;
    _platform_allocator_reallocate(0, entries, buffer_size * sizeof(Platform_Directory_Entry));
}

Platform_Error platform_directory_set_current_working(Platform_String new_working_dir)
{
    const wchar_t* path = _ephemeral_path(new_working_dir);

    bool state = _wchdir(path) == 0;
    gp_state.current_working_generation += 1;
    return _error_code(state);
}

const char* platform_directory_get_current_working()
{
    if(gp_state.current_working_cached_generation != gp_state.current_working_generation)
    {
        wchar_t* current_working = _wgetcwd(NULL, 0);
        if(current_working == NULL)
        {
            free(current_working);
            gp_state.directory_current_working_cached = _alloc_full_path(IO_NORMALIZE_DIRECTORY, current_working);
        }

        gp_state.current_working_cached_generation = gp_state.current_working_generation;
    }

    return gp_state.directory_current_working_cached;
}

const char* platform_get_executable_path()
{
    if(gp_state.directory_executable_cached == NULL)
    {
        _Buffer wide = _buffer_make(sizeof(wchar_t));
        _BUFFER_INIT_BACKED(&wide, wchar_t, MAX_PATH);

        for(int64_t i = 0; i < 50; i++)
        {
            int64_t len = GetModuleFileNameW(NULL, _wstring(wide), (DWORD) wide.size);
            if(len < wide.size)
                break;

            _buffer_resize(&wide, wide.size * 2);
        }

        gp_state.directory_executable_cached = _alloc_full_path(IO_NORMALIZE_FILE, _wstring(wide));
        _buffer_deinit(&wide);
    }
    
    return gp_state.directory_executable_cached;
}

static void _platform_cached_directory_deinit()
{
    _platform_allocator_dealloc_string(gp_state.directory_executable_cached);
    _platform_allocator_dealloc_string(gp_state.directory_current_working_cached);
}

typedef bool (*_File_Watch_Func)(void* context);

typedef struct _Platform_File_Watch_Context {
    HANDLE watch_handle;
    bool (*user_func)(void* context);
    void* user_context;
} _Platform_File_Watch_Context;

#define WATCH_ALIGN 8

int _file_watch_func(void* context)
{
    _Platform_File_Watch_Context* watch_context = (_Platform_File_Watch_Context*) context;
    int return_state = 0;
    while(true)
    {
        DWORD wait_state = WaitForSingleObject(watch_context->watch_handle, INFINITE);
        if(wait_state == WAIT_OBJECT_0)
        {
            bool user_state = watch_context->user_func(watch_context->user_context);
            if(user_state == false)
            {
                return_state = 0;
                break;
            }
        }
        //something else happened. It shouldnt but it did. We exit the thread.
        else
        {
            assert(wait_state != WAIT_ABANDONED && "only happens for mutexes");
            assert(wait_state != WAIT_TIMEOUT && "we didnt set timeout");
            return_state = 1;
            break;
        }

        if(FindNextChangeNotification(watch_context->watch_handle) == false)
        {
            //Some error occured
            assert(false);
            return_state = 1;
            break;
        }
    }
    
    _platform_allocator_reallocate(0, watch_context, sizeof *watch_context);

    if(watch_context->watch_handle != INVALID_HANDLE_VALUE)
        FindCloseChangeNotification(watch_context->watch_handle);
    return return_state;
}

Platform_Error platform_file_watch(Platform_File_Watch* file_watch, Platform_String file_or_dir_path, int32_t file_wacht_flags, bool (*async_func)(void* context), void* context)
{
    platform_file_unwatch(file_watch);
    assert(async_func != NULL);
    
    HANDLE watch_handle = INVALID_HANDLE_VALUE;
    _Platform_File_Watch_Context* watch_context = (_Platform_File_Watch_Context*) _platform_allocator_reallocate(sizeof *watch_context, NULL, 0);
    if(watch_context != NULL)
    {
        BOOL watch_subtree = false;
        DWORD notify_filter = 0;

        if(file_wacht_flags & PLATFORM_FILE_WATCH_CHANGE)
            notify_filter |= FILE_NOTIFY_CHANGE_LAST_WRITE;
        
        if(file_wacht_flags & PLATFORM_FILE_WATCH_DIR_NAME)
            notify_filter |= FILE_NOTIFY_CHANGE_DIR_NAME;
        
        if(file_wacht_flags & PLATFORM_FILE_WATCH_FILE_NAME)
            notify_filter |= FILE_NOTIFY_CHANGE_FILE_NAME;
        
        if(file_wacht_flags & PLATFORM_FILE_WATCH_ATTRIBUTES)
            notify_filter |= FILE_NOTIFY_CHANGE_SIZE | FILE_NOTIFY_CHANGE_ATTRIBUTES;
        
        if(file_wacht_flags & PLATFORM_FILE_WATCH_RECURSIVE)
            watch_subtree = true;
        
        const wchar_t* path = _ephemeral_path(file_or_dir_path);
        watch_handle = FindFirstChangeNotificationW(path, watch_subtree, notify_filter);

        if(watch_handle != INVALID_HANDLE_VALUE)
        {
            watch_context->user_context = context;
            watch_context->user_func = async_func;
            watch_context->watch_handle = watch_handle;
            file_watch->thread = platform_thread_create(_file_watch_func, watch_context, 0);
            file_watch->data = (void*) watch_handle;
        }
    }

    return _error_code(watch_handle != INVALID_HANDLE_VALUE);
}
void platform_file_unwatch(Platform_File_Watch* file_watch)
{
    platform_thread_destroy(&file_watch->thread);
    FindCloseChangeNotification((HANDLE) file_watch->data);
}

//=========================================
// Window managmenet
//=========================================
Platform_Window_Popup_Controls platform_window_make_popup(Platform_Window_Popup_Style desired_style, Platform_String message, Platform_String title)
{
    int style = 0;
    int icon = 0;
    switch(desired_style)
    {
        case PLATFORM_POPUP_STYLE_OK:            style = MB_OK; break;
        case PLATFORM_POPUP_STYLE_ERROR:         style = MB_OK; icon = MB_ICONERROR; break;
        case PLATFORM_POPUP_STYLE_WARNING:       style = MB_OK; icon = MB_ICONWARNING; break;
        case PLATFORM_POPUP_STYLE_INFO:          style = MB_OK; icon = MB_ICONINFORMATION; break;
        case PLATFORM_POPUP_STYLE_RETRY_ABORT:   style = MB_ABORTRETRYIGNORE; icon = MB_ICONWARNING; break;
        case PLATFORM_POPUP_STYLE_YES_NO:        style = MB_YESNO; break;
        case PLATFORM_POPUP_STYLE_YES_NO_CANCEL: style = MB_YESNOCANCEL; break;
        default: style = MB_OK; break;
    }
    
    int value = 0;
    const wchar_t* message_wide = _ephemeral_wstring_convert(message, false);
    const wchar_t* title_wide = _ephemeral_wstring_convert(title, false);
    value = MessageBoxW(0, message_wide, title_wide, style | icon);

    switch(value)
    {
        case IDABORT: return PLATFORM_POPUP_CONTROL_ABORT;
        case IDCANCEL: return PLATFORM_POPUP_CONTROL_CANCEL;
        case IDCONTINUE: return PLATFORM_POPUP_CONTROL_CONTINUE;
        case IDIGNORE: return PLATFORM_POPUP_CONTROL_IGNORE;
        case IDYES: return PLATFORM_POPUP_CONTROL_YES;
        case IDNO: return PLATFORM_POPUP_CONTROL_NO;
        case IDOK: return PLATFORM_POPUP_CONTROL_OK;
        case IDRETRY: return PLATFORM_POPUP_CONTROL_RETRY;
        case IDTRYAGAIN: return PLATFORM_POPUP_CONTROL_RETRY;
        default: return PLATFORM_POPUP_CONTROL_OK;
    }
}

#include <stdint.h>
#include <stdbool.h>
#include <windows.h>
#include <Psapi.h>

#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "dbghelp.lib")

// Some versions of imagehlp.dll lack the proper packing directives themselves
// so we need to do it.
#pragma pack( push, before_imagehlp, 8 )
#include <imagehlp.h>
#pragma pack( pop, before_imagehlp )

int64_t platform_capture_call_stack(void** stack, int64_t stack_size, int64_t skip_count)
{
    if(stack_size <= 0)
        return 0;

    int64_t captured = CaptureStackBackTrace((DWORD) skip_count + 1, (DWORD) stack_size, stack, NULL);
    return captured;
}

#define MAX_MODULES 128 
#define MAX_NAME_LEN 2048



static void _platform_stack_trace_init(const char* search_path)
{
    if(gp_state.stack_trace_init)
        return;

    gp_state.stack_trace_process = GetCurrentProcess();
    if (!SymInitialize(gp_state.stack_trace_process, search_path, false)) 
    {
        assert(false);
        gp_state.stack_trace_error = GetLastError();
        return;
    }

    DWORD symOptions = SymGetOptions();
    symOptions |= SYMOPT_LOAD_LINES | SYMOPT_UNDNAME;
    SymSetOptions(symOptions);
    
    DWORD module_handles_size_needed = 0;
    HMODULE module_handles[MAX_MODULES] = {0};
    TCHAR module_filename[MAX_NAME_LEN] = {0};
    TCHAR module_name[MAX_NAME_LEN] = {0};
    EnumProcessModules(gp_state.stack_trace_process, module_handles, sizeof(module_handles), &module_handles_size_needed);
    
    DWORD module_count = module_handles_size_needed/sizeof(HMODULE);
    for(int64_t i = 0; i < module_count; i++)
    {
        HMODULE module_handle = module_handles[i];
        assert(module_handle != 0);
        MODULEINFO module_info = {0};
        GetModuleInformation(gp_state.stack_trace_process, module_handle, &module_info, sizeof(module_info));
        GetModuleFileNameExW(gp_state.stack_trace_process, module_handle, module_filename, sizeof(module_filename));
        GetModuleBaseNameW(gp_state.stack_trace_process, module_handle, module_name, sizeof(module_name));
        
        bool load_state = SymLoadModuleExW(gp_state.stack_trace_process, 0, module_filename, module_name, (DWORD64)module_info.lpBaseOfDll, (DWORD) module_info.SizeOfImage, 0, 0);
        if(load_state == false)
        {
            assert(false);
            gp_state.stack_trace_error = GetLastError();
        }
    }

    gp_state.stack_trace_init = true;
}

static void _platform_stack_trace_deinit()
{
    SymCleanup(gp_state.stack_trace_process);
}

void platform_translate_call_stack(Platform_Stack_Trace_Entry* tanslated, const void** stack, int64_t stack_size)
{
    if(stack_size == 0)
        return;

    _platform_stack_trace_init("");
    char symbol_info_data[sizeof(SYMBOL_INFO) + MAX_NAME_LEN + 1] = {0};

    DWORD offset_from_symbol = 0;
    IMAGEHLP_LINE64 line = {0};
    line.SizeOfStruct = sizeof line;

    Platform_Stack_Trace_Entry null_entry = {0};
    for(int64_t i = 0; i < stack_size; i++)
    {
        Platform_Stack_Trace_Entry* entry = tanslated + i;
        DWORD64 address = (DWORD64) stack[i];
        *entry = null_entry;

        if (address == 0)
            continue;

        memset(symbol_info_data, '\0', sizeof symbol_info_data);

        SYMBOL_INFO* symbol_info = (SYMBOL_INFO*) symbol_info_data;
        symbol_info->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol_info->MaxNameLen = MAX_NAME_LEN;
        DWORD64 displacement = 0;
        SymFromAddr(gp_state.stack_trace_process, address, &displacement, symbol_info);
            
        if (symbol_info->Name[0] != '\0')
        {
            UnDecorateSymbolName(symbol_info->Name, entry->function, sizeof entry->function, UNDNAME_COMPLETE);
        }
           
        IMAGEHLP_MODULE module_info = {0};
        module_info.SizeOfStruct = sizeof(IMAGEHLP_MODULE);
        bool module_info_state = SymGetModuleInfo64(gp_state.stack_trace_process, address, &module_info);
        if(module_info_state)
        {
            int64_t copy_size = sizeof module_info.ImageName;
            if(copy_size > sizeof entry->module - 1)
                copy_size = sizeof entry->module - 1;

            memmove(entry->module, module_info.ImageName, copy_size);
        }
            
        if (SymGetLineFromAddr64(gp_state.stack_trace_process, address, &offset_from_symbol, &line)) 
        {
            entry->line = line.LineNumber;
            
            int64_t copy_size = strlen(line.FileName);
            if(copy_size > sizeof entry->file - 1)
                copy_size = sizeof entry->file - 1;

            memmove(entry->file, line.FileName, copy_size);
        }
    }
}

static int64_t _platform_stack_trace_walk(CONTEXT context, HANDLE process, HANDLE thread, DWORD image_type, void** frames, int64_t frame_count, int64_t skip_count)
{
    STACKFRAME64 frame = {0};
    #ifdef _M_IX86
        DWORD native_image = IMAGE_FILE_MACHINE_I386;
        frame.AddrPC.Offset    = context.Eip;
        frame.AddrPC.Mode      = AddrModeFlat;
        frame.AddrFrame.Offset = context.Ebp;
        frame.AddrFrame.Mode   = AddrModeFlat;
        frame.AddrStack.Offset = context.Esp;
        frame.AddrStack.Mode   = AddrModeFlat;
    #elif _M_X64
        DWORD native_image = IMAGE_FILE_MACHINE_AMD64;
        frame.AddrPC.Offset    = context.Rip;
        frame.AddrPC.Mode      = AddrModeFlat;
        frame.AddrFrame.Offset = context.Rsp;
        frame.AddrFrame.Mode   = AddrModeFlat;
        frame.AddrStack.Offset = context.Rsp;
        frame.AddrStack.Mode   = AddrModeFlat;
    #elif _M_IA64
        DWORD native_image = IMAGE_FILE_MACHINE_IA64;
        frame.AddrPC.Offset    = context.StIIP;
        frame.AddrPC.Mode      = AddrModeFlat;
        frame.AddrFrame.Offset = context.IntSp;
        frame.AddrFrame.Mode   = AddrModeFlat;
        frame.AddrBStore.Offset= context.RsBSP;
        frame.AddrBStore.Mode  = AddrModeFlat;
        frame.AddrStack.Offset = context.IntSp;
        frame.AddrStack.Mode   = AddrModeFlat;
    #else
        #error "Unsupported platform"
    #endif

    if(image_type == 0)
        image_type = native_image; 
    
    (void) process;
    int64_t i = 0;
    for(; i < frame_count; i++)
    {
        bool ok = StackWalk64(native_image, gp_state.stack_trace_process, thread, &frame, &context, NULL, SymFunctionTableAccess64, SymGetModuleBase64, NULL);
        if (ok == false)
            break;

        if(skip_count > 0)
        {
            skip_count--;
            i --;
            continue;
        }

        frames[i] = (void*) frame.AddrPC.Offset;
    }
    
    return i;
}

typedef void (*Sandbox_Error_Func)(void* context, Platform_Sandox_Error error_code);

__declspec(thread) Sandbox_Error_Func sandbox_error_func;
__declspec(thread) void* sandbox_error_context;
__declspec(thread) Platform_Sandox_Error sanbox_error_code;
__declspec(thread) bool sandbox_is_signal_handler_set = false;

void platform_abort()
{
    RaiseException(PLATFORM_EXCEPTION_ABORT, 0, 0, NULL);
}

void platform_terminate()
{
    RaiseException(PLATFORM_EXCEPTION_TERMINATE, 0, 0, NULL);
}

LONG WINAPI _sanbox_exception_filter(EXCEPTION_POINTERS * ExceptionInfo)
{
    Platform_Sandox_Error error = PLATFORM_EXCEPTION_NONE;
    switch(ExceptionInfo->ExceptionRecord->ExceptionCode)
    {
        case EXCEPTION_ACCESS_VIOLATION:
            error = PLATFORM_EXCEPTION_ACCESS_VIOLATION;
            break;
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
            error = PLATFORM_EXCEPTION_ACCESS_VIOLATION;
            break;
        case EXCEPTION_BREAKPOINT:
            error = PLATFORM_EXCEPTION_BREAKPOINT;
            break;
        case EXCEPTION_DATATYPE_MISALIGNMENT:
            error = PLATFORM_EXCEPTION_DATATYPE_MISALIGNMENT;
            break;
        case EXCEPTION_FLT_DENORMAL_OPERAND:
            error = PLATFORM_EXCEPTION_FLOAT_DENORMAL_OPERAND;
            break;
        case EXCEPTION_FLT_DIVIDE_BY_ZERO:
            error = PLATFORM_EXCEPTION_FLOAT_DIVIDE_BY_ZERO;
            break;
        case EXCEPTION_FLT_INEXACT_RESULT:
            error = PLATFORM_EXCEPTION_FLOAT_INEXACT_RESULT;
            break;
        case EXCEPTION_FLT_INVALID_OPERATION:
            error = PLATFORM_EXCEPTION_FLOAT_INVALID_OPERATION;
            break;
        case EXCEPTION_FLT_OVERFLOW:
            error = PLATFORM_EXCEPTION_FLOAT_OVERFLOW;
            break;
        case EXCEPTION_FLT_STACK_CHECK:
            error = PLATFORM_EXCEPTION_STACK_OVERFLOW;
            break;
        case EXCEPTION_FLT_UNDERFLOW:
            error = PLATFORM_EXCEPTION_FLOAT_UNDERFLOW;
            break;
        case EXCEPTION_ILLEGAL_INSTRUCTION:
            error = PLATFORM_EXCEPTION_ILLEGAL_INSTRUCTION;
            break;
        case EXCEPTION_IN_PAGE_ERROR:
            error = PLATFORM_EXCEPTION_PAGE_ERROR;
            break;
        case EXCEPTION_INT_DIVIDE_BY_ZERO:
            error = PLATFORM_EXCEPTION_INT_DIVIDE_BY_ZERO;
            break;
        case EXCEPTION_INT_OVERFLOW:
            error = PLATFORM_EXCEPTION_INT_OVERFLOW;
            break;
        case EXCEPTION_INVALID_DISPOSITION:
            error = PLATFORM_EXCEPTION_OTHER;
            break;
        case EXCEPTION_NONCONTINUABLE_EXCEPTION:
            error = PLATFORM_EXCEPTION_OTHER;
            break;
        case EXCEPTION_PRIV_INSTRUCTION:
            error = PLATFORM_EXCEPTION_PRIVILAGED_INSTRUCTION;
            break;
        case EXCEPTION_SINGLE_STEP:
            error = PLATFORM_EXCEPTION_BREAKPOINT_SINGLE_STEP;
            break;
        case EXCEPTION_STACK_OVERFLOW:
            error = PLATFORM_EXCEPTION_STACK_OVERFLOW;
            break;
        case PLATFORM_EXCEPTION_ABORT:
            error = PLATFORM_EXCEPTION_ABORT;
            break;
        case PLATFORM_EXCEPTION_TERMINATE:
            error = PLATFORM_EXCEPTION_TERMINATE;
            break;
        default:
            error = PLATFORM_EXCEPTION_OTHER;
            break;
    }

    sanbox_error_code = error;
    if(sandbox_error_func != NULL && error != PLATFORM_EXCEPTION_STACK_OVERFLOW)
        sandbox_error_func(sandbox_error_context, sanbox_error_code);
    return EXCEPTION_EXECUTE_HANDLER;
}

Platform_Sandox_Error platform_exception_sandbox(
    void (*sandboxed_func)(void* context),   
    void* sandbox_context,
    void (*error_func)(void* context, Platform_Sandox_Error error_code),   
    void* error_context
)
{
    if(sandbox_is_signal_handler_set == false)
    {
        sandbox_is_signal_handler_set = true;
        SetUnhandledExceptionFilter(_sanbox_exception_filter);
        AddVectoredExceptionHandler(1, _sanbox_exception_filter);
        SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOALIGNMENTFAULTEXCEPT | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    }
        
    Sandbox_Error_Func prev_func = sandbox_error_func;
    void* prev_context = sandbox_error_context;
    
    sandbox_error_func = error_func;
    sandbox_error_context = error_context;
    __try 
    { 
        sandboxed_func(sandbox_context);
    } 
    __except(1)
    { 
        sandbox_error_func = prev_func;
        sandbox_error_context = prev_context;
        return sanbox_error_code;
    }

    sandbox_error_func = prev_func;
    sandbox_error_context = prev_context;
    return PLATFORM_EXCEPTION_NONE;
}

const char* platform_sandbox_error_to_string(Platform_Sandox_Error error)
{
    switch(error)
    {
        case PLATFORM_EXCEPTION_NONE: return "PLATFORM_EXCEPTION_NONE";
        case PLATFORM_EXCEPTION_ACCESS_VIOLATION: return "PLATFORM_EXCEPTION_ACCESS_VIOLATION";
        case PLATFORM_EXCEPTION_DATATYPE_MISALIGNMENT: return "PLATFORM_EXCEPTION_DATATYPE_MISALIGNMENT";
        case PLATFORM_EXCEPTION_FLOAT_DENORMAL_OPERAND: return "PLATFORM_EXCEPTION_FLOAT_DENORMAL_OPERAND";
        case PLATFORM_EXCEPTION_FLOAT_DIVIDE_BY_ZERO: return "PLATFORM_EXCEPTION_FLOAT_DIVIDE_BY_ZERO";
        case PLATFORM_EXCEPTION_FLOAT_INEXACT_RESULT: return "PLATFORM_EXCEPTION_FLOAT_INEXACT_RESULT";
        case PLATFORM_EXCEPTION_FLOAT_INVALID_OPERATION: return "PLATFORM_EXCEPTION_FLOAT_INVALID_OPERATION";
        case PLATFORM_EXCEPTION_FLOAT_OVERFLOW: return "PLATFORM_EXCEPTION_FLOAT_OVERFLOW";
        case PLATFORM_EXCEPTION_FLOAT_UNDERFLOW: return "PLATFORM_EXCEPTION_FLOAT_UNDERFLOW";
        case PLATFORM_EXCEPTION_FLOAT_OTHER: return "PLATFORM_EXCEPTION_FLOAT_OTHER";
        case PLATFORM_EXCEPTION_PAGE_ERROR: return "PLATFORM_EXCEPTION_PAGE_ERROR";
        case PLATFORM_EXCEPTION_INT_DIVIDE_BY_ZERO: return "PLATFORM_EXCEPTION_INT_DIVIDE_BY_ZERO";
        case PLATFORM_EXCEPTION_INT_OVERFLOW: return "PLATFORM_EXCEPTION_INT_OVERFLOW";
        case PLATFORM_EXCEPTION_ILLEGAL_INSTRUCTION: return "PLATFORM_EXCEPTION_ILLEGAL_INSTRUCTION";
        case PLATFORM_EXCEPTION_PRIVILAGED_INSTRUCTION: return "PLATFORM_EXCEPTION_PRIVILAGED_INSTRUCTION";
        case PLATFORM_EXCEPTION_BREAKPOINT: return "PLATFORM_EXCEPTION_BREAKPOINT";
        case PLATFORM_EXCEPTION_BREAKPOINT_SINGLE_STEP: return "PLATFORM_EXCEPTION_BREAKPOINT_SINGLE_STEP";
        case PLATFORM_EXCEPTION_STACK_OVERFLOW: return "PLATFORM_EXCEPTION_STACK_OVERFLOW";
        case PLATFORM_EXCEPTION_ABORT: return "PLATFORM_EXCEPTION_ABORT";
        case PLATFORM_EXCEPTION_TERMINATE: return "PLATFORM_EXCEPTION_TERMINATE";
        case PLATFORM_EXCEPTION_OTHER: return "PLATFORM_EXCEPTION_OTHER";
        default:
            return "PLATFORM_EXCEPTION_OTHER";
    }
}

bool _platform_set_console_output_escape_sequences()
{
    // Set output mode to handle virtual terminal sequences
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE)
        return false;

    DWORD dwOriginalOutMode = 0;
    if (!GetConsoleMode(hOut, &dwOriginalOutMode))
        return false;

    DWORD dwOutMode = dwOriginalOutMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING | DISABLE_NEWLINE_AUTO_RETURN;
    if (!SetConsoleMode(hOut, dwOutMode))
    {
        dwOutMode = dwOriginalOutMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        if (!SetConsoleMode(hOut, dwOutMode))
            return false;
    }

    return true;
}

void _platform_set_console_utf8()
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    setlocale(LC_ALL, ".UTF-8");
}

void platform_init(Platform_Allocator* allocator_or_null)
{
    platform_deinit();

    if(allocator_or_null)
        gp_state.allocator = *allocator_or_null;

    (void) allocator_or_null;
    platform_perf_counter();
    platform_startup_epoch_time();
    platform_perf_counter_startup();

    _platform_set_console_utf8();
    _platform_set_console_output_escape_sequences();
}
void platform_deinit()
{
    _translated_deinit_all();
    _ephemeral_wstring_deinit_all();
    _platform_cached_directory_deinit();
    _platform_stack_trace_deinit();
    memset(&gp_state, 0, sizeof(gp_state));
}