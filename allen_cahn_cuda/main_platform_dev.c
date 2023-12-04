
#define JOT_ALL_IMPL

#include "lib/platform.h"
#include "lib/assert.h"
#include "lib/array.h"
#include <stdio.h>


void deep_func1()
{
    log_callstack("app", LOG_TYPE_WARN, -1, 0);
}

void deep_func2()
{
    deep_func1();
}

void deep_func3()
{
    deep_func2();
}

#define _SECOND_MILLISECONDS    ((int64_t) 1000LL)
#define _SECOND_MICROSECS       ((int64_t) 1000000LL)
#define _SECOND_NANOSECS        ((int64_t) 1000000000LL)

typedef long long lli;
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <limits.h>
int main()
{
    platform_init();
    puts("hi");

    deep_func3();

    u8_Array arr = {0};
    array_deinit(&arr);

    platform_deinit();
}