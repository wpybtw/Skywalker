#include "roller.cuh"
#include "kernel.cuh"
#include "sampler.cuh"
#include "alias_table.cuh"
#include "sampler_result.cuh"
#include "util.cuh"

// #include <cooperative_groups.h>
// #include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

DECLARE_bool(debug);
DECLARE_bool(v);
DECLARE_double(tp);
DECLARE_bool(printresult);
DECLARE_int32(m);
DECLARE_bool(peritr);

DECLARE_bool(static);
DECLARE_bool(buffer);