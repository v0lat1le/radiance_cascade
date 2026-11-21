#pragma once

#ifndef __SLANG__
#include "NoGraphicsAPI/types.h"
#endif

struct CascadeData {
    float ray_start;
    float ray_end;
    float ray_step;
    float angle_delta;
    float ray_count_inv;
    uint32_t texture_indices;
    uint32_t padding[2];
};
