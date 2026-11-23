#pragma once

#ifndef __SLANG__
#include "NoGraphicsAPI/types.h"
#endif

struct CascadeData {
    uint32_t texture_indices;
    uint16_t cones_count;
    uint16_t padding1;
    uint32_t padding2[2];
};
