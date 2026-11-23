#pragma once

#ifndef __SLANG__
#include "NoGraphicsAPI/types.h"
#endif

struct CascadeData {
    uint32_t texture_indices;
    uint16_t cones_count;
    uint16_t width1;
    uint16_t height1;
    uint16_t width2;
    uint16_t height2;
    uint16_t padding;
};
