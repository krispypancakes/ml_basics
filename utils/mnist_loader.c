# include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


/* apparently mnist data is written in big-endian notation
 so we need to reverse the order of bytes */ 
uint32_t reverseBytes(uint32_t value) {
    uint32_t b0 = (value & 0x000000ff) << 24u;
    uint32_t b1 = (value & 0x0000ff00) << 8u;
    uint32_t b2 = (value & 0x00ff0000) >> 8u;
    uint32_t b3 = (value & 0xff000000) >> 24u;
    return b0 | b1 | b2 | b3;
}