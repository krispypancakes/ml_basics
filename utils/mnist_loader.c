# include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


/* 
apparently mnist data is written in big-endian notation
 so we need to reverse the order of bytes.
 basically the '&' operation masks the value to shift each byte
 in the right place and the '|' operation returns nonzero binary
 values to reconstruct the the bytes sequence 
 */
uint32_t reverseBytes(uint32_t value) {
    // move byte 0 to position 3 - 24 bits to the left and so on
    uint32_t b0 = (value & 0x000000ff) << 24u;
    uint32_t b1 = (value & 0x0000ff00) << 8u;
    uint32_t b2 = (value & 0x00ff0000) >> 8u;
    uint32_t b3 = (value & 0xff000000) >> 24u;
    return b0 | b1 | b2 | b3;
}

/*
loading the images, we need to first check the magic number,
n of columns, n of rows and n of images
*/
unsigned char* loadMNISTImages(const char* filename, uint32_t* numberOfImages, uint32_t* rows, uint32_t* cols) {
    
}