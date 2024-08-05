#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdint.h>

typedef struct {
    unsigned char **images;
    unsigned char *labels;
    uint32_t n_items;
    uint32_t n_rows;
    uint32_t n_cols;
} MNIST_DATASET;

void loadImages(const char* filename, MNIST_DATASET *dataset);
void loadLabels(const char* filename, MNIST_DATASET *dataset);
void freeDataset(MNIST_DATASET *dataset);

#endif
