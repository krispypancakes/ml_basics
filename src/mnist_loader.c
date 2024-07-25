// this makes sure the actual implementations of the functions are included as well
#define STB_IMAGE_WRITE_IMPLEMENTATION 
#include "stb_image_write.h"
# include <stdio.h>
# include <stdlib.h>
# include <stdint.h>
# include <string.h>


# define IMG_MAGIC_NUMBER 2051
# define LABEL_MAGIC_NUMBER 2049

typedef struct {
    unsigned char **images; // pointer to an array of pointers
    unsigned char *labels; // pointer to an array of unsigned char
    uint32_t n_items;
    uint32_t n_rows;
    uint32_t n_cols;
} MNIST_DATASET;

/* 
apparently mnist data is written in big-endian notation
 so we need to reverse the order of bytes.
 basically the '&' operation masks the value to shift each byte
 in the right place and the '|' operation returns nonzero binary
 values to reconstruct the the bytes sequence 
 */

uint32_t reverseBytes(uint32_t value) {
    uint32_t b0 = (value & 0x000000ff) << 24;
    uint32_t b1 = (value & 0x0000ff00) << 8;
    uint32_t b2 = (value & 0x00ff0000) >> 8;
    uint32_t b3 = (value & 0xff000000) >> 24;
    return b0 | b1 | b2 | b3;
}

/*
loading the images, we need to first check the magic number,
n of columns, n of rows and n of images
*/

void loadImages(const char* filename, MNIST_DATASET *dataset) {
    printf("loading images from: %s\n", filename);

    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("File is fukked.\n");
        exit(1);
    }

    uint32_t magic_number = 0;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    printf("magic number before byte reversal: %u\n", magic_number);
    magic_number = reverseBytes(magic_number);
    printf("magic number after byte reversal: %u\n", magic_number);


    if (magic_number != IMG_MAGIC_NUMBER) {
        printf("Invalid magic number; does not correspond to image file\n");
        exit(1);
    }

    uint32_t num_images = 0, num_rows = 0, num_cols = 0;

    fread(&num_images, sizeof(int), 1, file); // the -> operator 'points' to a member of the struct
    printf("number of images before reversal: %u\n", num_images);
    num_images = reverseBytes(num_images);
    printf("number of images after reversal: %u\n", num_images);

    fread(&num_rows, sizeof(int), 1, file);
    num_rows = reverseBytes(num_rows);

    fread(&num_cols, sizeof(int), 1, file);
    num_cols = reverseBytes(num_cols);

    dataset->n_items = num_images;
    dataset->n_rows = num_rows;
    dataset->n_cols = num_cols;

    // this allocates memory for the pointers that later should point to the actual memory for the images
    dataset->images = (unsigned char **)malloc(dataset->n_items * sizeof(unsigned char *));
    for (int i = 0; i < dataset->n_items; ++i) {
        // allocate memory for each image, dataset->images[i] is the i-th pointer in the array of pointers, each the beginning of an image array
        dataset->images[i] = (unsigned char *)malloc(dataset->n_rows * dataset->n_cols * sizeof(unsigned char));
        // read in the image into the array
        fread(dataset->images[i], sizeof(unsigned char), dataset->n_rows * dataset->n_cols, file);
    }

    fclose(file);
}


void loadLabels(const char* filename, MNIST_DATASET *dataset) {
    printf("loading labels from: %s\n", filename);

    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("File is fukked.\n");
        exit(1);
    }

    uint32_t magic_number = 0;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = reverseBytes(magic_number);
    printf("magic number: %u\n", magic_number);

    if (magic_number != LABEL_MAGIC_NUMBER) {
        printf("Invalid magic number; does not correspond to image file\n");
        exit(1);
    }

    u_int32_t num_labels = 0;
    fread(&num_labels, sizeof(int), 1, file);
    num_labels = reverseBytes(num_labels);
    printf("number of labels in the file: %d\n", num_labels);

    if (dataset->n_items != num_labels) {
        printf("n of images and labels don't match\n");
        exit(1);
    }

    unsigned char* labels;
    labels = (unsigned char *)malloc(num_labels * sizeof(unsigned char));
    fread(labels, sizeof(unsigned char), num_labels, file);
    dataset->labels = labels;

    printf("loaded %d labels\n", num_labels);

    fclose(file);

}


void freeDataset(MNIST_DATASET *dataset) {
    // loop over all images in dataset and free
    for (int i = 0; i < dataset->n_items; ++i) {
        free(dataset->images[i]);
    }
    free(dataset->images);
    free(dataset->labels);
}


void saveImageAsPNG(unsigned char *image, int rows, int cols, const char *filename) {
    stbi_write_png(filename, cols, rows, 1, image, cols);
    printf("Saved image to %s\n", filename);
}


int main() {
    MNIST_DATASET training_data = {0};
    MNIST_DATASET test_data = {0};

    loadImages("data/train-images-idx3-ubyte", &training_data);
    loadLabels("data/train-labels-idx1-ubyte", &training_data);
    loadImages("data/t10k-images-idx3-ubyte", &test_data);
    loadLabels("data/t10k-labels-idx1-ubyte", &test_data);

    printf("Loaded %d training images and %d test images.\n", training_data.n_items, test_data.n_items);

    // Save sample training images
    for (int i = 0; i < 3 && i < training_data.n_items; i++) {
        char filename[100];
        snprintf(filename, sizeof(filename), "data/train_sample_%d_image_%d.png", i, training_data.labels[i]);
        saveImageAsPNG(training_data.images[i], training_data.n_rows, training_data.n_cols, filename);
    }
    // Save sample test images
    for (int i = 0; i < 3 && i < test_data.n_items; i++) {
        char filename[100];
        snprintf(filename, sizeof(filename), "data/test_sample_%d_image_%d.png", i, test_data.labels[i]);
        saveImageAsPNG(test_data.images[i], test_data.n_rows, test_data.n_cols, filename);
    }
    // Free allocated memory
    freeDataset(&training_data);
    freeDataset(&test_data);

    return 0;
}
