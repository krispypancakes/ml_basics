# include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


# define IMG_MAGIC_NUMBER 2051
# define LABEL_MAGIC_NUMBER 2049

typedef struct {
    unsigned char **images; // pointer to an array of pointers
    unsigned char *labels; // pointer to an array of unsigned char
    int num_items;
    int n_rows;
    int n_cols;
} MNIST_DATASET;

/* 
apparently mnist data is written in big-endian notation
 so we need to reverse the order of bytes.
 basically the '&' operation masks the value to shift each byte
 in the right place and the '|' operation returns nonzero binary
 values to reconstruct the the bytes sequence 
 */

int reverseBytes(int value) {
    // move byte 0 to position 3 - 24 bits to the left and so on
    unsigned char b0 = (value & 0x000000ff) << 24u;
    unsigned char b1 = (value & 0x0000ff00) << 8u;
    unsigned char b2 = (value & 0x00ff0000) >> 8u;
    unsigned char b3 = (value & 0xff000000) >> 24u;
    return b0 | b1 | b2 | b3;
}

/*
loading the images, we need to first check the magic number,
n of columns, n of rows and n of images
*/

void loadImages(const char* filename, MNIST_DATASET *dataset) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("File is fukked.");
        exit(1);
    }

    // read first four bits one by one to get the metadata of the dataset
    int magic_number = 0;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = reverseBytes(magic_number);

    if (magic_number != IMG_MAGIC_NUMBER) {
        printf("Invalid magic number; does not correspond to image file");
        exit(1);
    }

    fread(&dataset->num_items, sizeof(int), 1, file); // the -> operator 'points' to a member of the struct
    dataset->num_items = reverseBytes(dataset->num_items);

    fread(&dataset->n_rows, sizeof(int), 1, file);
    dataset->n_rows = reverseBytes(dataset->n_rows);

    fread(&dataset->n_cols, sizeof(int), 1, file);
    dataset->n_cols = reverseBytes(dataset->n_cols);

    dataset->images = (unsigned char **)malloc(dataset->num_items * sizeof(unsigned char *));
    for (int i = 0; i < dataset->num_items; ++i) {
        // allocate memory for each image
        dataset->images[i] = (unsigned char *)malloc(dataset->n_rows * dataset->n_cols * sizeof(unsigned char));
        // read in the image into the array
        fread(dataset->images[i], sizeof(unsigned char), dataset->n_rows * dataset->n_cols, file);
    }

    fclose(file);
}

void loadLabels() {

}

int main() {
    MNIST_DATASET training_data;
    MNIST_DATASET test_data;
}
