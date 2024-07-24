# include <stdio.h>
# include <stdlib.h>
# include <stdint.h>
# include <string.h>


# define IMG_MAGIC_NUMBER 2051
# define LABEL_MAGIC_NUMBER 2049

typedef struct {
    unsigned char **images; // pointer to an array of pointers
    unsigned char *labels; // pointer to an array of unsigned char
    int num_items;
    int n_rows;
    int n_cols;
    unsigned char *sample_train_image; // Store a sample training image
    unsigned char *sample_test_image;  // Store a sample test image
    unsigned char sample_train_label;  // Store the label of the sample training image
    unsigned char sample_test_label;   // Store the label of the sample test image
} MNIST_DATASET;

/* 
apparently mnist data is written in big-endian notation
 so we need to reverse the order of bytes.
 basically the '&' operation masks the value to shift each byte
 in the right place and the '|' operation returns nonzero binary
 values to reconstruct the the bytes sequence 
 */

int reverseBytes(int value) {
    unsigned char b0 = (value & 0x000000ff) << 24u;
    unsigned char b1 = (value & 0x0000ff00) << 8u;
    unsigned char b2 = (value & 0x00ff0000) >> 8u;
    unsigned char b3 = (value & 0xff000000) >> 24u;
    return (int)b0 | (int)b1 | (int)b2 | (int)b3;
}

/*
loading the images, we need to first check the magic number,
n of columns, n of rows and n of images
*/

void loadImages(const char* filename, MNIST_DATASET *dataset) {
    printf("loading images\n");

    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("File is fukked.\n");
        exit(1);
    }

    uint8_t buffer[16];
    size_t bytesRead = fread(buffer, 1, sizeof(buffer), file);
    printf("First %zu bytes: ", bytesRead);
    for (size_t i = 0; i < bytesRead; i++) {
        printf("%02X ", buffer[i]);
    }
    printf("\n");
    fseek(file, 0, SEEK_SET); // Reset file pointer to the beginning

    // read first four bits one by one to get the metadata of the dataset
    // int magic_number = 0;
    // fread(&magic_number, sizeof(int), 1, file);
    // printf("magic number before reversing bits: %d\n", magic_number);
    // magic_number = reverseBytes(magic_number);
    // printf("magic number after reversing bits: %d\n", magic_number);

    // if (magic_number != IMG_MAGIC_NUMBER) {
    //     printf("Invalid magic number; does not correspond to image file\n");
    //     exit(1);
    // }
    uint32_t magic_number = 0;
    uint8_t byte;
    for (int i = 0; i < 4; i++) {
        fread(&byte, sizeof(uint8_t), 1, file);
        magic_number = (magic_number << 8) | byte;
    }
    printf("magic number: %u\n", magic_number);

    if (magic_number != IMG_MAGIC_NUMBER) {
        printf("Invalid magic number; does not correspond to image file\n");
        exit(1);
    }

    fread(&dataset->num_items, sizeof(int), 1, file); // the -> operator 'points' to a member of the struct
    dataset->num_items = reverseBytes(dataset->num_items);

    fread(&dataset->n_rows, sizeof(int), 1, file);
    dataset->n_rows = reverseBytes(dataset->n_rows);

    fread(&dataset->n_cols, sizeof(int), 1, file);
    dataset->n_cols = reverseBytes(dataset->n_cols);

    // this allocates memory for the pointers that later should point to the actual memory for the images
    dataset->images = (unsigned char **)malloc(dataset->num_items * sizeof(unsigned char *));
    for (int i = 0; i < dataset->num_items; ++i) {
        // allocate memory for each image
        dataset->images[i] = (unsigned char *)malloc(dataset->n_rows * dataset->n_cols * sizeof(unsigned char));
        // read in the image into the array
        fread(dataset->images[i], sizeof(unsigned char), dataset->n_rows * dataset->n_cols, file);
    }

    fclose(file);

    // After loading all images
    if (dataset->sample_train_image == NULL && strcmp(filename, "data/train-images-idx3-ubyte") == 0) {
        dataset->sample_train_image = (unsigned char *)malloc(dataset->n_rows * dataset->n_cols * sizeof(unsigned char));
        memcpy(dataset->sample_train_image, dataset->images[0], dataset->n_rows * dataset->n_cols * sizeof(unsigned char));
    } else if (dataset->sample_test_image == NULL && strcmp(filename, "data/t10k-images-idx3-ubyte") == 0) {
        dataset->sample_test_image = (unsigned char *)malloc(dataset->n_rows * dataset->n_cols * sizeof(unsigned char));
        memcpy(dataset->sample_test_image, dataset->images[0], dataset->n_rows * dataset->n_cols * sizeof(unsigned char));
    }
}


void loadLabels(const char* filename, MNIST_DATASET *dataset) {
    printf("loading labels\n");

    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("File is fukked.\n");
        exit(1);
    }
    // int magic_number = 0;
    // fread(&magic_number, sizeof(int), 1, file);
    // magic_number = reverseBytes(magic_number);
    // if (magic_number != LABEL_MAGIC_NUMBER) {
    //     printf("Invalid magic number; does not correspond to labels file\n");
    //     exit(1);
    // }

    uint32_t magic_number = 0;
    uint8_t byte;
    for (int i = 0; i < 4; i++) {
        fread(&byte, sizeof(uint8_t), 1, file);
        magic_number = (magic_number << 8) | byte;
    }
    printf("magic number: %u\n", magic_number);

    if (magic_number != LABEL_MAGIC_NUMBER) {
        printf("Invalid magic number; does not correspond to image file\n");
        exit(1);
    }

    int num_items = 0;
    fread(&num_items, sizeof(int), 1, file);
    num_items = reverseBytes(num_items);

    if (dataset->num_items != num_items) {
        printf("n of images and labels don't match\n");
        exit(1);
    }

    dataset->labels = (unsigned char *)malloc(num_items * sizeof(unsigned char));
    fread(&dataset->labels, sizeof(unsigned char), num_items, file);
    
    fclose(file);

    // After loading all labels
    if (strcmp(filename, "data/train-labels-idx3-ubyte") == 0) {
        dataset->sample_train_label = dataset->labels[0];
    } else if (strcmp(filename, "data/t10k-labels-idx3-ubyte") == 0) {
        dataset->sample_test_label = dataset->labels[0];
    }
}


void freeDataset(MNIST_DATASET *dataset) {
    // loop over all images in dataset and free
    for (int i = 0; i < dataset->num_items; ++i) {
        free(dataset->images[i]);
    }
    free(dataset->images);
    free(dataset->labels);
}


void printImage(unsigned char *image, int rows, int cols) {
    const char *intensity = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
    int intensityLevels = strlen(intensity);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            unsigned char pixel = image[i * cols + j];
            int index = pixel * (intensityLevels - 1) / 255;
            printf("%c", intensity[index]);
            printf("%c", intensity[index]); // Print each character twice for a less stretched look
        }
        printf("\n");
    }
}


int main() {
    MNIST_DATASET training_data = {0};
    MNIST_DATASET test_data = {0};

    loadImages("data/train-images-idx3-ubyte", &training_data);
    loadLabels("data/train-labels-idx1-ubyte", &training_data);
    loadImages("data/t10k-images-idx3-ubyte", &test_data);
    loadLabels("data/t10k-labels-idx1-ubyte", &test_data);

    printf("Sample training image (label: %d):\n", training_data.sample_train_label);
    printImage(training_data.sample_train_image, training_data.n_rows, training_data.n_cols);

    printf("\nSample test image (label: %d):\n", test_data.sample_test_label);
    printImage(test_data.sample_test_image, test_data.n_rows, test_data.n_cols);

    // Free allocated memory
    freeDataset(&training_data);
    freeDataset(&test_data);

    return 0;
}
