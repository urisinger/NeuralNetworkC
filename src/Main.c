#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "Layer.h"

double sigmoid(double x) {
	return ((tanh(x)));
}

double sigmoidder(double x) {
    double tah = tanh(x);
	return ((1-tah*tah));
}

int ChangeEndianness(int value) {
    int result = 0;
    result |= (value & 0x000000FF) << 24;
    result |= (value & 0x0000FF00) << 8;
    result |= (value & 0x00FF0000) >> 8;
    result |= (value & 0xFF000000) >> 24;
    return result;
}



void main()
{

    FILE* imageTrainFiles = fopen("../../data/train-images.idx3-ubyte", "rb");
    FILE* imageTrainLabel = fopen("../../data/train-labels.idx1-ubyte", "rb");

    if (imageTrainFiles == NULL) {
        perror("File Not Found");
        return 1;
    }

    int magic_number;
    fread(&magic_number, sizeof(int), 1, imageTrainFiles);

    if (ChangeEndianness(magic_number) != 2051) {
        printf("Invalid magic number : %d\n", ChangeEndianness(magic_number));
        return;
    }

    srand(time(NULL));
    Matrix* sample = NewMat(1000, 784);
    Matrix* label = NewMat(1000, 10);

    GetSample(sample, imageTrainFiles, 16);  // Read the first image starting at offset 16

    GetLabel(label, imageTrainLabel, 8);

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (sample->vals[7][i * 28 + j])
                printf("%1.3f|", sample->vals[7][i * 28 + j]);
            else
                printf("     |");
        }
        printf("\n");
    }

    Layer* HeadLayer = NewNetwork(NewVec(784) ,16, sigmoid, sigmoidder);

	NewTailLayer(HeadLayer, 16, sigmoid, sigmoidder);

    NewTailLayer(HeadLayer, 16, sigmoid, sigmoidder);

    NewTailLayer(HeadLayer, 10, sigmoid, sigmoidder);

	Vector* output = NewVec(1);

	for (int i = 0; i < 100; ++i) {
        LearnSample(HeadLayer,sample,label,0.1);
    }


    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (sample->vals[55][i * 28 + j])
                printf("%1.3f|", sample->vals[55][i * 28 + j]);
            else
                printf("     |");
        }
        printf("\n");
    }

    HeadLayer->input->vals = sample->vals[55];

    output = Forward(HeadLayer);
    PrintVec(output);

	FreeNetwork(HeadLayer);
}
