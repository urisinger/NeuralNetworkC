#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "Layer.h"

double sigmoid(double x) {
	return (1.0/(1+exp(x)));
}

double sigmoidder(double x) {
    double sig = tanh(x);
	return (sig*(1-sig));
}

double tanhder(double x) {
    return (1 - tanh(x) * tanh(x));
}

int ChangeEndianness(int value) {
    int result = 0;
    result |= (value & 0x000000FF) << 24;
    result |= (value & 0x0000FF00) << 8;
    result |= (value & 0x00FF0000) >> 8;
    result |= (value & 0xFF000000) >> 24;
    return result;
}

void PrintNum(Matrix* sample, int index) {

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (sample->vals[index][i * 28 + j])
                printf("%1.3f|", sample->vals[index][i * 28 + j]);
            else
                printf("     |");
        }
        printf("\n");
    }
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
    Matrix* sample = NewMat(45000, 784);
    Matrix* label = NewMat(45000, 10);

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

    Layer* HeadLayer = NewNetwork(NewVec(784) ,16, tanh, tanhder);

    NewTailLayer(HeadLayer, 16, tanh, tanhder);

    NewTailLayer(HeadLayer, 10, tanh, tanhder);

	Vector* output = NewVec(10);

	for (int i = 0; i < 5; ++i) {
        LearnSample(HeadLayer,sample,label,0.1);
    }


    while (1) {
        int index;
        printf("\nindex is: ");
        scanf_s("%d",&index);
        HeadLayer->input->vals = sample->vals[index];

        output = Forward(HeadLayer);
        double max = -1;
        int maxnum;
        for (int i = 0; i < 10; i++) {
            if (max < output->vals[i]) {
                max = output->vals[i];
                maxnum = i;
            }

        }

        PrintVec(output);
        PrintNum(sample, index);
        printf("%d", maxnum);
    }

    FreeNetwork(HeadLayer);


}
