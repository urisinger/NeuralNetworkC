#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "Layer.h"

double sigmoid(double x) {
	return ((1/(1+exp(-x))));
}

double sigmoidder(double x) {
    double sig = sigmoid(x);
	return (sig*(1-sig));
}

double tanhder(double x){
    double tanhh = tanh(x);
    return 1-tanhh*tanhh;
}
double relu(double x){
    return x <= 0 ? 0:x;
}

double reluder(double x){
    return x <= 0 ? 0:1;
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
        return;
    }

    int magic_number;
    fread(&magic_number, sizeof(int), 1, imageTrainFiles);

    if (ChangeEndianness(magic_number) != 2051) {
        printf("Invalid magic number : %d\n", ChangeEndianness(magic_number));
        return;
    }

    srand(time(NULL));
    Matrix* sample = NewMat(50000, 784);
    Matrix* label = NewMat(50000, 10);

    GetSample(sample, imageTrainFiles, 16);  // Read the first image starting at offset 16

    GetLabel(label, imageTrainLabel, 8);

    Layer* HeadLayer = NewNetwork(NewVec(784) ,100, sigmoid, sigmoidder);

    NewTailLayer(HeadLayer, 32, sigmoid,sigmoidder);

    NewTailLayer(HeadLayer, 10, sigmoid, sigmoidder);

	Vector* output = NewVec(1);

	for (int i = 0; i < 1; ++i) {
        LearnSample(HeadLayer,sample,label,0.1);
    }


    PrintMat(HeadLayer->NextLayer->Weights);
    Vector* err = NewVec(10);

    while (1) {
        int index;
        printf("\nindex is: ");
        scanf("%d",&index);
        HeadLayer->input->vals = sample->vals[index];

        output = Forward(HeadLayer);
        double max = -1;
        int maxnum;
        for (int i = 0; i < 10; i++) {
            if (max < output->vals[i]) {
                max = output->vals[i];
                maxnum = i;
            }
            err->vals[i] = output->vals[i] - label->vals[index][i];
        }
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                if (sample->vals[index][i * 28 + j])
                    printf("%1.3f|", sample->vals[index][i * 28 + j]);
                else
                    printf("     |");
            }
            printf("\n");
        }
        printf("\n\n");
        PrintVec(output);
        printf("\n\n");
        PrintVec(err);
        printf("%d", maxnum);
    }

	FreeNetwork(HeadLayer);
}
