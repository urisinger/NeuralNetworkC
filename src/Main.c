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



int main()
{

    srand(time(NULL));

    FILE* imageTrainFiles = fopen("../../data/train-images.idx3-ubyte", "rb");
    FILE* imageTrainlabels = fopen("../../data/train-labels.idx1-ubyte", "rb");

    if (imageTrainFiles == NULL) {
        perror("File Not Found");
        return -1;
    }

    int magic_number;
    fread(&magic_number, sizeof(int), 1, imageTrainFiles);

    if (ChangeEndianness(magic_number) != 2051) {
        printf("Invalid magic number : %d\n", ChangeEndianness(magic_number));
        return -1;
    }

    Matrix* samples = NewMat(50000, 784);
    Matrix* labels = NewMat(50000, 10);

    GetSample(samples, imageTrainFiles, 16);  // Read the first image starting at offset 16

    GetLabel(labels, imageTrainlabels, 8);


    Layer* HeadLayer = NewNetwork(NewVec(784) ,128, sigmoid, sigmoidder);

    NewTailLayer(HeadLayer, 128, relu,reluder);

    NewTailLayer(HeadLayer, 10, sigmoid, sigmoidder);

    Vector* output;

    LearnBatch(HeadLayer,samples,labels,5,0.01);


    FreeMat(samples);
    FreeMat(labels);

    FILE* imageTestFiles = fopen("../../data/t10k-images.idx3-ubyte", "rb");
    FILE* imageTestlabels = fopen("../../data/t10k-labels.idx1-ubyte", "rb");


    samples = NewMat(10000, 784);
    labels = NewMat(10000, 10);

    GetSample(samples, imageTestFiles, 16);  // Read the first image starting at offset 16

    GetLabel(labels, imageTestlabels, 8);


    Vector* err = NewVec(10);



    while (1) {
        int index;
        printf("\nindex is: ");
        scanf("%d",&index);
        HeadLayer->input->vals = samples->vals[index];

        output = Forward(HeadLayer);
        double max = -1;
        int maxnum;
        for (int i = 0; i < 10; i++) {
            if (max < output->vals[i]) {
                max = output->vals[i];
                maxnum = i;
            }
            err->vals[i] = output->vals[i] - labels->vals[index][i];
        }
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                if (samples->vals[index][i * 28 + j])
                    printf("%1.3f|", samples->vals[index][i * 28 + j]);
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

    return -1;
}
