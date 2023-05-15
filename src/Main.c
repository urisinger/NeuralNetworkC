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

    FILE *imageTestFiles = fopen("../data/train-images.idx3-ubyte","r");
    if(imageTestFiles == NULL) {
        perror("File Not Found");
    }
    int magic_number_bytes;
    fread(&magic_number_bytes, sizeof(int), 1, imageTestFiles);

    printf("%d\n", ChangeEndianness(magic_number_bytes));

	srand(time(NULL));
    Matrix* sample = NewMat(4,2);

    Matrix* label = NewMat(4,1);

    sample->vals[0][0]= 0;
    sample->vals[0][1]= 0;
    label->vals[0][0] = 0;

    sample->vals[1][0]= 1;
    sample->vals[1][1]= 1;
    label->vals[1][0] = 0;

    sample->vals[2][0]= 1;
    sample->vals[2][1]= 0;
    label->vals[2][0] = 1;

    sample->vals[3][0]= 0;
    sample->vals[3][1]= 1;
    label->vals[3][0] = 1;

    Vector* inputReal = NewVec(2);

    Layer* HeadLayer = NewNetwork(inputReal,2, sigmoid, sigmoidder);

	NewTailLayer(HeadLayer, 16, sigmoid, sigmoidder);

    NewTailLayer(HeadLayer, 16, sigmoid, sigmoidder);

    NewTailLayer(HeadLayer, 1, sigmoid, sigmoidder);

	Vector* output = NewVec(1);

	for (int i = 0; i < 10000; ++i) {
        LearnSample(HeadLayer,sample,label,0.1);
    }

    for(double i = 0; i <= 1;i+=0.05){
        for(double j = 0; j <= 1; j+=0.05){
            inputReal->vals[0] = i;
            inputReal->vals[1] = j;
            output = Forward(HeadLayer);
            printf("%f|",output->vals[0]);
            FreeVec(output);
        }
        printf("\n");
    }

	FreeNetwork(HeadLayer);
}
