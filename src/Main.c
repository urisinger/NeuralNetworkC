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
	Vector* inputReal = NewVec(2);


	Layer* HeadLayer = NewNetwork(inputReal,2, sigmoid, sigmoidder);

	NewTailLayer(HeadLayer, 16, sigmoid, sigmoidder);

    NewTailLayer(HeadLayer, 16, sigmoid, sigmoidder);

    NewTailLayer(HeadLayer, 1, sigmoid, sigmoidder);
	
	Vector* output = NewVec(1);
	Vector* err;

	for (int i = 0; i < 10000; ++i) {

        inputReal->vals[0] = 0;
        inputReal->vals[1] = 0;

        err = NewVec(1);

        output = Forward(HeadLayer);
        err->vals[0] = (output->vals[0] - (!(int) (inputReal->vals[0]) != !(int) (inputReal->vals[1])));
        //printf("\n %f : ", err->vals[0]);

        FreeVec(output);

        BackPropogate(FindTail(HeadLayer), err, 0.1);

        inputReal->vals[0] = 1;
        inputReal->vals[1] = 0;

        err = NewVec(1);

        output = Forward(HeadLayer);
        err->vals[0] = (output->vals[0] - (!(int) (inputReal->vals[0]) != !(int) (inputReal->vals[1])));
        //printf("\n %f", err->vals[0]);


        FreeVec(output);

        BackPropogate(FindTail(HeadLayer), err, 0.1);

        inputReal->vals[0] = 0;
        inputReal->vals[1] = 1;

        err = NewVec(1);

        output = Forward(HeadLayer);
        err->vals[0] = (output->vals[0] - (!(int) (inputReal->vals[0]) != !(int) (inputReal->vals[1])));
        //printf("\n %f", err->vals[0]);

        FreeVec(output);

        BackPropogate(FindTail(HeadLayer), err, 0.1);

        inputReal->vals[0] = 1;
        inputReal->vals[1] = 1;

        err = NewVec(1);

        output = Forward(HeadLayer);
        err->vals[0] = (output->vals[0] - (!(int) (inputReal->vals[0]) != !(int) (inputReal->vals[1])));
        //mkiiprintf("\n %f", err->vals[0]);

        FreeVec(output);
        BackPropogate(FindTail(HeadLayer), err, 0.1);
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
