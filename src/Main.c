#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "Layer.h"

float sigmoid(float x) {
	return (1/(1+exp(x)));
}

float sigmoidder(float x) {
	return (sigmoid(x)*(1- sigmoid(x)));
}

void main()
{
	srand(time(NULL));

	Vector* inputReal = NewUniformVec(2,1);

	Layer* HeadLayer = NewNetwork(inputReal,5, sigmoid, sigmoidder);

	NewTailLayer(HeadLayer, 1, sigmoid, sigmoidder);
	
	Vector* output = NewVec(1);
	Vector* err;

	for (int i = 0; i < 10000; ++i) {
		inputReal->vals[0] = rand()%2;
		inputReal->vals[1] = rand()%2;

		err = NewVec(1);

		output = Forward(HeadLayer);
		err->vals[0] = (output->vals[0] - (!(int)(inputReal->vals[0]) != !(int)(inputReal->vals[1])));
		//printf("%f : ", err->vals[0]);
		//PrintVec(output);
		FreeVec(output);
		//printf("\n");

		//PrintMat(HeadLayer->Weights);

		BackPropogate(HeadLayer->NextLayer, err, 0.1f);

	}

	inputReal->vals[0] = 1;
	inputReal->vals[1] = 1;

	printf("%d : ", (!(int)(inputReal->vals[0]) != !(int)(inputReal->vals[1])));
	PrintVec(Forward(HeadLayer));


	FreeNetwork(HeadLayer);
}
