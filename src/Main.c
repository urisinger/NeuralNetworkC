#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "Layer.h"

double sigmoid(double x) {
	return (tanh(x));
}

double sigmoidder(double x) {
	return (1-tanh(x)*tanh(x));
}

void main()
{
	srand(time(NULL));

	Vector* inputReal = NewUniformVec(2,1);

	Layer* HeadLayer = NewNetwork(inputReal,3, sigmoid, sigmoidder);

	NewTailLayer(HeadLayer, 5, sigmoid, sigmoidder);

	NewTailLayer(HeadLayer, 1, sigmoid, sigmoidder);
	
	Vector* output = NewVec(1);
	Vector* err;

	for (int i = 0; i < 10000; ++i) {

		inputReal->vals[0] = 0;
		inputReal->vals[1] = 0;

		err = NewVec(1);

		output = Forward(HeadLayer);
		err->vals[0] = (output->vals[0] - (!(int)(inputReal->vals[0]) != !(int)(inputReal->vals[1])));
		//printf("\n %f : ", err->vals[0]);

		FreeVec(output);
	
		BackPropogate(HeadLayer->NextLayer->NextLayer, err, 0.1);

		inputReal->vals[0] = 1;
		inputReal->vals[1] = 0;

		err = NewVec(1);

		output = Forward(HeadLayer);
		err->vals[0] = (output->vals[0] - (!(int)(inputReal->vals[0]) != !(int)(inputReal->vals[1])));
		//printf("\n %f", err->vals[0]);


		FreeVec(output);

		BackPropogate(HeadLayer->NextLayer->NextLayer, err, 0.1);

		inputReal->vals[0] = 0;
		inputReal->vals[1] = 1;

		err = NewVec(1);

		output = Forward(HeadLayer);
		err->vals[0] = (output->vals[0] - (!(int)(inputReal->vals[0]) != !(int)(inputReal->vals[1])));
		//printf("\n %f", err->vals[0]);

		FreeVec(output);

		BackPropogate(HeadLayer->NextLayer->NextLayer, err, 0.1);

		inputReal->vals[0] = 1;
		inputReal->vals[1] = 1;

		err = NewVec(1);

		output = Forward(HeadLayer);
		err->vals[0] = (output->vals[0] - (!(int)(inputReal->vals[0]) != !(int)(inputReal->vals[1])));
		//mkiiprintf("\n %f", err->vals[0]);

		FreeVec(output);

		BackPropogate(HeadLayer->NextLayer->NextLayer, err, 0.1);
	}

	printf("\n\n");
	inputReal->vals[0] = 1;
	inputReal->vals[1] = 1;

	output = Forward(HeadLayer);

	PrintVec(output);
	printf("\n");
	FreeVec(output);

	inputReal->vals[0] = 0;
	inputReal->vals[1] = 1;

	output = Forward(HeadLayer);

	PrintVec(output);
	printf("\n");
	FreeVec(output);

	inputReal->vals[0] = 1;
	inputReal->vals[1] = 0;

	output = Forward(HeadLayer);

	PrintVec(output);
	printf("\n");
	FreeVec(output);

	inputReal->vals[0] = 0;
	inputReal->vals[1] = 0;

	output = Forward(HeadLayer);

	PrintVec(output);
	printf("\n");
	FreeVec(output);

	FreeNetwork(HeadLayer);
}
