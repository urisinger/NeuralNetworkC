#pragma 
#include "LinearAlgebra.h"

typedef struct Layer {
	int size;
	struct Layer* NextLayer;
	struct Layer* LastLayer;
	Matrix* Weights;
	Vector* Biases;
	Vector* input;
	Activation ActivationLayer;
	Activation ActivationDervtive;
}Layer;

Layer* NewNetwork(Vector* input, int size, Activation ActivationLayer, Activation ActivationDervtive);
void NewLayer(Layer* LastLayer, int size, Activation ActivationLayer, Activation ActivationDervtive);
void NewTailLayer(Layer* Head, int size, Activation ActivationLayer, Activation ActivationDervtive);

Vector* Forward(Layer* layer);
void BackPropogate(Layer* layer, Vector* error_grad, float learnrate);

void FreeNetwork(Layer* layer);