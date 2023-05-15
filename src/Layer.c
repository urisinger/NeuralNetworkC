#include "Layer.h"
#include <stdlib.h>


Layer* NewNetwork(Vector* input,int size, Activation ActivationLayer, Activation ActivationDervtive) {
	Layer* newlayer = (Layer*)malloc(sizeof(Layer));
	if (newlayer == NULL) {
		exit(1);
	}
	newlayer->input = input;
	newlayer->size = size;
	newlayer->NextLayer = 0;
	newlayer->LastLayer = 0;
	newlayer->Weights = NewRandMat(size, input->size, -1, 1);
	newlayer->Biases = NewRandVec(size, -1, 1);
	newlayer->ActivationLayer = ActivationLayer;
	newlayer->ActivationDervtive = ActivationDervtive;

	return newlayer;
}

void NewLayer(Layer* LastLayer,int size, Activation ActivationLayer, Activation ActivationDervtive) {
	LastLayer->NextLayer = (Layer*)malloc(sizeof(Layer));
	if (LastLayer->NextLayer == NULL) {
		exit(1);
	}
	Layer* tmp = LastLayer->NextLayer;
	tmp->size = size;
	tmp->Weights = NewRandMat(size, LastLayer->size, -1, 1);
	tmp->Biases = NewRandVec(size, -1, 1);
	tmp->ActivationLayer = ActivationLayer;
	tmp->ActivationDervtive = ActivationDervtive;
	tmp->input = 0;
	tmp->NextLayer = 0;
	tmp->LastLayer = LastLayer;
}

Layer* FindTail(Layer* Head){
    while (Head->NextLayer) {
        Head = Head->NextLayer;
    }
    return(Head);
}

void NewTailLayer(Layer* Head, int size, Activation ActivationLayer, Activation ActivationDervtive) {
	while (Head->NextLayer) {
		Head = Head->NextLayer;
	}
	NewLayer(FindTail(Head), size, ActivationLayer, ActivationDervtive);
}

void LearnSample(Layer * head, Matrix* Sample, Matrix* Labels,double learnrate){
    if(head->input->size != Sample->cols){
        exit(-1);
    }
    Vector *output;
    for(int i = 0; i < Sample->rows; ++i){
        Vector* err = NewVec(Labels->cols);
        head->input->vals = Sample->vals[i];
        PrintVec(head->input);
        printf("\n");
        output = Forward(head);
        for(int j = 0; j < output->size; ++j){
            err->vals[j] = 2.0*(output->vals[j] - Labels->vals[i][j])/output->size;
        }
        BackPropogate(FindTail(head), err, learnrate);
        FreeVec(output);
    }
}

Vector* Forward(Layer* layer){

	Vector* tmp1 = DotVecMat(layer->Weights, layer->input);
	Vector* next_in = AddVec(layer->Biases, tmp1);
	ApplyFuncVec(next_in, layer->ActivationLayer);
	FreeVec(tmp1);

	if (!layer->NextLayer)
		return next_in;

	if (layer->NextLayer->input)
		FreeVec(layer->NextLayer->input);

	layer->NextLayer->input = next_in;
	Forward(layer->NextLayer);
}

void FreeNetwork(Layer* layer) {
	FreeVec(layer->Biases);
	FreeMat(layer->Weights);
	if (layer->NextLayer) {
		FreeNetwork(layer->NextLayer);
	}
	free(layer);
}

void BackPropogate(Layer* layer, Vector* error_grad, double learnrate) {

	Matrix* tmp = DotTransposeVec(error_grad ,layer->input);

	Matrix* weights_grad = MatScaler(tmp, -learnrate);

	Matrix* weight_transpose = Transpose(layer->Weights);
	Vector* next_err = DotVecMat(weight_transpose, error_grad);

	Vector* scaled_out = VecScaler(error_grad, -learnrate);

	Vector* new_bias = AddVec(layer->Biases, scaled_out);
	FreeVec(layer->Biases);
	layer->Biases = new_bias;

	Matrix* new_weight = AddMat(layer->Weights, weights_grad);
    FreeMat(layer->Weights);
	FreeMat(weights_grad);
	layer->Weights = new_weight;

	FreeVec(error_grad);
	FreeVec(scaled_out);
	FreeMat(weight_transpose);
	FreeMat(tmp);



	if (!layer->LastLayer)
		return;
	for (int i = 0; i < next_err->size; ++i) {
		next_err->vals[i] *= layer->ActivationDervtive(layer->input->vals[i]);
	}
	BackPropogate(layer->LastLayer,next_err ,learnrate);
}
