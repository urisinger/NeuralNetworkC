#include "Layer.h"
#include <stdlib.h>



Vector* HadamardVec(Vector* vec1, Vector* vec2) {
    if (vec1->size != vec2->size) {
        exit(-1);  // Handle error: Vectors must have the same size
    }

    Vector* result = NewVec(vec1->size);
    for (int i = 0; i < vec1->size; i++) {
        result->vals[i] = vec1->vals[i] * vec2->vals[i];
    }

    return result;
}


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
	newlayer->Weightoffset = NewUniformMat(size, input->size, 0);

	newlayer->Biases = NewRandVec(size, -1, 1);
	newlayer->BiasOffset = NewUniformVec(size, 0);
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
	tmp->Weightoffset = NewUniformMat(size, LastLayer->size, 0);
	tmp->Biases = NewRandVec(size, -1, 1);
	tmp->BiasOffset = NewUniformVec(size, 0);
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

void AddOffsets(Layer* Head,int scaler) {
	while (Head->NextLayer) {
		Matrix* tmp_mat = AddMat(Head->Weights, Head->Weightoffset);
		Matrix* new_mat = MatScaler(tmp_mat, 1.0 / scaler);
		FreeMat(Head->Weights);
		FreeMat(tmp_mat);
		Head->Weights = new_mat;
		UniformMat(Head->Weightoffset, 0);

		Vector* tmp_vec = AddVec(Head->Biases, Head->BiasOffset);
		Vector* new_vec = VecScaler(tmp_vec, 1.0 / scaler);

		FreeVec(Head->Biases);
		FreeVec(tmp_vec);
		Head->Biases = new_vec;
		UniformMat(Head->BiasOffset, 0);

		Head = Head->NextLayer;
	}
}

void NewTailLayer(Layer* Head, int size, Activation ActivationLayer, Activation ActivationDervtive) {
	while (Head->NextLayer) {
		Head = Head->NextLayer;
	}
	NewLayer(FindTail(Head), size, ActivationLayer, ActivationDervtive);
}


void LearnSample(Layer * head, Matrix* Sample, Matrix* Labels,double learnrate,int start,int end){
    if(head->input->size != Sample->cols){
        exit(-1);
    }
    Vector *output;
    for(int i = start; i < end; ++i){
        Vector* err = NewVec(Labels->cols);
        head->input->vals = Sample->vals[i];
        output = Forward(head);
        for(int j = 0; j < output->size; ++j){
            err->vals[j] = 2.0*(output->vals[j] - Labels->vals[i][j])/output->size;
        }

        BackPropogate(FindTail(head), err, learnrate);
        FreeVec(output);

    }

	AddOffsets(head,end-start);
}

Vector* Forward(Layer* layer){

	Vector* tmp1 = DotVecMat(layer->Weights, layer->input);
	Vector* next_in = AddVec(layer->Biases, tmp1);
    Vector* next_in_copy = CopyVec(next_in);

    ApplyFuncVec(next_in, layer->ActivationLayer);

    if(layer->NoActivateInput)
        FreeVec(layer->NoActivateInput);

    layer->NoActivateInput = next_in_copy;
	FreeVec(tmp1);

	if (!layer->NextLayer)
		return next_in;

	if (layer->NextLayer->input)
		FreeVec(layer->NextLayer->input);
		layer->NextLayer->input = NULL;
	}

    layer->NextLayer->input = next_in;
	return Forward(layer->NextLayer);
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

	Vector* new_bias_offset = AddVec(layer->BiasOffset, scaled_out);
	FreeVec(layer->BiasOffset);
	layer->BiasOffset = new_bias_offset;

	Matrix* new_weight_offset = AddMat(layer->Weightoffset, weights_grad);
	FreeMat(layer->Weightoffset);
	FreeMat(weights_grad);
	layer->Weightoffset = new_weight_offset;

	FreeVec(error_grad);
	FreeVec(scaled_out);
	FreeMat(weight_transpose);
	FreeMat(tmp);


    if (layer->LastLayer) {

        Vector* derivative = CopyVec(layer->input);
        ApplyFuncVec(derivative,layer->ActivationDervtive);
        Vector* next_err_scaled = HadamardVec(next_err, derivative);
        FreeVec(next_err);
        FreeVec(derivative);
        BackPropogate(layer->LastLayer, next_err_scaled, learnrate);
}
