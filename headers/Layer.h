#pragma once
#include "LinearAlgebra.h"


//layer functions
double sigmoid(double x);

double sigmoidder(double x);

double tanhder(double x);
double relu(double x);

double reluder(double x);


typedef struct Layer {
    int size;
    struct Layer* NextLayer;
    struct Layer* LastLayer;
    Matrix* Weights;
    Vector* Biases;
    Vector* input;
    Vector* NoActivateInput;
    Activation ActivationLayer;
    Activation ActivationDervtive;
}Layer;

Layer* NewNetwork(Vector* input, int size, Activation ActivationLayer, Activation ActivationDervtive);
void NewLayer(Layer* LastLayer, int size, Activation ActivationLayer, Activation ActivationDervtive);
Layer* FindTail(Layer* Head);
void NewTailLayer(Layer* Head, int size, Activation ActivationLayer, Activation ActivationDervtive);

Vector* Forward(Layer* layer);
void BackPropogate(Layer* layer, Vector* error_grad, double learnrate);
void LearnBatch(Layer* head, Matrix* Sample, Matrix* Labels,int epochs, double learnrate);

void FreeNetwork(Layer* layer);