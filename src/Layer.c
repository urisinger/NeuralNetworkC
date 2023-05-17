#include "Layer.h"
#include <stdlib.h>
#include <time.h>


//layer functions
void sigmoid(Vector* x) {
    for (int i = 0; i < x->size; i++) {
        x->vals[i] = 1 / (1 + exp(-x->vals[i]));
    }
}

void sigmoidder(Vector* x) {

    Vector* sig = NewVec(x->size);
    sigmoid(sig);
    for (int i = 0; i < sig->size; i++) {
        x->vals[i] = sig->vals[i] * (1 - sig->vals[i]);
    }
    FreeVec(sig);
}

void tanh2(Vector* x) {
    for (int i = 0; i < x->size; i++) {
        x->vals[i] = tanh(x->vals[i]);
    }
}

void tanhder(Vector* x) {
    for (int i = 0; i < x->size; i++) {
        double tanhh = tanh(x->vals[i]);
        x->vals[i] = 1 - tanhh * tanhh;
    }
}

void relu(Vector* x) {
    for (int i = 0; i < x->size; i++) {
        x->vals[i] = x->vals[i] <= 0 ? 0 : x->vals[i];
    }
}

void reluder(Vector* x) {
    for (int i = 0; i < x->size; i++) {
        x->vals[i] = x->vals[i] <= 0 ? 0 : 1;
    }
}

void softmax(Vector* x) {

    double sum = 0.0;
    for (int i = 0; i < x->size; i++) {
        x->vals[i] = exp(x->vals[i]);
        sum += x->vals[i];
    }

    for (int i = 0; i < x->size; i++) {
        x->vals[i] /= sum;
    }
}


void softmaxder(Vector* x) {
    Vector* softm = NewVec(x->size);
    softmax(softm);
    for (int i = 0; i < softm->size; i++) {
        x->vals[i] = softm->vals[i] * (1 - softm->vals[i]);
    }
    FreeVec(softm);
}



//create a network and set the params
Layer* NewNetwork(Vector* input, int size, Activation ActivationLayer, Activation ActivationDervtive) {
    Layer* newlayer = (Layer*)malloc(sizeof(Layer));
    if (newlayer == NULL) {
        exit(1);
    }
    newlayer->input = input;
    newlayer->PreActivateOut = 0;
    newlayer->size = size;
    newlayer->NextLayer = 0;
    newlayer->LastLayer = 0;
    newlayer->Weights = NewRandMat(size, input->size, -1, 1);
    newlayer->Biases = NewRandVec(size, -1, 1);
    newlayer->ActivationLayer = ActivationLayer;
    newlayer->ActivationDervtive = ActivationDervtive;

    return newlayer;
}

//create a layer at the end of the network
void NewLayer(Layer* LastLayer, int size, Activation ActivationLayer, Activation ActivationDervtive) {
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
    tmp->PreActivateOut = 0;
    tmp->NextLayer = 0;
    tmp->LastLayer = LastLayer;
}

//find tail of headlayer
Layer* FindTail(Layer* Head) {
    while (Head->NextLayer) {
        Head = Head->NextLayer;
    }
    return(Head);
}

//create layer at tail
void NewTailLayer(Layer* Head, int size, Activation ActivationLayer, Activation ActivationDervtive) {
    //find tail
    while (Head->NextLayer) {
        Head = Head->NextLayer;
    }
    //new layer
    NewLayer(FindTail(Head), size, ActivationLayer, ActivationDervtive);
}



void LearnBatch(Layer* head, Matrix* Sample, Matrix* Labels, int epochs, double learnrate) {
    if (head->input->size != Sample->cols) {
        exit(-1);
    }
    Vector* output;
    double errsum = 0;
    clock_t lastbatch = clock();
    for (int k = 0; k < epochs; k++) {
        for (int i = 0; i < Sample->rows; ++i) {

            //find err
            Vector* err = NewVec(Labels->cols);
            head->input->vals = Sample->vals[i];
            output = Forward(head);
            for (int j = 0; j < output->size; ++j) {
                double error = (output->vals[j] - Labels->vals[i][j]);
                err->vals[j] = 2 * error / output->size;
                errsum += error * error;
            }

            //backprop
            BackPropogate(FindTail(head), err, learnrate);
            FreeVec(output);

            //print shit
            if (!(i % 1000)) {

                system("cls");
                printf("trraining model... sample %d/%d. error is : %f ", k, epochs, errsum / (Labels->cols * 1000));
                errsum = 0;
                printf("[");
                for (int j = 0; j < Sample->rows; j += 1000) {
                    if (j < i)
                        printf("|");
                    else
                        printf(" ");

                }
                printf("]\n");
                printf("time since last batch: %f", (double)(clock() - lastbatch) / CLOCKS_PER_SEC);
                lastbatch = clock();
            }
        }
    }
    printf("\n");
}

Vector* Forward(Layer* layer) {

    Vector* tmp1 = DotVecMat(layer->Weights, layer->input);
    Vector* next_in = AddVec(layer->Biases, tmp1);
    Vector* next_in_copy = CopyVec(next_in);

    layer->ActivationLayer(next_in);

    if (layer->PreActivateOut)
        FreeVec(layer->PreActivateOut);

    layer->PreActivateOut = next_in_copy;

    FreeVec(tmp1);

    if (!layer->NextLayer)
        return next_in;


    if (layer->NextLayer->input)
        FreeVec(layer->NextLayer->input);

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

    Vector* derivative = CopyVec(layer->PreActivateOut);
    layer->ActivationDervtive(derivative);

    Vector* error_grad_scaled = HadamardVec(error_grad, derivative);

    Matrix* tmp = DotTransposeVecVec(error_grad_scaled, layer->input);

    Matrix* weights_grad = MatScaler(tmp, -learnrate);

    Matrix* weight_transpose = Transpose(layer->Weights);


    Vector* scaled_out = VecScaler(error_grad_scaled, -learnrate);

    Vector* new_bias = AddVec(layer->Biases, scaled_out);
    FreeVec(layer->Biases);
    layer->Biases = new_bias;

    Matrix* new_weight = AddMat(layer->Weights, weights_grad);
    FreeMat(layer->Weights);
    FreeMat(weights_grad);
    layer->Weights = new_weight;

    FreeVec(scaled_out);
    FreeMat(tmp);



    if (layer->LastLayer) {

        Vector* next_err = DotVecMat(weight_transpose, error_grad_scaled);

        /*Vector* derivative = CopyVec(layer->LastLayer->PreActivateOut);
        layer->LastLayer->ActivationDervtive(derivative);
        Vector* next_err_scaled = HadamardVec(next_err, derivative);*/
        FreeMat(weight_transpose);
        FreeVec(error_grad);
        FreeVec(error_grad_scaled);
        FreeVec(derivative);
        BackPropogate(layer->LastLayer, next_err, learnrate);
    }
    else {
        FreeVec(error_grad);
        FreeVec(error_grad_scaled);
        FreeMat(weight_transpose);
        FreeVec(derivative);
    }
}