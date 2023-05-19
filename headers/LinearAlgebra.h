#pragma once
#include <stdio.h>


typedef struct Matrix {
	double** vals;
	int rows, cols;
}Matrix;

typedef struct Vector {
	double* vals;
	int size;
}Vector;

typedef void (*Activation)(Vector*);


//constactors
Matrix* NewMat(int rows, int cols);
Matrix* NewUniformMat(int rows, int cols, double val);
Matrix* NewRandMat(int rows, int cols, double min, double max);
Matrix* CopyMat(Matrix* OldMat);

Vector* NewVec(int size);
Vector* NewUniformVec(int size, double val);
Vector* NewRandVec(int size, double min, double max);
Vector* CopyVec(Vector* OldMat);


//destructors
void FreeMat(Matrix* Mat);
void FreeVec(Vector* Vec);

//visualize
void PrintMat(Matrix* Mat);
void PrintVec(Vector* Vec);

//fillers
void UniformMat(Matrix* Mat, double val);
void RandomizeMat(Matrix* Mat, double min, double max);
void GetSample(Matrix* Mat, FILE* dataFile, size_t offset);
void GetLabel(Matrix* mat, FILE* dataFile, size_t offset);

void UniformVec(Vector* Vec, double val);
void RandomizeVec(Vector* Vec, double min, double max);

//opers
Vector* AddVec(Vector* Vec1, Vector* Vec2);
Matrix* AddMat(Matrix* Mat1, Matrix* Mat2);
Vector* SubVec(Vector* Vec1, Vector* Vec2);
Matrix* SubMat(Matrix* Mat1, Matrix* Mat2);

Matrix* MatScaler(Matrix* Mat, double scaler);
Vector* VecScaler(Vector* Vec, double scaler);

Vector* HadamardVec(Vector* vec1, Vector* vec2);

Matrix* Dot(Matrix* Mat1, Matrix* Mat2);
Vector* DotVecMat(Matrix* Mat1, Vector* Vec);
Matrix* DotTransposeVecVec(Vector* Vec1, Vector* Vec2);
Matrix* TransposeDot(Matrix* Mat1, Matrix* Mat2);

Matrix* Transpose(Matrix* Mat);

void ShuffleMatrixRows(Matrix* matrix1, Matrix* matrix2);
