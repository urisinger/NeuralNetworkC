#pragma once
#include <stdio.h>


typedef struct Matrix {
	float** vals;
	int rows, cols;
}Matrix;

typedef struct Vector {
	float* vals;
	int size;
}Vector;

typedef void (*Activation)(Vector*);


//constactors
Matrix* NewMat(int rows, int cols);
Matrix* NewUniformMat(int rows, int cols, float val);
Matrix* NewRandMat(int rows, int cols, float min, float max);
Matrix* CopyMat(Matrix* OldMat);

Vector* NewVec(int size);
Vector* NewUniformVec(int size, float val);
Vector* NewRandVec(int size, float min, float max);
Vector* CopyVec(Vector* OldMat);


//destructors
void FreeMat(Matrix* Mat);
void FreeVec(Vector* Vec);

//visualize
void PrintMat(Matrix* Mat);
void PrintVec(Vector* Vec);

//fillers
void UniformMat(Matrix* Mat, float val);
void RandomizeMat(Matrix* Mat, float min, float max);	
void GetSample(Matrix* Mat, FILE* dataFile, size_t offset);
void GetLabel(Matrix* mat, FILE* dataFile, size_t offset);

void UniformVec(Vector* Vec, float val);
void RandomizeVec(Vector* Vec, float min, float max);

//opers
Vector* AddVec(Vector* Vec1, Vector* Vec2);
Matrix* AddMat(Matrix* Mat1, Matrix* Mat2);
Vector* SubVec(Vector* Vec1, Vector* Vec2);
Matrix* SubMat(Matrix* Mat1, Matrix* Mat2);

Matrix* MatScaler(Matrix* Mat, float scaler);
Vector* VecScaler(Vector* Vec, float scaler);

Vector* HadamardVec(Vector* vec1, Vector* vec2);

Matrix* Dot(Matrix* Mat1, Matrix* Mat2);
Vector* DotVecMat(Matrix* Mat1, Vector* Vec);
Matrix* DotTransposeVecVec(Vector* Vec1, Vector* Vec2);
Matrix* TransposeDot(Matrix* Mat1, Matrix* Mat2);

Matrix* Transpose(Matrix* Mat);

void ShuffleMatrixRows(Matrix* matrix1, Matrix* matrix2);
