#pragma once


typedef double (*Activation)(double);

typedef struct Matrix {
	double** vals;
	int rows, cols;
}Matrix;

typedef struct Vector {
	double* vals;
	int size;
}Vector;

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

void UniformVec(Vector* Vec, double val);
void RandomizeVec(Vector* Vec, double min, double max);

//opers
Vector* AddVec(Vector* Vec1, Vector* Vec2);
Matrix* AddMat(Matrix* Mat1, Matrix* Mat2);
Vector* SubVec(Vector* Vec1, Vector* Vec2);
Matrix* SubMat(Matrix* Mat1, Matrix* Mat2);

Matrix* MatScaler(Matrix* Mat, double scaler);
Vector* VecScaler(Vector* Vec, double scaler);


Matrix* Dot(Matrix* Mat1, Matrix* Mat2);
Vector* DotVecMat(Matrix* Mat1, Vector* Vec);
Matrix* DotTransposeVec(Vector* Vec1, Vector* Vec2);

Matrix* Transpose(Matrix* Mat);
Matrix* TransposeDot(Matrix* Mat1, Matrix* Mat2);

void ApplyFunc(Matrix* Mat, Activation function);
Vector* ApplyFuncVec(Vector* Mat, Activation function);

