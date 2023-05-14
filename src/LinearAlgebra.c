#include "LinearAlgebra.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

Matrix* NewMat(int rows, int cols) {
	Matrix* Mat = (Matrix*)malloc(sizeof(Matrix));
	if (Mat == NULL)
		exit(1);
	Mat->rows = rows, Mat->cols = cols;

	Mat->vals = (double**)malloc(sizeof(double*)*rows);
	if (Mat->vals == NULL)
		exit(1);
	for (int i = 0; i < rows; ++i)
		Mat->vals[i] = (double*)malloc(sizeof(double) * cols);
	return Mat;
}

Matrix* NewUniformMat(int rows, int cols, double val) {
	Matrix* Mat = NewMat(rows, cols);
	UniformMat(Mat, val);
	return Mat;
}

Matrix* NewRandMat(int rows, int cols, double min, double max) {
	Matrix* Mat = NewMat(rows, cols);
	RandomizeMat(Mat, min, max);
	return Mat;
}

Matrix* CopyMat(Matrix* OldMat) {
	Matrix* Mat = NewMat(OldMat->rows, OldMat->cols);
	for (int i = 0; i < OldMat->rows; ++i) {
		for (int j = 0; j < OldMat->cols; ++j) {
			Mat->vals[i][j] = OldMat->vals[i][j];
		}
	}
	return Mat;
}

void FreeMat(Matrix* Mat) {
	for (int i = 0; i < Mat->rows; free(Mat->vals[i]), ++i);
	free(Mat->vals);
	free(Mat);
}

void PrintMat(Matrix* Mat) {
	for (int i = 0; i < Mat->rows; ++i) {
		for (int j = 0; j < Mat->cols; ++j) {
			printf("%f |", Mat->vals[i][j]);
		}
		printf("\n");
	}
}

void UniformMat(Matrix* Mat, double val) {
	for (int i = 0; i < Mat->rows; ++i) {
		for (int j = 0; i < Mat->cols; ++j) {
			Mat->vals[i][j] = val;
		}
	}
}

void RandomizeMat(Matrix* Mat, double min, double max) {
	for (int i = 0; i < Mat->rows; ++i) {
		for (int j = 0; j < Mat->cols; ++j) {
			Mat->vals[i][j] = min + ((double)rand() / (double)RAND_MAX) * (max - min);
		}
	}
}

Vector* AddVec(Vector* Vec1, Vector* Vec2) {
	if (Vec1->size != Vec2->size) {
		printf("cant add those 2");
		exit(1);
	}

	Vector* sum = NewVec(Vec1->size);

	for (int i = 0; i < Vec1->size; ++i) {
		sum->vals[i] = Vec1->vals[i] + Vec2->vals[i];
	}
	return sum;
}

Vector* SubVec(Vector* Vec1, Vector* Vec2) {
	if (Vec1->size != Vec2->size) {
		printf("cant add those 2");
		exit(1);
	}

	Vector* sum = NewVec(Vec1->size);

	for (int i = 0; i < Vec1->size; ++i) {
		sum->vals[i] = Vec1->vals[i] - Vec2->vals[i];
	}
	return sum;
}

Matrix* AddMat(Matrix* Mat1, Matrix* Mat2) {
	if (Mat1->rows != Mat2->rows || Mat1->cols != Mat2->cols) {
		printf("cant add those 2");
		exit(1);
	}

	Matrix* sum = NewMat(Mat1->rows, Mat1->cols);

	for (int i = 0; i < sum->rows; ++i) {
		for (int j = 0; j < sum->cols; ++j) {
			sum->vals[i][j] = Mat1->vals[i][j] + Mat2->vals[i][j];
		}
	}
	return sum;
}

Matrix* SubMat(Matrix* Mat1, Matrix* Mat2) {
	if (Mat1->rows != Mat2->rows || Mat1->cols != Mat2->cols) {
		printf("cant add those 2");
		exit(1);
	}

	Matrix* sum = NewMat(Mat1->rows, Mat1->cols);

	for (int i = 0; i < sum->rows; ++i) {
		for (int j = 0; j < sum->cols; ++j) {
			sum->vals[i][j] = Mat1->vals[i][j] - Mat2->vals[i][j];
		}
	}
	return sum;
}

Matrix* MatScaler(Matrix* Mat, double scaler) {
	Matrix* prodact = NewMat(Mat->rows, Mat->cols);

	for (int i = 0; i < prodact->rows; ++i) {
		for (int j = 0; j < prodact->cols; ++j) {
			prodact->vals[i][j] = Mat->vals[i][j]*scaler;
		}
	}
	return prodact;
}

Vector* VecScaler(Vector* Vec, double scaler) {
	Vector* prodact = NewVec(Vec->size);

	for (int i = 0; i < Vec->size; ++i) {
		prodact->vals[i] = Vec->vals[i]* scaler;
	}
	return prodact;
}

Matrix* Dot(Matrix* Mat1, Matrix* Mat2) {
	if (Mat1->cols != Mat2->rows) {
		printf("Cant dot this! %dx%d incompatble with %dx%d", Mat1->rows, Mat1->cols, Mat2->rows, Mat2->cols);
		exit(0);
	}

	Matrix* Mat = NewMat(Mat1->rows, Mat2->cols);


	for (int i = 0; i < Mat->rows; ++i) {
		for (int j = 0; j < Mat->cols; ++j) {
			double sum = 0;
			for (int k = 0; k < Mat1->cols; ++k) {
				sum += Mat1->vals[i][k] * Mat2->vals[k][j];
			}
			Mat->vals[i][j] = sum;
		}
	}
	return Mat;
}


Vector* DotVecMat(Matrix* Mat1, Vector* Vec) {
	if (Mat1->cols != Vec->size) {
		printf("Cant dot this! %dx%d incompatble with %dx%d", Mat1->rows, Mat1->cols, Vec->size, 1);
		exit(0);
	}

	Vector* VecOut = NewVec(Mat1->rows);

	for (int i = 0; i < VecOut->size; ++i) {
		double sum = 0;
		for (int k = 0; k < Mat1->cols; ++k)
			sum += Mat1->vals[i][k] * Vec->vals[k];
		VecOut->vals[i] = sum;
	}
	return VecOut;
}

Matrix* DotTransposeVec(Vector* Vec1, Vector* Vec2) {
	Matrix* MatOut = NewMat(Vec1->size, Vec2->size);

	for (int i = 0; i < Vec1->size; ++i) {
		for (int j = 0; j < Vec2->size; ++j) {
			MatOut->vals[i][j] = Vec1->vals[i] * Vec2->vals[j];
		}
	}
	return MatOut;
}

Vector* DotVecsAndFree(Matrix* Mat1, Vector* Vec) {
	if (Mat1->cols != Vec->size) {
		printf("Cant dot this! %dx%d incompatble with %dx%d", Mat1->rows, Mat1->cols, Vec->size, 1);
		exit(0);
	}

	Vector* VecOut = NewVec(Mat1->rows);

	for (int i = 0; i < VecOut->size; ++i) {
		double sum = 0;
		for (int k = 0; k < Mat1->cols; ++k)
			sum += Mat1->vals[i][k] * Vec->vals[k];
		VecOut->vals[i] = sum;
	}

	FreeVec(Vec);
	return VecOut;
}

Matrix* Transpose(Matrix* Mat) {
	Matrix* mew_mat = NewMat(Mat->cols, Mat->rows);

	for (int i = 0; i < Mat->rows; ++i) {
		for (int j = 0; j < Mat->cols; ++j) {
			mew_mat->vals[j][i] = Mat->vals[i][j];
		}
	}
	return mew_mat;
}

Matrix* TransposeDot(Matrix* Mat1, Matrix* Mat2) {
	if (Mat1->cols != Mat2->cols) {
		printf("Cant dot this! %dx%d incompatble with %dx%d", Mat1->rows, Mat1->cols, Mat2->cols, Mat2->rows);
		exit(0);
	}

	Matrix* Mat = NewMat(Mat1->rows, Mat2->rows);


	for (int i = 0; i < Mat->rows; ++i) {
		for (int j = 0; j < Mat->cols; ++j) {
			double sum = 0;
			for (int k = 0; k < Mat1->cols; ++k) {
				sum += Mat1->vals[i][k] * Mat2->vals[j][k];
			}
			Mat->vals[i][j] = sum;
		}
	}
	return Mat;
}

void ApplyFunc(Matrix* Mat, Activation function) {
	for (int i = 0; i < Mat->rows; ++i) {
		for (int j = 0; j < Mat->cols; ++j) {
			Mat->vals[i][j] = function(Mat->vals[i][j]);
		}
	}

}


Vector* NewVec(int size) {
	Vector* Vec = (Vector*)malloc(sizeof(Vector));
	if (Vec == NULL)
		exit(1);
	Vec->size = size;

	Vec->vals = (double*)malloc(sizeof(double) * size);
	if (Vec->vals == NULL)
		exit(1);
	return Vec;
}

Vector* NewUniformVec(int size, double val) {
	Vector* Vec = NewVec(size);
	UniformVec(Vec, val);

	return Vec;
}

Vector* NewRandVec(int size, double min, double max) {
	Vector* Vec = NewVec(size);
	RandomizeVec(Vec, min, max);
	
	return Vec;
}

Vector* CopyVec(Vector* OldVec) {
	Vector* Vec = NewVec(OldVec->size);

	for (int i = 0; i < Vec->size; ++i) {
		Vec->vals[i] = OldVec->vals[i];
	}
	return Vec;
}

void FreeVec(Vector* Vec) {
	free(Vec->vals);
	free(Vec);
}

void PrintVec(Vector* Vec) {
	for (int i = 0; i < Vec->size; ++i) {
		printf("%f |", Vec->vals[i]);
	}
}


void UniformVec(Vector* Vec, double val) {
	for (int i = 0; i < Vec->size; ++i) {
		Vec->vals[i] = val;
	}
}

void RandomizeVec(Vector* Vec, double min, double max) {
	for (int i = 0; i < Vec->size; ++i) {
		Vec->vals[i] = min + ((double)rand() / (double)RAND_MAX) * (max - min);
	}
}

Vector* ApplyFuncVec(Vector* Vec, Activation function) {
	for (int i = 0; i < Vec->size; ++i) {
		Vec->vals[i] = function(Vec->vals[i]);
	}
	return Vec;
}