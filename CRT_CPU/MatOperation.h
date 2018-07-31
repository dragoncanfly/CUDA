#ifndef MATOPERATION_H
#define MATOPERATION_H
#include<mat.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include"mkl.h"

typedef struct matrix
{
	float *data;
	int rows;
	int cols;
	int size;
	int byteSize;
}matrix;

typedef struct array_int{
	int* data;
	int num;
}array_int;


enum hostError{
	hostErrNoError,
	hostErrError,
	hostErrFileOpenFailed,
	hostErrMemAlreadyAlloc
};
typedef hostError hostError_t;

void convert2Matrix(mxArray *ar, float *data);

hostError_t readFile(const char* file_path, const char *var, matrix *result);

hostError_t writeFile(matrix *accrucy_NRS, const char *path, const char *var);

hostError_t writeFile(float* data, int rows, int cols, const char *path, const char *var);

void getDimensionMatrix(float *matrix, float *subMatrix, int len, int start, int end, int offset);

float findMatrixMaxValue(matrix *Mat);

float findMatrixMaxValue(float* data, int rows, int cols);

void matrix_trans(float* data,float* data_trans ,int width, int height);

void matrix_trans(matrix* matrix);

void nomalizing(float* data,int width,int height,float maxValue);

void matrix_free(matrix *p1);

void release(float** p, int m, int n);

void matrix_multi(float* pxp,float* p1, float* p2, int m, int n, int l);

void matrix_add_sub(float* leftmatrix, float* rightmatrix,int widht,int height,char oper);

void swap(float *a, float *b);

int inv(float *p, int n);

void printArray(float* data, int rows, int cols);

float norms(float* data, int rows, int cols, int p);

void getMatrixCol(float* data, float* X, int rows, int cols, int index);

void getMatrixCol(float* data, float* X, int rows, int cols, int* bands, int counts);

void matrix_dot_multi(float b, float* data, int rows, int cols);

float min(float* data, int rows, int cols, int colth);

float max(float* data, int rows, int cols, int colth);

int inv2(float* a, int n);

void inv3(float* a, int n);

#endif