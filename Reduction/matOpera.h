#pragma once
#include<stdio.h>
#include<stdlib.h>
#include<mat.h>
#include<time.h>

typedef struct matrix
{
	double *data;
	int rows;
	int cols;
	int size;
	int byteSize;
}matrix;

void zeros(double *p, int m, int n);
void convert2Matrix(mxArray *ar, double *data);
int readFile(const char*file_path, const char* var, matrix* res);
void writeFile(double* D, int rows, int cols, const char *path, const char *var);
void aveRow(double *in, double *out, int m, int n);
void matrix_trans(double *data, double *tran_data, int m, int n);
