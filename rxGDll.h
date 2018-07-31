#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <mat.h>
#include <time.h>
#include <math.h>

typedef struct matrix
{
	float *data;
	int rows;
	int cols;
	int size;
	int byteSize;
}matrix;

void convert2Matrix(mxArray *ar, float* data);
void writeFile(float* p1, int m, int n, const char *path, const char *var);
extern "C" void readFile(const char*file_path, const char* var, matrix* res);
extern "C" void map(float *p, float *res, int m, int n, char operater);
extern "C" float max_value(float *p, int m, int n);
extern "C" void linspace(float a, int m, float *p);
extern "C" void And(float *p1, float *p2, float *res, int m, int n);
extern "C" void matrix_dotmulti(float *p1, float *p2, float *r, int m, int n);
extern "C" float sum(float *p, int m, int n);