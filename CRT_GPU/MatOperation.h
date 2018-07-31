#include<mat.h>
#include<string.h>
#include<stdlib.h>
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


typedef unsigned char uchar;
typedef struct{
	uchar* table;
	int samples;
	int num;
}Mapping;

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

hostError_t writeFile(int* data, int rows, int cols, const char *path, const char *var);

void getDimensionMatrix(float *matrix, float *subMatrix, int len, int start, int end, int offset);

float findMatrixMaxValue(matrix *Mat);

float min(float* data, int rows, int cols, int colth);

float max(float* data, int rows, int cols, int colth);