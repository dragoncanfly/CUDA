#include "rxGDll.h"

void convert2Matrix(mxArray *ar, float *data)
{
	int rows = mxGetM(ar);
	int cols = mxGetN(ar);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{

			data[i*cols + j] = mxGetPr(ar)[rows*j + i];
		}
	}
}

void writeFile(float* D, int rows, int cols, const char *path, const char *var)
{
	MATFile *pw = matOpen(path, "w");
	if (pw == NULL)
	{
		printf("File open failed");
	}
	mxArray *br = mxCreateDoubleMatrix(rows, cols, mxREAL);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			mxGetPr(br)[j * rows + i] = (float)D[i * cols + j];
		}
	}
	int status = matPutVariable(pw, var, br);
	if (status != 0)
	{
		printf("%s\n", "save result failed");
	}
	mxDestroyArray(br);
	matClose(pw);
}

void readFile(const char*file_path, const char* var, matrix* res)
{
	MATFile *fmat = NULL;
	fmat = matOpen(file_path, "r");
	if (fmat == NULL)
	{
		printf("%s\n", "File open failed");
	}
	mxArray *ar;
	ar = matGetVariable(fmat, var);
	if (ar == NULL)
	{
		printf("%s\n", "Get matrix failed");
	}
	else
	{
		int rows = mxGetM(ar);
		int cols = mxGetN(ar);
		res->data = (float*)malloc(sizeof(float)*rows*cols);
		convert2Matrix(ar, res->data);
		res->rows = rows;
		res->cols = cols;
		res->size = rows*cols;
		res->byteSize = res->size * sizeof(float);

		mxDestroyArray(ar);
		matClose(fmat);
	}
}

void map(float *p, float *res, int m, int n, char operater)
{
	if (operater == 'a')
	{
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
			{
				if (p[i * n + j] != 0) {
					res[i * n + j] = 1;
				}
				else if (p[i * n + j] == 0) {
					res[i * n + j] = 0;
				}
			}
	}
	else if (operater == 'n')
	{
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
			{
				if (p[i * n + j] != 0) {
					res[i * n + j] = 0;
				}
				else if (p[i * n + j] == 0) {
					res[i * n + j] = 1;
				}
			}
	}
}

float max_value(float *p, int m, int n)
{
	float MAX_VALUE = 0.0;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (p[i * n + j] >= MAX_VALUE)
			{
				MAX_VALUE = p[i * n + j];
			}
		}
	}
	return MAX_VALUE;
}

void linspace(float a, int m, float *p)
{
	float t = a / (m - 1);
	for (int i = 0; i < m; i++) {
		p[0] = 0;
		p[i + 1] = p[i] + t;
	}
}
void And(float *p1, float *p2, float *res, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (p1[i * n + j] == 1 && p2[i * n + j] == 1) {
				res[i * n + j] = 1;
			}
			else {
				res[i * n + j] = 0;
			}
		}
	}
}

void matrix_dotmulti(float *p1, float *p2, float *r, int m, int n)
{
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			r[i * n + j] = p1[i * n + j] * p2[i * n + j];
		}
	}
}

float sum(float *p, int m, int n)
{
	float SUM = 0.0;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			SUM += p[i * n + j];
		}
	}
	return SUM;
}