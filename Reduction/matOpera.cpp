#include"matOpera.h"

void zeros(double *p, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)

		{
			p[i * n + j] = 0.0;
		}
	}
}

void convert2Matrix(mxArray *ar, double *data)
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

int readFile(const char*file_path, const char* var, matrix* res)
{
	MATFile *fmat = NULL;
	fmat = matOpen(file_path, "r");
	if (fmat == NULL)
	{
		return 1;
	}
	mxArray *ar;
	ar = matGetVariable(fmat, var);
	if (ar == NULL)
	{
		return 2;
	}
	else
	{
		int rows = mxGetM(ar);
		int cols = mxGetN(ar);
		res->data = (double*)malloc(sizeof(double)*rows*cols);
		convert2Matrix(ar, res->data);
		res->rows = rows;
		res->cols = cols;
		res->size = rows*cols;
		res->byteSize = res->size * sizeof(double);

		mxDestroyArray(ar);
		return 0;
	}
}

void writeFile(double* D, int rows, int cols, const char *path, const char *var)
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
			mxGetPr(br)[j * rows + i] = (double)D[i * cols + j];
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

void aveRow(double *in, double *out, int m, int n)
{
	double sum = 0.0;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			sum += in[i * n + j];
		}
		out[i] = sum / n;
		sum = 0.0;
	}
}

void matrix_trans(double *data, double *tran_data, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
			tran_data[j * m + i] = data[i * n + j];
	}
}