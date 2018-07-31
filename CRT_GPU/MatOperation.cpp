#include "MatOperation.h"

//declare



/*convert mxArray to Matrix
*row priority in c
*cols priority in Mat
*/
void convert2Matrix(mxArray *ar, float *data)
{
	int rows = mxGetM(ar);
	int cols = mxGetN(ar);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			data[i*cols + j] = (float)mxGetPr(ar)[rows*j + i];
		}
	}
}


float findMatrixMaxValue(matrix *Mat){
	float maxValue = *Mat->data;
	for (int i = 0; i < Mat->rows; i++){
		for (int j = 0; j < Mat->cols; j++){
			if (maxValue<Mat->data[i*Mat->cols + j])
			{
				maxValue = Mat->data[i*Mat->cols + j];
			}
		}
	}
	return maxValue;
}

//read Mat file
hostError_t readFile(const char* file_path, const char *var, matrix *result)
{
	MATFile *fmat = NULL;
	fmat = matOpen(file_path, "r");
	if (fmat == NULL)
	{
		return hostErrFileOpenFailed;
	}
	mxArray *ar;
	ar = matGetVariable(fmat, var);
	if (ar == NULL){
		return hostErrError;
	}
	else{

		int rows = mxGetM(ar);
		int cols = mxGetN(ar);
		result->data = (float*)malloc(sizeof(float)*rows*cols);
		convert2Matrix(ar, result->data);
		result->rows = rows;
		result->cols = cols;
		result->size = rows*cols;
		result->byteSize = result->size*sizeof(float);

		mxDestroyArray(ar);
		return hostErrNoError;
	}

}
//write result to mat file
hostError_t writeFile(matrix *accrucy_NRS, const char *path, const char *var)
{
	float* data = accrucy_NRS->data;
	int rows = accrucy_NRS->rows;
	int cols = accrucy_NRS->cols;
	MATFile *pw = matOpen(path, "w");
	if (pw == NULL) return hostErrFileOpenFailed;
	mxArray *br = mxCreateDoubleMatrix(rows, cols, mxREAL);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			mxGetPr(br)[j*rows + i] = (double)data[i*cols + j];
		}
	}
	int status = matPutVariable(pw, var, br);
	if (status != 0)
	{
		//printf("%s", "save result failed");
		return hostErrError;
	}
	mxDestroyArray(br);
	matClose(pw);
	return hostErrNoError;
}

hostError_t writeFile(float* data, int rows, int cols, const char *path, const char *var)
{
	MATFile *pw = matOpen(path, "w");
	if (pw == NULL) return hostErrFileOpenFailed;
	mxArray *br = mxCreateDoubleMatrix(rows, cols, mxREAL);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			mxGetPr(br)[j*rows + i] = (double)data[i*cols + j];
		}
	}
	int status = matPutVariable(pw, var, br);
	if (status != 0)
	{
		//printf("%s", "save result failed");
		return hostErrError;
	}
	mxDestroyArray(br);
	matClose(pw);
	return hostErrNoError;
}

hostError_t writeFile(int* data, int rows, int cols, const char *path, const char *var)
{
	MATFile *pw = matOpen(path, "w");
	if (pw == NULL) return hostErrFileOpenFailed;
	mxArray *br = mxCreateDoubleMatrix(rows, cols, mxREAL);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			mxGetPr(br)[j*rows + i] = (double)data[i*cols + j];
		}
	}
	int status = matPutVariable(pw, var, br);
	if (status != 0)
	{
		//printf("%s", "save result failed");
		return hostErrError;
	}
	mxDestroyArray(br);
	matClose(pw);
	return hostErrNoError;
}

//get subMatrix
void getDimensionMatrix(float *matrix, float *subMatrix, int len, int start, int end, int offset)
{

	int rows = end - start + 1;
	for (int j = 0; j < rows*len; j++)
	{
		subMatrix[offset*len + j] = matrix[start*len + j];
	}
}

float min(float* data, int rows, int cols, int colth){
	float MIN_VALUE = data[0];
	for (int i = 0; i < rows; i++){
		if (data[i*cols + colth]<MIN_VALUE){
			MIN_VALUE = data[i*cols + colth];
		}
	}
	return MIN_VALUE;
}

float max(float* data, int rows, int cols, int colth){
	float MAX_VALUE = data[0];
	for (int i = 0; i < rows; i++){
		if (data[i*cols + colth]>MAX_VALUE){
			MAX_VALUE = data[i*cols + colth];
		}
	}
	return MAX_VALUE;
}
