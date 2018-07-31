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

			data[i*cols + j] = mxGetPr(ar)[rows*j + i];
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

float findMatrixMaxValue(float* data,int rows,int cols){
	float maxValue = data[0];
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			if (maxValue<data[i*cols + j])
			{
				maxValue = data[i*cols + j];
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
			mxGetPr(br)[j*rows + i] = (float)data[i*cols + j];
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

hostError_t writeFile(float* data,int rows,int cols ,const char *path, const char *var)
{
	MATFile *pw = matOpen(path, "w");
	if (pw == NULL) return hostErrFileOpenFailed;
	mxArray *br = mxCreateDoubleMatrix(rows, cols, mxREAL);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			mxGetPr(br)[j*rows + i] = (float)data[i*cols + j];
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

void matrix_trans(float* data,float* data_trans ,int width, int height){
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			data_trans[j*height+i] = data[i*width + j];
		}
	}
}

void matrix_trans(matrix* dataSet){
	float* datatrans = (float*)malloc(sizeof(float)*dataSet->byteSize);
	int height = dataSet->rows;
	int width = dataSet->cols;
	matrix_trans(dataSet->data, datatrans, width, height);
	free(dataSet->data);
	dataSet->data = datatrans;
	dataSet->rows = width;
	dataSet->cols = height;
}

void nomalizing(float* data,int width,int height,float maxValue){
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			data[i*width + j] = data[i*width + j] / maxValue;
		}
	}
}

void matrix_free(matrix *p1)
{
	free(p1->data);
	free(p1);
	p1 = NULL;
}

void release(float** p, int m, int n)
{
	for (int i = 0; i < m; i++){
		free(p[i]);
		p[i] = NULL;
	}

	free(p);
	p = NULL;
}

//p1:m*n   px:n*l   pxp:m*l
void matrix_multi(float* pxp,float* p1, float* p2, int m, int n, int l){
	/*for (int i = 0; i < m; i++)
	{
		for (int k = 0; k < n; k++){
			float r = p1[i*n+k];
			for (int j = 0; j < l; j++)
			{
				pxp[i*l+j] += r*p2[k*l+j];
			}
		}
	}*/
	/*for (int i = 0; i < m; i++)
	{
		for (int j = 0; j <l; j++)
		{
			for (int k = 0; k < n; k++)
			{
				pxp[i*l + j] += p1[i*n + k] * p2[k*l + j];
			}
		}
	}*/
	double alpha = 1.0;
	double beta = 0.0;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, l, n, alpha, p1, n, p2, l, beta, pxp, l);
}

void matrix_add_sub(float* leftmatrix, float* rightmatrix,int width,int height,char oper){
	if (oper == '+'){
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				leftmatrix[i*width + j] += rightmatrix[i*width + j];
			}
		}
	}
	else if (oper=='-')
	{
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				leftmatrix[i*width + j] = rightmatrix[i*width + j] - leftmatrix[i*width + j];
			}
		}
	}
}

void swap(float *a, float *b){
	float c;
	c = *a;
	*a = *b;
	*b = c;
}

int inv(float *p, int n){
	int *is, *js, i, j, k, l;
	float temp, fmax;
	is = (int *)malloc(n*sizeof(int));
	js = (int *)malloc(n*sizeof(int));
	for (k = 0; k<n; k++)
	{
		fmax = 0.0;
		for (i = k; i<n; i++)
		for (j = k; j<n; j++)
		{
			temp = fabsf(p[i*n+j]);//find max
			if (temp>fmax)
			{
				fmax = temp;
				is[k] = i; js[k] = j;
			}
		}
		if ((fmax + 1.0) == 1.0)
		{
			free(is); free(js);
			printf("no inv");
			return(0);
		}
		if ((i = is[k]) != k)
		for (j = 0; j<n; j++)
			swap(&p[k*n+j], &p[i*n+j]);//swape pointer
		if ((j = js[k]) != k)
		for (i = 0; i<n; i++)
			swap(&p[i*n+k], &p[i*n+j]);  //swape pointer
		p[k*n+k] = 1.0 / p[k*n+k];
		for (j = 0; j<n; j++)
		if (j != k)
			p[k*n+j] *= p[k*n+k];
		for (i = 0; i<n; i++)
		if (i != k)
		for (j = 0; j<n; j++)
		if (j != k)
			p[i*n+j] = p[i*n+j] - p[i*n+k] * p[k*n+j];
		for (i = 0; i<n; i++)
		if (i != k)
			p[i*n+k] *= -p[k*n+k];
	}
	for (k = n - 1; k >= 0; k--)
	{
		if ((j = js[k]) != k)
		for (i = 0; i<n; i++)
			swap(&p[j*n+i], &p[k*n+i]);
		if ((i = is[k]) != k)
		for (j = 0; j<n; j++)
			swap(&p[j*n+i], &p[j*n+k]);
	}
	free(is);
	free(js);
	return 1;
}

void printArray(float* data, int rows, int cols){
	for (int i = 0; i < rows; i++){
		for (int j = 0; j <cols; j++)
		{
			printf("%lf\t", data[i*cols + j]);
		}
		printf("\n");
		break;
	}
}

float norms(float* data, int rows, int cols, int p){
	float sum = 0.0;
	if (rows == 1){
		if (p == 1)
		{
			for (int j = 0; j < cols; j++){
				sum += fabsf(data[j]);
			}
		}
		else{
			for (int i = 0; i < cols; i++){
				sum += data[i] * data[i];
			}
			sum = sqrtf(sum);
		}

	}
	return sum;
}

//get any one column of data
void getMatrixCol(float* data, float* X, int rows, int cols, int index){
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
				if (j ==index){
					X[i] = data[i*cols + j];
					break;
				}
			}
		}
}

//getMultiColums
void getMatrixCol(float* data,float* X,int rows,int cols,int* bands,int counts){
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			for (int k = 0; k < counts; k++){
				if (j == bands[k]){
					X[i*counts+k] = data[i*cols + j];
					break;
				}
			}
		}
	}
}

void matrix_dot_multi(float b,float* data,int rows,int cols){
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			data[i*cols + j] = b*data[i*cols + j];
		}
	}
}

float min(float* data, int rows,int cols, int colth){
	float MIN_VALUE = data[0];
	for (int i = 0; i < rows; i++){
		if (data[i*cols+colth]<MIN_VALUE){
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

int blluu(float* a, int n, float* l, float* u){
	int i, j, k, w, v, ll;
	for (k = 0; k <= n - 2; k++)
	{
		ll = k*n + k;
		if (fabsf(a[ll]) + 1.0 == 1.0)
		{
			printf("fail\n"); return(0);
		}
		for (i = k + 1; i <= n - 1; i++)
		{
			w = i*n + k; a[w] = a[w] / a[ll];
		}
		for (i = k + 1; i <= n - 1; i++)
		{
			w = i*n + k;
			for (j = k + 1; j <= n - 1; j++)
			{
				v = i*n + j;
				a[v] = a[v] - a[w] * a[k*n + j];
			}
		}
	}
	for (i = 0; i <= n - 1; i++)
	{
		for (j = 0; j<i; j++)
		{
			w = i*n + j; l[w] = a[w]; u[w] = 0.0;
		}
		w = i*n + i;
		l[w] = 1.0; u[w] = a[w];
		for (j = i + 1; j <= n - 1; j++)
		{
			w = i*n + j; l[w] = 0.0; u[w] = a[w];
		}
	}
	return(1);
}

int inv2(float* a, int n){
	const int size = n*n;
	float* L = (float*)malloc(sizeof(float)*size);
	float* U = (float*)malloc(sizeof(float)*size);
	float* r = (float*)malloc(sizeof(float)*size);
	float* u = (float*)malloc(sizeof(float)*size);

	memset(L, 0, sizeof(float)*size);
	memset(U, 0, sizeof(float)*size);
	memset(r, 0, sizeof(float)*size);
	memset(u, 0, sizeof(float)*size);

	int k, i, j;
	int flag = 0;
	float s, t;
	flag = blluu(a, n, L, U);
	if (flag == 1){
		memset(a, 0, sizeof(float)*size);
		/////////////////////求L和U矩阵的逆
		for (i = 0; i<n; i++) /*求矩阵U的逆 */
		{
			u[i*n + i] = 1 / U[i*n + i];//对角元素的值，直接取倒数
			for (k = i - 1; k >= 0; k--)
			{
				s = 0;
				for (j = k + 1; j <= i; j++)
					s = s + U[k*n + j] * u[j*n + i];
				u[k*n + i] = -s / U[k*n + k];//迭代计算，按列倒序依次得到每一个值，
			}
		}
		for (i = 0; i<n; i++) //求矩阵L的逆 
		{
			r[i*n + i] = 1; //对角元素的值，直接取倒数，这里为1
			for (k = i + 1; k<n; k++)
			{
				for (j = i; j <= k - 1; j++)
					r[k*n + i] = r[k*n + i] - L[k*n + j] * r[j*n + i];   //迭代计算，按列顺序依次得到每一个值
			}
		}

		for (i = 0; i<n; i++)
		{
			for (j = 0; j<n; j++)
			{
				for (k = 0; k<n; k++)
				{
					a[i*n + j] += u[i*n + k] * r[k*n + j];
				}
			}
		}


	}
	free(L);
	free(U);
	free(r);
	free(u);
	return flag;
}

void inv3(float* a, int n)
{
	int *ipiv = (int *)malloc(sizeof(int) * n);
	LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, a, n, ipiv);
	LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, a, n, ipiv);
	free(ipiv);
}