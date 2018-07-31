#include "BandSelect.cuh"
#include "Basic.h"

static culaStatus status;
//const +1
EXTERN_C const float CNST1 = 1;
//const 0
EXTERN_C const float CNST0 = 0;
//const -1
EXTERN_C const float CNSTm1 = -1;

//get any one column of data
void getMatrixCol(float* data, float* X, int rows, int cols, int index){
	/*for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			if (j == index){
				X[i] = data[i*cols + j];
				break;
			}
		}
	}*/
	for (int i = 0; i < rows; i++){
		X[i] = data[index + i*cols];
	}
}

//getMultiColums
void getMatrixCol(float* data, float* X, int rows, int cols, int* bands, int counts){
	/*for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			for (int k = 0; k < counts; k++){
				if (j == bands[k]){
					X[i*counts + k] = data[i*cols + j];
					break;
				}
			}
		}
	}*/
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < counts; j++){
			X[i*counts + j] = data[i*cols + bands[j]];
		}
	}
}

void initArray(int* array, int value){
	for (int i = 0; i < BANDNUM; i++){
		array[i] = value;
	}
}

void calculateMAXError(matrix perBlock, float* e,int &start){
	for (int i = 0; i < perBlock.rows; i++){
		for (int j = 0; j < perBlock.cols; j++){
			e[j] += perBlock.data[i*perBlock.cols + j];
		}
	}
	e[start] = 0;
	float MAX_VALUE;
	int index;
	//find initial error position
	for (int j = 0; j <perBlock.cols; j++){
		if (j != start){
			index = j;
			MAX_VALUE = e[j];
			break;
		}
	}
	for (int j = 0; j < perBlock.cols; j++){
		if (e[j]>MAX_VALUE){
			MAX_VALUE = e[j];
			index = j;
		}
	}
	start = index;
}

void bandlp(float* dev_X,float* dev_sz,int rows,int x_cols,int bandnum,float* perBlock,dim3 grids,dim3 blocks){
	//X'X
	float* dev_buffer;//x_cols*x_cols
	float* dev_ipiv;
	float* dev_temp;//x_cols*rows
	float* dev_b;//x_cols*bandnum
	float* dev_Y_hat;//rows*bandnum
	checkCudaErrors(cudaMalloc((void**)&dev_buffer, sizeof(float)*x_cols*x_cols));
	checkCudaErrors(cudaMalloc((void**)&dev_ipiv, sizeof(float)*x_cols));
	checkCudaErrors(cudaMalloc((void**)&dev_temp,sizeof(float)*x_cols*rows));
	checkCudaErrors(cudaMalloc((void**)&dev_b, sizeof(float)*x_cols*bandnum));
	checkCudaErrors(cudaMalloc((void**)&dev_Y_hat,sizeof(float)*rows*bandnum));
	status = culaDeviceSgemm(
		'N',
		'T',
		x_cols, x_cols, rows,
		CNST1,
		dev_X, x_cols,
		dev_X, x_cols,
		CNST0,
		dev_buffer, x_cols
		);
	checkStatus(status);

	status = culaDeviceSgetrf(
		x_cols, x_cols,
		dev_buffer, x_cols,
		(culaDeviceInt*)dev_ipiv
		);
	checkStatus(status);

	status = culaDeviceSgetri(
		x_cols,
		dev_buffer,
		x_cols,
		(culaDeviceInt*)dev_ipiv
		);
	checkStatus(status);

	status = culaDeviceSgemm(
		'T',
		'N',
		rows, x_cols, x_cols,
		CNST1,
		dev_X, x_cols,
		dev_buffer, x_cols,
		CNST0,
		dev_temp, rows
		);
	checkStatus(status);
	//b
	status = culaDeviceSgemm(
		'N',
		'N',
		bandnum, x_cols, rows,
		CNST1,
		dev_sz, bandnum,
		dev_temp, rows,
		CNST0,
		dev_b, bandnum
		);
	checkStatus(status);
	//Y_hat=X*b
	status = culaDeviceSgemm(
		'N',
		'N',
		bandnum, rows, x_cols,
		CNST1,
		dev_b, bandnum,
		dev_X, x_cols,
		CNST0,
		dev_Y_hat, bandnum
		);
	checkStatus(status);
	
	norm_kernel<1024> << <grids, blocks>> >(dev_sz, dev_Y_hat, bandnum, rows, perBlock);
	checkCudaErrors(cudaDeviceSynchronize());
	cudaFree(dev_buffer);
	cudaFree(dev_ipiv);
	cudaFree(dev_temp);
	cudaFree(dev_b);
	cudaFree(dev_Y_hat);
}

int* findStart(matrix* dataSet, float* dev_dataSet){
	int rows = dataSet->rows;
	int bandnum = dataSet->cols;
	float* sz = dataSet->data;
	int* final = (int*)malloc(sizeof(float)*BANDNUM);
	initArray(final, -1);
	int* bstart = (int*)malloc(sizeof(int*)*BANDNUM);
	//float* dev_y;
	float* X = (float*)malloc(sizeof(float)*rows);
	float* dev_X;
	checkCudaErrors(cudaMalloc((void**)&dev_X, sizeof(float)*rows));
	//checkCudaErrors(cudaMalloc((void**)&dev_y,sizeof(float)*rows));
	dim3 blocks(1, 1024);
	dim3 grids(bandnum, DIV_UP(rows, blocks.y));
	float* dev_perBlock;
	checkCudaErrors(cudaMalloc((void**)&(dev_perBlock), sizeof(float)*grids.y*grids.x));
	matrix* h_perBlock=(matrix*)malloc(sizeof(matrix));
	h_perBlock->data = (float*)malloc(sizeof(float)*grids.y*grids.x);
	h_perBlock->rows = grids.y;
	h_perBlock->cols = grids.x;
	int start;
	float* e = (float*)malloc(sizeof(float)*bandnum);
	for (int k = 0; k < BANDNUM; k++){
		memset(bstart, 0, sizeof(int)*BANDNUM);
		bstart[0] = k;
		start = k;
		for (int i = 0; i < BANDNUM; i++){
			getMatrixCol(sz, X, rows, bandnum,start);
			checkCudaErrors(cudaMemcpy(dev_X,X,sizeof(float)*rows,cudaMemcpyHostToDevice));
			memset(e, 0, sizeof(float)*(bandnum));
			bandlp(dev_X, dev_dataSet, rows, 1, bandnum, dev_perBlock,grids,blocks);
			checkCudaErrors(cudaMemcpy(h_perBlock->data,dev_perBlock, sizeof(float)*grids.y*grids.x, cudaMemcpyDeviceToHost));
			calculateMAXError(h_perBlock[0], e,start);
			bool finish_this_band_select = false;
			for (int j = 0; j < BANDNUM; j++){
				if (bstart[j] == start){
					final[k] = start;
					finish_this_band_select = true;
					break;
				}
			}
			if (finish_this_band_select){
				break;
			}
			bstart[i + 1] = start;
		}
	}
	free(X);
	cudaFree(dev_X);
	cudaFree(dev_perBlock);
	free(h_perBlock->data);
	free(h_perBlock);
	free(e);
	/*cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*BANDNUM);
	for (int i = 0; i < BANDNUM; i++){
		cudaStreamCreate(&streams[i]);
		checkCudaErrors(cudaMemcpyAsync(dev_X[i],X[i],sizeof(float)*BANDNUM,cudaMemcpyHostToDevice,streams[i]));
		
	}
	checkCudaErrors(cudaMemcpy(h_perBlock, dev_perBlock, sizeof(matrix*)*BANDNUM, cudaMemcpyDeviceToHost));
	for (int i = 0; i < BANDNUM; i++){
		memset(e, 0, sizeof(float)*bandnum);
		
	}*/
	return final;
}

void calculateMAXError2(matrix perBlock, float* e, int* bands,int index){
	for (int i = 0; i < perBlock.rows; i++){
		for (int j = 0; j < perBlock.cols; j++){
			e[j] += perBlock.data[i*perBlock.cols + j];
		}
	}
	for (int i = 0; i < index; i++){
		int selectedBand = bands[i];
		e[selectedBand] = 0;
	}
	int position = 0;
	bool hasSelected;
	float MAX_VALUE;
	//find initial error position
	for (int j = 0; j < perBlock.cols; j++){
		hasSelected = false;
		for (int k = 0; k < index; k++){
			if (j == bands[k]){
				hasSelected = true;
				break;
			}
		}
		if (hasSelected)continue;
		MAX_VALUE = e[j];
		position = j;
		break;
	}
		
	for (int j = 0; j < perBlock.cols; j++){
		if (e[j]>MAX_VALUE){
			MAX_VALUE = e[j];
			position = j;
		}
	}
	bands[index] = position;
}

int* bs1(matrix* dataSet, int ind, float* dev_dataSet){
	const int height = dataSet->rows;
	const int bandnum = dataSet->cols;
	float* sz = dataSet->data;
	int* bands = (int*)malloc(sizeof(int)*BANDNUM);
	bands[0] = ind;
	int selectedCount = 1;
	dim3 blocks(1, 1024);
	dim3 grids(bandnum, DIV_UP(height, blocks.y));
	float* dev_perBlock;
	checkCudaErrors(cudaMalloc((void**)&(dev_perBlock), sizeof(float)*grids.y*grids.x));
	matrix* h_perBlock = (matrix*)malloc(sizeof(matrix));
	h_perBlock->data = (float*)malloc(sizeof(float)*grids.y*grids.x);
	h_perBlock->rows = grids.y;
	h_perBlock->cols = grids.x;
	float* e = (float*)malloc(sizeof(float)*bandnum);
	for (int i = 0; i < BANDNUM - 1; i++){
		float* X = (float*)malloc(sizeof(float)*height*(i + 1));
		getMatrixCol(sz, X, height, bandnum, bands, selectedCount);
		memset(e, 0, sizeof(float)*(bandnum));
		float* dev_X;
		checkCudaErrors(cudaMalloc((void**)&dev_X, sizeof(float)*height*(i + 1)));
		checkCudaErrors(cudaMemcpy(dev_X, X, sizeof(float)*height*(i + 1), cudaMemcpyHostToDevice));
		bandlp(dev_X, dev_dataSet, height, i + 1, bandnum, dev_perBlock,grids,blocks);
		checkCudaErrors(cudaMemcpy(h_perBlock->data, dev_perBlock, sizeof(float)*grids.y*grids.x, cudaMemcpyDeviceToHost));
		calculateMAXError2(h_perBlock[0],e,bands,i+1);
		selectedCount++;
		free(X);
		cudaFree(dev_X);
	}
	free(h_perBlock->data);
	free(h_perBlock);
	cudaFree(dev_perBlock);
	return bands;
}


EXTERN_C int* bandSelect(matrix* dataSet,float* dev_dataSet){
	int* ind=findStart(dataSet, dev_dataSet);
	//make elements of ind +1
	for (int i = 0; i < BANDNUM; i++){
		ind[i] += 1;
	}
	int* a = (int*)malloc(sizeof(int)*(dataSet->cols + 1));
	memset(a, 0, sizeof(int)*(dataSet->cols + 1));
	for (int i = 0; i < BANDNUM; i++){
		a[ind[i]]++;
	}
	int MAX_COUNT = a[0];
	int position = 0;
	for (int i = 0; i < dataSet->cols + 1; i++){
		if (a[i]>MAX_COUNT){
			a[i] = MAX_COUNT;
			position = i;
		}
	}
	free(a);
	int firstBand = position - 1;
	int* bsn;
	if (firstBand == -1)
	{
		bsn = bs1(dataSet, BANDNUM - 1,dev_dataSet);
	}
	else{
		bsn = bs1(dataSet, firstBand,dev_dataSet);
	}
	return bsn;
}