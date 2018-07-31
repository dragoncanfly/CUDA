#include"matOpera.h"
#include"kernelRed.cuh"
#include<helper_cuda.h>

/************test sumReduction************/
int main()
{
	matrix *dataSet = (matrix*)malloc(sizeof(matrix));
	readFile("../../../hydice.mat", "X", dataSet);
	int m = dataSet->rows;
	int n = dataSet->cols;
	double *data = (double*)malloc(sizeof(double)*m*n);
	memcpy(data, dataSet->data, sizeof(double)*m*n);
	double *reC = (double*)malloc(sizeof(double)*m);
	zeros(reC, m, 1);
	float t1 = clock();
	aveRow(data, reC, m, n);
	float t2 = clock();
	printf("Serialize time is %f ms\n", t2 - t1);
	double *dev_data;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	checkCudaErrors(cudaMalloc((void**)&dev_data, sizeof(double)*m*n));
	checkCudaErrors(cudaMemcpy(dev_data, data, sizeof(double)*m*n, cudaMemcpyHostToDevice));
	double *dev_result1;
	checkCudaErrors(cudaMalloc((void**)&dev_result1, sizeof(double) * m));
	double *res = (double*)malloc(sizeof(double) * m);
	double *res2 = (double*)malloc(sizeof(double) * m);
	dim3 block(512, 1);//dim3 block(1024, 1);
	dim3 grid(8, m);//dim3 grid(4, m);
	int sharedSize = sizeof(double)*block.x;
	int nP = n / block.x;
	double *dev_partial;
	checkCudaErrors(cudaMalloc((void**)&dev_partial, sizeof(double)*m*nP));
	cudaEventRecord(start, 0);
	unsigned int numThreads = block.x;
	//sumReduction1 << <grid, block, sharedSize, 0>> > (dev_result1, dev_data, m, n);
	//sumReduction1_unroll << <grid, block, sharedSize, 0 >> > (dev_result1, dev_data, m, n);
	templateReduction(dev_result1, dev_data, m, n, numThreads, grid, block);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	float elapsedTime = 0.0;
	cudaEventElapsedTime(&elapsedTime, start, end);
	checkCudaErrors(cudaMemcpy(res, dev_result1, sizeof(double)*m, cudaMemcpyDeviceToHost));
	//writeFile(res, m, 1, "D:\\Eg\\cuda\\kernel_data_test\\re2.mat", "re2");
	printf("Computation done!\n");
	printf("The computation time is %f ms", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	getchar();
	return 0;
}
/**********************************************/

