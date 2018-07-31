#include "rxGDll.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cula_lapack_device.h"
#include "cula_blas_device.h"
#include "helper_cuda.h"

#define DIV_UP(a, b) (((a) + (b - 1)) / b)

#define DIV(a, b) (a / b)

const double CNST1 = 1.0;
const double CNST0 = 0.0;

EXTERN_C void checkStatus(culaStatus status)
{
	char buf[256];
	if (!status)
		return;
	culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
	printf("%s\n", buf);
	culaShutdown();
	system("pause");
	exit(EXIT_FAILURE);
}

__global__ void addSub_kernel(float* dev_Xl, float* dev_Xr, int width, int height, char opera)
{
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if ((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		if (opera == '+')
		{
			dev_Xl[index_in] += dev_Xr[index_in];
		}
		else if (opera == '-')
		{
			dev_Xl[index_in] = dev_Xl[index_in] - dev_Xr[index_in];
		}
	}
}



__global__ void sumReduction_kernel(float* out, float* in, int m, int n, int blockNumPerRow)
{
	extern __shared__ float temp1[];

	//int N = n / blockDim.x;
	float sum = 0.0;
	if (blockIdx.x < blockNumPerRow) {
		for (int i = threadIdx.x; i < n; i += blockDim.x)
		{
			int index = blockIdx.y * n + i;
			sum += in[index];
		}
		temp1[threadIdx.x] = sum;
		__syncthreads();
	}

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (threadIdx.x < offset) {
			temp1[threadIdx.x] += temp1[threadIdx.x + offset];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)
	{
		out[blockIdx.y] = temp1[0] / n;
	}
}

__global__ void sub1_kernel(float* dev_x_o, float* dev_x_i, float* dev_mean, int m, int n)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index = idy * n + idx;
	if ((idx < n) && (idy < m))
	{
		dev_x_o[index] = dev_x_i[index] - dev_mean[idy];
	}
}

__global__ void dist_kernel(float* dev_A, float* dev_B, float* out, int m, int n)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int A_index = tid;
	int B_index = tid;
	float sum = 0.0;
#pragma unroll
	for (int i = 0; i < n; i++)
	{
		sum += dev_A[A_index] * dev_B[B_index];
		A_index += m;
		B_index += m;
	}
	out[tid] = sum;
}

__global__ void bandSub_kernel(float *p, float *res, int m, int n, int k)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int back_index = k * n + tid;
	int ford_index = tid;
#pragma unroll
	for (int i = 0; i < m - k; i++)
	{
		res[i * n + tid] = p[back_index] - p[ford_index];
		if (res[i * n + tid] < 0) {
			res[i * n + tid] = fabs(res[i * n + tid]);
		}
		back_index += n;
		ford_index += n;
	}
}

__global__ void bandSub1_kernel(float *p, float *res, int m, int n, int k)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = idy * n + idx;
	int back_tid = tid + k * n;
	if (idx < n && idy < m)
	{
		res[tid] = fabs(p[back_tid] - p[tid]);//可能有点问题
	}

}



__global__ void order2_kernel(float *p, float *res, int m, int n, int k)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int back_index = 2 * k * n + tid;
	int ford_index = tid;
	int midl_index = k * n + tid;

#pragma unroll
	for (int i = 0; i < (m - 2 * k); i++)
	{
		res[i * n + tid] = p[back_index] + p[ford_index] - 2 * p[midl_index];  //
		if (res[i * n + tid] < 0) {
			res[i * n + tid] = fabs(res[i * n + tid]);
		}
		back_index += n;
		ford_index += n;
		midl_index += n;
	}
}

float rxGpu(char *path1, char *var1, char *path2, char *var2)
{
	matrix *X = (matrix*)malloc(sizeof(matrix));
	matrix *MaskS = (matrix*)malloc(sizeof(matrix));
	readFile(path1, var1, X);
	readFile(path2, var2, MaskS);
	float *Mask = MaskS->data;
	int r = MaskS->rows;
	int s = MaskS->cols;
	int M = X->rows;
	int N = X->cols;
	float *anomaly_map = (float*)malloc(sizeof(float)*r*s);
	float *normal_map = (float*)malloc(sizeof(float)*r*s);
	map(Mask, anomaly_map, r, s, 'a');
	map(Mask, normal_map, r, s, 'n');
	int o = 5000; int l = 4096;
	float *taus = (float*)malloc(sizeof(float)*(o + 1));
	float *anomaly_map_rx = (float*)malloc(sizeof(float) * N);
	float *PF1 = (float*)malloc(sizeof(float) * o);
	float *PD1 = (float*)malloc(sizeof(float) * o);
	float *X1 = (float*)malloc(sizeof(float) * l);
	float *X2 = (float*)malloc(sizeof(float) * l);
	float *a_n = (float*)malloc(sizeof(float) * N);
	float *a_a = (float*)malloc(sizeof(float) * N);
	float *re = (float*)malloc(sizeof(float) * l);

	printf("RX anomaly detection on GPU...\n");
	printf("compute the accuracy of detection...\n");
	const int n1 = 1;
	const int m = X->rows;
	const int n = X->cols;
	float CNSTn = 0.0;
	CNSTn = 1.0 / n;
	int BLOCK = DIV_UP(n, 1024);
	int k = 1;
	int M1 = m - k;
	int T = 100;
	float *dev_Xsub;
	float *dev_Xmean;
	float *dev_XFinal;
	float *dev_sigma;
	float *dev_ipiv;
	float *dev_X;
	float *dev_buffer;
	float* dev_dist;
	float * host_dist = (float *)malloc(sizeof(float) * n);
	checkCudaErrors(cudaMalloc((void**)&dev_X, m*n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_Xsub, M1 * n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_Xmean, M1 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_XFinal, M1*n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_sigma, M1*M1 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_ipiv, M1 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_buffer, M1*n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_dist, sizeof(float)*n));
	culaStatus status;
	status = culaInitialize();
	checkStatus(status);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	float start2, stop2;
	float totalTime = 0.0;

	for (int i = 0; i < T; i++)
	{
		checkCudaErrors(cudaMemcpy(dev_X, X->data, m*n * sizeof(float), cudaMemcpyHostToDevice));
		start2 = clock();
		dim3 Grid(BLOCK, M1);
		dim3 Block(1024, 1);
		bandSub1_kernel << <Grid, Block >> > (dev_X, dev_Xsub, m, n, k);
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));
		cudaDeviceSynchronize();
		stop2 = clock();
		float perTime = stop2 - start2;
		totalTime = totalTime + perTime;
		dim3 sumGrid(BLOCK, M1);
		dim3 sumBlock(1024, 1);
		int sharedSize = sumBlock.x * sizeof(float);
		sumReduction_kernel << <sumGrid, sumBlock, sharedSize, 0 >> > (dev_Xmean, dev_Xsub, M1, n, BLOCK);
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));
		sub1_kernel << <sumGrid, sumBlock >> > (dev_XFinal, dev_Xsub, dev_Xmean, M1, n);
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));

		//sigma = (x*x')/N = (x'*x)/N
		status = culaDeviceSgemm(
			'T',
			'N',
			M1, M1, n,
			CNSTn,
			dev_XFinal, n,
			dev_XFinal, n,
			CNST0,
			dev_sigma, M1
		);
		checkStatus(status);

		status = culaDeviceSgetrf(M1, M1, dev_sigma, M1, (culaDeviceInt*)dev_ipiv);
		checkStatus(status);
		status = culaDeviceSgetri(M1, dev_sigma, M1, (culaDeviceInt*)dev_ipiv);
		checkStatus(status);
		printf("%s\n", "CULA inverse had done!");

		//loop multimatrix 
		status = culaDeviceSgemm(
			'N',
			'T',
			n, M1, M1,
			CNST1,
			dev_XFinal, n,
			dev_sigma, M1,
			CNST0,
			dev_buffer, n
		);
		checkStatus(status);
		dist_kernel << <BLOCK, 1024 >> > (dev_buffer, dev_XFinal, dev_dist, n, M1);
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));

		checkCudaErrors(cudaMemcpy(host_dist, dev_dist, sizeof(float) * n, cudaMemcpyDeviceToHost));
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Get derivative features time use= %.3fms\n", totalTime / T);
	printf("computation of M distance time use with first order：%.3fms\n", elapsedTime / T);
	culaShutdown();

	float start1 = clock();
	float r_max = max_value(host_dist, 1, N);
	linspace(r_max, o, taus);


	for (int i = 0; i < o; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if ((host_dist[j]) > taus[i]) {
				anomaly_map_rx[j] = 1;
			}
			else {
				anomaly_map_rx[j] = 0;
			}
		}

		And(anomaly_map_rx, normal_map, a_n, 1, N);
		And(anomaly_map_rx, anomaly_map, a_a, 1, N);
		float temp1 = sum(a_n, 1, N);
		float temp2 = sum(a_a, 1, N);
		float temp3 = sum(normal_map, 1, N);
		float temp4 = sum(anomaly_map, 1, N);
		PF1[i] = temp1 / temp3;
		PD1[i] = temp2 / temp4;
	}
	for (int i = 0; i < l; i++)
	{
		X1[i] = PF1[i] - PF1[i + 1];
		X2[i] = PD1[i] + PD1[i + 1];
	}

	matrix_dotmulti(X1, X2, re, 1, l);
	float area = sum(re, 1, l);
	float area1 = area / 2;
	float end = clock();
	float AUCtime = end - start1;
	printf("The time to compute AUC use: %.3fms\n", AUCtime);

	printf("---------------\n");
	free(host_dist);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("RX anoamly detection on GPU has done...\n");

	cudaFree(dev_dist);
	cudaFree(dev_buffer);
	cudaFree(dev_ipiv);
	cudaFree(dev_sigma);
	cudaFree(dev_XFinal);
	cudaFree(dev_Xmean);
	cudaFree(dev_X);
	cudaFree(dev_Xsub);

	free(taus);
	free(anomaly_map_rx);
	free(PF1);
	free(PD1);
	free(X1);
	free(X2);
	free(a_n);
	free(a_a);
	free(re);
	return area1;
}


int main()
{
	char *path1 = "../../../SPECTIR.mat";  
	char *path2 = "../../../mask.mat";
	char *var1 = "X";
	char *var2 = "mask";
	float AUC = rxGpu(path1, var1, path2, var2);
	printf("Area under ROC is %f\n", AUC);
	getchar();
	return 0;
}