#define BLOCK_SIZE 16
#include<cuda_runtime.h>
#include"device_launch_parameters.h"

#define DIV_UP(a,b) (a + b - 1) / b

__global__ void sumReduction1(double *out, double *in, int m, int n)
{
	extern __shared__ double temp[];
	double sum = 0.0;
	for (int i = threadIdx.x; i < n; i += blockDim.x)
	{
		int index = blockIdx.y * n + i;
		sum += in[index];
	}
	temp[threadIdx.x] = sum;
	__syncthreads();
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (threadIdx.x < offset)
		{
			temp[threadIdx.x] += temp[threadIdx.x + offset];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)
		out[blockIdx.y] = temp[0] / n;
}

__global__ void sumReduction1_unroll(double *out, double *in, int m, int n)
{
	extern __shared__ double temp[];
	double sum = 0.0;
	for (int i = threadIdx.x; i < n; i += blockDim.x)
	{
		int index = blockIdx.y * n + i;
		sum += in[index];
	}
	temp[threadIdx.x] = sum;
	__syncthreads();
	for (int offset = blockDim.x / 2; offset > 32; offset >>= 1)
	{
		if (threadIdx.x < offset)
		{
			temp[threadIdx.x] += temp[threadIdx.x + offset];
		}
		__syncthreads();
	}
	if (threadIdx.x < 32)
	{
		volatile double *aveRow = temp;
		if (blockDim.x > 32) aveRow[threadIdx.x] += aveRow[threadIdx.x + 32];
		aveRow[threadIdx.x] += aveRow[threadIdx.x + 16];
		aveRow[threadIdx.x] += aveRow[threadIdx.x + 8];
		aveRow[threadIdx.x] += aveRow[threadIdx.x + 4];
		aveRow[threadIdx.x] += aveRow[threadIdx.x + 2];
		aveRow[threadIdx.x] += aveRow[threadIdx.x + 1];
		if (threadIdx.x == 0)
		{
			volatile double *aveRow = temp;
			out[blockIdx.y] = aveRow[0] / n;
		}
	}
}


template<unsigned int numThreads>
__global__ void sumReduction1_template(double *out, double *in, int m, int n)
{
	extern __shared__ double temp[];
	double sum = 0.0;
	for (int i = threadIdx.x; i < n; i += blockDim.x)
	{
		int index = blockIdx.y * n + i;
		sum += in[index];
	}
	temp[threadIdx.x] = sum;
	__syncthreads();


	if (numThreads >= 1024)
	{
		if (threadIdx.x < 512)
			temp[threadIdx.x] += temp[threadIdx.x + 512];
		__syncthreads();
	}
	if (numThreads >= 512)
	{
		if (threadIdx.x < 256)
			temp[threadIdx.x] += temp[threadIdx.x + 256];
		__syncthreads();
	}
	if (numThreads >= 256)
	{
		if (threadIdx.x < 128)
			temp[threadIdx.x] += temp[threadIdx.x + 128];
		__syncthreads();
	}
	if (numThreads >= 128)
	{
		if (threadIdx.x < 64)
			temp[threadIdx.x] += temp[threadIdx.x + 64];
		__syncthreads();
	}
	if (threadIdx.x < 32)
	{
		volatile double *aveRow = temp;
		if (numThreads >= 64)
			aveRow[threadIdx.x] += aveRow[threadIdx.x + 32];
		if (numThreads >= 32)
			aveRow[threadIdx.x] += aveRow[threadIdx.x + 16];
		if (numThreads >= 16)
			aveRow[threadIdx.x] += aveRow[threadIdx.x + 8];
		if (numThreads >= 8)
			aveRow[threadIdx.x] += aveRow[threadIdx.x + 4];
		if (numThreads >= 4)
			aveRow[threadIdx.x] += aveRow[threadIdx.x + 2];
		if (numThreads >= 2)
			aveRow[threadIdx.x] += aveRow[threadIdx.x + 1];
		if (threadIdx.x == 0)
			out[blockIdx.y] = aveRow[0] / n;
	}
}

void templateReduction(double *out, double *in, int m, int n, int numThreads, dim3 grid, dim3 block)
{
	int sharedSize = block.x * sizeof(double);
	switch (numThreads)
	{
	case    1: sumReduction1_template<   1> << <grid, block, sharedSize >> > (out, in, m, n); break;
	case    2: sumReduction1_template<   2> << <grid, block, sharedSize >> > (out, in, m, n); break;
	case    4: sumReduction1_template<   4> << <grid, block, sharedSize >> > (out, in, m, n); break;
	case    8: sumReduction1_template<   8> << <grid, block, sharedSize >> > (out, in, m, n); break;
	case   16: sumReduction1_template<  16> << <grid, block, sharedSize >> > (out, in, m, n); break;
	case   32: sumReduction1_template<  32> << <grid, block, sharedSize >> > (out, in, m, n); break;
	case   64: sumReduction1_template<  64> << <grid, block, sharedSize >> > (out, in, m, n); break;
	case  128: sumReduction1_template< 128> << <grid, block, sharedSize >> > (out, in, m, n); break;
	case  256: sumReduction1_template< 256> << <grid, block, sharedSize >> > (out, in, m, n); break;
	case  512: sumReduction1_template< 512> << <grid, block, sharedSize >> > (out, in, m, n); break;
	case 1024: sumReduction1_template<1024> << <grid, block, sharedSize >> > (out, in, m, n); break;
	}
}


__global__ void matrixMul(double *dev_A, double *dev_B, double *dev_C, int m, int n, int k)
{
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
	double cValue = 0;
	if (idx < m && idy < k)
	{
		for (int i = 0; i < n; i++)
		{
			cValue += dev_A[idy*n + i] * dev_B[i * m + idx];
		}
		dev_C[idy * k + idx] = cValue;
	}
}

//multi with shared memory 
__global__ void matrixMulS(double *dev_A, double *dev_B, double *dev_C, int m, int n, int k)
{
	unsigned int by = blockIdx.y, bx = blockIdx.x;
	unsigned int ty = threadIdx.y, tx = threadIdx.x;
	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;
	__shared__ double Ax[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double Bx[BLOCK_SIZE][BLOCK_SIZE];
	double cValue = 0;
	for (int t = 0; t < DIV_UP(n, BLOCK_SIZE); t++)
	{
		if (row < m && t * BLOCK_SIZE + tx < n)
			Ax[tx][ty] = dev_A[row * n + t * BLOCK_SIZE + tx];
		else
			Ax[tx][ty] = 0.0;
		if (t * BLOCK_SIZE + ty < n && col < k)
			Bx[tx][ty] = dev_B[(t * BLOCK_SIZE + ty) * k + col];
		else
			Bx[tx][ty] = 0.0;
		__syncthreads();
		for (int i = 0; i < BLOCK_SIZE; i++)
			cValue += Ax[i][ty] * Bx[tx][i];
		__syncthreads();
		if (row < m && col < k)
			dev_C[row * k + col] = cValue;
	}
}

//转置貌似有问题，下面先直接考虑乘法的问题了
__global__ void transpose(double *dev_in, double *dev_out, int m, int n)
{
	int in_corner_i = blockIdx.x * BLOCK_SIZE, in_corner_j = blockIdx.y * BLOCK_SIZE;
	int out_corner_i = blockIdx.y * BLOCK_SIZE, out_corner_j = blockIdx.x * BLOCK_SIZE;
	int x = threadIdx.x, y = threadIdx.y;
	__shared__ double temp[BLOCK_SIZE][BLOCK_SIZE];
	temp[y][x] = dev_in[(in_corner_i + x) + (in_corner_j + y) * n];
	__syncthreads();
	dev_out[(out_corner_i + x) + (out_corner_j + y) * n] = temp[x][y];
}