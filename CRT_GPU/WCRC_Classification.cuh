#include "MatOperation.h"
#ifndef BASIC_CUDA_HEAD
#define BASIC_CUDA_HEAD
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include<helper_cuda.h>
#endif

#define BLOCK_DIM 16
#define threads 16
#define ThreadHandleNum 4

//#define Z_rows 145 
//#define Z_cols 145
//#define Z_dimension 220



__global__ void LBP_EXTRACTION(float* dev_Z, float* dev_LBP_data, int width, int height){
	float pixVal;
	unsigned int val;
	float center;
	unsigned int pixcoord;
	unsigned int xIndex = blockIdx.x *blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y *blockDim.y + threadIdx.y;
	if ((xIndex < width - 1) && (yIndex < height - 1))
	{
		unsigned int index_in = yIndex * width + xIndex;
		if (xIndex>0 && yIndex>0){
			//(float)(x,y)
			center = dev_Z[index_in];
			//(float)(x-1,y-1)
			pixcoord = (yIndex - 1)*width + (xIndex - 1);
			pixVal = dev_Z[pixcoord];
			val = (center < pixVal);
			//(float)(x,y-1)
			pixcoord = (yIndex - 1)*width + xIndex;
			pixVal = dev_Z[pixcoord];
			val |= (center < pixVal) << 1;
			//(float)(x+1,y-1)
			pixcoord = (yIndex - 1)*width + (xIndex + 1);
			pixVal = dev_Z[pixcoord];
			val += (center < pixVal) << 2;
			//(float)(x+1,y)
			pixcoord = yIndex*width + (xIndex + 1);
			pixVal = dev_Z[pixcoord];
			val += (center < pixVal) << 3;
			//(float)(x+1,y+1)
			pixcoord = (yIndex + 1)*width + (xIndex + 1);
			pixVal = dev_Z[pixcoord];
			val += (center < pixVal) << 4;
			//(float)(x,y+1)
			pixcoord = (yIndex + 1)*width + xIndex;
			pixVal = dev_Z[pixcoord];
			val += (center < pixVal) << 5;
			//(float)(x-1,y+1)
			pixcoord = (yIndex + 1)*width + (xIndex - 1);
			pixVal = dev_Z[pixcoord];
			val += (center < pixVal) << 6;
			//(float)(x-1,y)
			pixcoord = yIndex*width + (xIndex - 1);
			pixVal = dev_Z[pixcoord];
			val += (center < pixVal) << 7;

			dev_LBP_data[index_in] = (float)val;
		}

	}
}

__global__ void normsKernel(float* dataTest, float* classTrain, float* norms, int length){
	float* dataTe = dataTest + length* blockIdx.x;
	float* dataTr = classTrain + length* threadIdx.x;
	float* norma = norms + blockIdx.x*blockDim.x + threadIdx.x;
	float sum = 0.0;
#pragma unroll
	for (int i = 0; i < length; i++)
	{
		float t = dataTe[i] - dataTr[i];
		sum += t*t;
	}
	*norma = sum;
}

__global__ void normsKernel2(float* dataTest, float* classTrain, float* norms, int len,int height,int width){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if (yIndex < height && xIndex<width){
		float* norma = norms +yIndex*width+xIndex;
		float* dataTe = dataTest + yIndex*len;
		float* dataTr = classTrain + xIndex*len;
		float sum = 0.0;
#pragma unroll
		for (int i = 0; i < len; i++)
		{
			float t = dataTe[i] - dataTr[i];
			sum += t*t;
		}
		*norma = sum;
	}
}

__global__  void geagKernel(float* A, float* x, int m, float lambda)
{
	int tid = threadIdx.x;
	if (tid < m)
	{
		for (int i = tid*m; i < (tid + 1)*m; i++){
			if (i == (tid*m + tid))
			{
				A[i] = lambda*x[tid];
			}
			else
			{
				A[i] = 0;
			}
		}
	}
}

__global__ void onesKernel(float* A, int width, int height, float param){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if ((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		if (xIndex == yIndex){
			A[index_in] = 1.0*param;
		}
		else{
			A[index_in] = 0.0;
		}
	}
}

__global__ void addSubKernel(float* dev_Xsq, float* buffer, int width, int height, char opera){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if ((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		if (opera == '+')
		{
			buffer[index_in] += dev_Xsq[index_in];
		}
		else if (opera == '-'){
			dev_Xsq[index_in] = buffer[index_in] - dev_Xsq[index_in];
			if (dev_Xsq[index_in] < 0)dev_Xsq[index_in] = fabsf(dev_Xsq[index_in]);
		}

	}
}

__global__ void signment_kernel(float* dev_buf, float* dev_weights, int offset, int trainOffset, int trainSample_rows){
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<blockDim.x*gridDim.x)
		dev_buf[tid] = dev_weights[trainOffset+ blockIdx.x*trainSample_rows + threadIdx.x];
}

__global__ void signment_kernel2(float* dev_buf, float* dev_weights, int offset, int trainOffset, int trainSample_rows, int width, int height){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if (xIndex < width && yIndex < height){
		unsigned int tid = yIndex * width + xIndex;
		dev_buf[tid] = dev_weights[trainOffset + yIndex*trainSample_rows + xIndex];
	}
}


__global__ void sigment2_kernel(float*dev_weight, float* deta, int width){
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<width)
	{
		dev_weight[tid] = deta[tid];
	}

}

__global__ void histoKernel(int* buffer, long size, int* histo, int label)
{
	__shared__  int temp[threads];
	temp[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	while (i<size)
	{
		atomicAdd(&(temp[buffer[i]]), 1);
		i += offset;
	}

	__syncthreads();
	histo[0] = temp[label];
}

__global__ void norm_dist_Y_kernel(float* dev_norm_dist_Y, float** dev_dist_Y, int Nt){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	float sum = 0.0;
	for (int i = 0; i < Nt; i++){
		sum += dev_dist_Y[threadIdx.x][blockIdx.x*Nt + i] * dev_dist_Y[threadIdx.x][blockIdx.x*Nt + i];
	}
	dev_norm_dist_Y[tid] = sqrtf(sum);
}

//--------------------------------------------------------------------------------
__global__ void distNorms_kernel2(float* dev_dataTest,float* dev_dist_Y,float* dev_perBlock,int width,int no_class,int class_th){
	extern __shared__ float temp[];
	if (threadIdx.x < width){
		unsigned int index_in = blockIdx.x* width +threadIdx.x;
		temp[threadIdx.x] = dev_dataTest[index_in] - dev_dist_Y[index_in];
		temp[threadIdx.x] = temp[threadIdx.x] * temp[threadIdx.x];
	}
	else{
		temp[threadIdx.x] = 0;
	}
	__syncthreads();
#pragma unroll
	for (int offset = blockDim.x / 2; offset>0; offset >>= 1){
		if (threadIdx.x < offset){
			temp[threadIdx.x] += temp[threadIdx.x + offset];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)
		dev_perBlock[blockIdx.x*no_class +class_th] = temp[0];
}




__global__ void distNorms_kernel(float* dev_dataTest, float* dev_dist_Y, float* dev_perBlock, int width, int height, int no_class, int class_th){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	extern __shared__ float temp[];
	if (xIndex < width && yIndex< height){
		unsigned int index_in = yIndex* width + xIndex;
		temp[threadIdx.x] = dev_dataTest[index_in] - dev_dist_Y[index_in];
		temp[threadIdx.x] = temp[threadIdx.x] * temp[threadIdx.x];
	}
	else{
		temp[threadIdx.x] = 0;
	}
	__syncthreads();
#pragma unroll
	for (int offset = blockDim.x / 2; offset>0; offset >>= 1){
		if (threadIdx.x < offset){
			temp[threadIdx.x] += temp[threadIdx.x + offset];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)
		dev_perBlock[blockIdx.y*no_class + class_th] = temp[0];
}