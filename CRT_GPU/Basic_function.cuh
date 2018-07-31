//#include<time.h>
//#include<math.h>
#ifndef BASIC_CUDA_HEAD
#define BASIC_CUDA_HEAD
#include <cuda_runtime.h>
#include<helper_cuda.h>
#include "device_launch_parameters.h"
#endif

#define ThreadHandleNum 4
#ifndef DIV_UP
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#endif


static
__global__ void findMaxValue_kernel(float* dev_sz, int width, int height, float* perBlock){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	extern __shared__ float temp[];
	if ((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		temp[threadIdx.y] = dev_sz[index_in];
	}
	else{
		temp[threadIdx.y] = 0;
	}
	__syncthreads();

#pragma unroll
	for (int offset = blockDim.y / 2; offset>0; offset >>= 1){
		if (threadIdx.y < offset){
			if (temp[threadIdx.y]<temp[threadIdx.y+offset])
			temp[threadIdx.y] = temp[threadIdx.y + offset];
		}
		__syncthreads();
	}
	if (threadIdx.y == 0)
		perBlock[blockIdx.y*width + blockIdx.x] = temp[0];
}

static
__global__ void normalization_kernel(float* Mat, float* Value, int width, int height){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int threadY = DIV_UP(height,ThreadHandleNum);
	if ((xIndex < width) && (yIndex < threadY))
	{
		unsigned int offset = yIndex*ThreadHandleNum * width + xIndex;
#pragma unroll
		for (int i = 0; i < ThreadHandleNum; i++){
			if ((yIndex*ThreadHandleNum + i) < height){
				Mat[offset + i*width] = Mat[offset + i*width] / Value[0];
			}
		}

	}
}