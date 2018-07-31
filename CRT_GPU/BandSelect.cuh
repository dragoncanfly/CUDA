#include <stdio.h>
#include "MatOperation.h"
#include<math.h>
#ifndef BASIC_CUDA_HEAD
#define BASIC_CUDA_HEAD
#include <cuda_runtime.h>
#include<helper_cuda.h>
#include "device_launch_parameters.h"
#endif
#define BANDNUM 10
#ifndef DIV_UP
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#endif

template<int block_y>
__global__ void norm_kernel(float* dev_sz,float* dev_Y_hat,int width,int height,float* perBlock){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ float temp[block_y];
	if ((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		temp[threadIdx.y] = dev_sz[index_in]-dev_Y_hat[index_in];
		temp[threadIdx.y] = temp[threadIdx.y]*temp[threadIdx.y];
	}
	else{
		temp[threadIdx.y] = 0;
	}
	__syncthreads();
		
#pragma unroll
		for (int offset = blockDim.y/2;offset>0;offset>>=1){
			if (threadIdx.y < offset){
				temp[threadIdx.y] += temp[threadIdx.y + offset];
			}
			__syncthreads();
		}
		if (threadIdx.y == 0)
			perBlock[blockIdx.y*width + blockIdx.x] = temp[0];
}

