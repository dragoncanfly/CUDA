#include <stdio.h>
#include "MatOperation.h"
#include<time.h>
#include<math.h>
#ifndef BASIC_CUDA_HEAD
#define BASIC_CUDA_HEAD
#include <cuda_runtime.h>
#include<helper_cuda.h>
#include "device_launch_parameters.h"
#endif
//#define EVERAGE_TIME

#define checkHostStatus(hostStatus) _checkHostStatus(hostStatus)


#define COMPUTENUM 10
#define Z_rows 145 
#define Z_cols 145
#define Z_dimension  200
#define threads  16

//#define SAMPLE_NUM 50
#define ThreadHandleNum 4


#define BLOCK_DIM 16
#define threads 16
#ifndef DIV_UP
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#endif

//declare
EXTERN_C void WCRC_Classification(array_int* CTest,array_int* CTrain, int no_class, int Testnum, int Nt, float* dev_dataTrain, float* dev_dataTest, int trainSample_rows);
EXTERN_C int* bandSelect(matrix* dataSet,float* dev_dataSet);
EXTERN_C Mapping* getMapping(int samples);
EXTERN_C void LBP_feature_global(matrix* Feature_P,float* dev_LBP_feature ,matrix* dataSet, int* bsn, Mapping* mapping, int radius, matrix* map, int num_point, int W0);
	
EXTERN_C void WCRC_Classification_part(array_int* CTrain, int no_class, int Testnum, int Nt, float* dev_dataTrain, float* dev_dataTest, int trainSample_rows, float* dev_Xsq, float param, int* classlabel, int TestNumberOffset);

//__global__ void normalization_kernel(float* Mat, float Value, int width, int height){
//	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
//	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
//	unsigned int threadY = (height + (ThreadHandleNum - 1)) / ThreadHandleNum;
//	if ((xIndex < width) && (yIndex < threadY))
//	{
//		unsigned int offset = yIndex*ThreadHandleNum * width + xIndex;
//#pragma unroll
//		for (int i = 0; i < ThreadHandleNum; i++){
//			if ((yIndex*ThreadHandleNum + i) < height){
//				Mat[offset + i*width] = Mat[offset + i*width] / Value;
//			}
//		}
//
//	}
//}

__global__ void trans_kernel(float* dev_mat_src, float* dev_mat_dst, int width, int height){
	__shared__ float block[BLOCK_DIM][BLOCK_DIM];

	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if ((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = dev_mat_src[index_in];
	}

	__syncthreads();

	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if ((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		dev_mat_dst[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

template<int threadNum>
__global__ void trans_kernel2(float* dev_mat_src, float* dev_mat_dst, int width, int height){
	__shared__ float block[BLOCK_DIM][BLOCK_DIM*threadNum];

	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	unsigned int pitch_X = DIV_UP(width, threadNum);
	if ((xIndex < pitch_X) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex*threadNum;
#pragma unroll
		for (int i = 0; i < threadNum;i++)
			if (xIndex*threadNum+i<width)
				block[threadIdx.y][threadIdx.x*threadNum+i] = dev_mat_src[index_in+i];
	}

	__syncthreads();

	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if ((xIndex < height) && (yIndex <pitch_X))
	{
		unsigned int index_out = (yIndex*threadNum) * height + xIndex;
#pragma unroll
		for (int j = 0; j < threadNum; j++)
			if (yIndex*threadNum+j<width)
				dev_mat_dst[index_out+j*height] = block[threadIdx.x][threadIdx.y*threadNum+j];
	}
}

__global__ void copyData_kernel(float* dev_dst,float* dev_src,int width,int height){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int pitch_X = DIV_UP(width,ThreadHandleNum);
	if ((xIndex < pitch_X) && (yIndex < height))
	{
		unsigned int offset = yIndex*width + xIndex*ThreadHandleNum;
#pragma unroll
		for (int i = 0; i < ThreadHandleNum; i++){
			if ((xIndex*ThreadHandleNum + i) < width){
				dev_dst[offset + i] = dev_src[offset + i];
			}
		}

	}
}
