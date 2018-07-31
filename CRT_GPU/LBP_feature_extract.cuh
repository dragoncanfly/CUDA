#include <stdio.h>
#include "MatOperation.h"
#ifndef BASIC_CUDA_HEAD
#define BASIC_CUDA_HEAD
#include <cuda_runtime.h>
#include<helper_cuda.h>
#include "device_launch_parameters.h"
#endif
#define Z_rows 145
#define Z_cols 145
#define thread 16

#define W 10
#define LBP_DIMENSION 59
#define BAND_NUM 10

//#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
texture<float,1> texRef0;
texture<float,1> texRef1;
texture<float,1> texRef2;
texture<float,1> texRef3;
texture<float,1> texRef4;
texture<float,1> texRef5;
texture<float,1> texRef6;
texture<float,1> texRef7;
texture<float,1> texRef8;
texture<float,1> texRef9;

__global__ void hopCount_kernel(int* dev_a,int* dev_hopCount){
	unsigned int tid = threadIdx.x;
	int k = 7;
#pragma unroll
	while (tid)
	{
		dev_a[tid*8+k] = tid & 1;
		tid >>= 1;
		--k;
	}
#pragma unroll
	for (int k = 0; k < 8; k++){
		int offset = k + 1 == 8 ? 0 : k + 1;
		if (dev_a[tid * 8 + k] != dev_a[tid*8+offset])
		{
			++dev_hopCount[tid];
		}
	}
}


__global__ void assign_center_kernel(float* dev_I, float* dev_allData, int offset,int width,int radius,int I_rows,int I_cols){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if ((xIndex < Z_cols) && (yIndex < Z_rows)){
		unsigned int allData_tid = yIndex*width + xIndex+offset;
		unsigned int I_tid = (yIndex+radius)*I_cols + radius + xIndex;
		dev_I[I_tid] = dev_allData[allData_tid];
	}
}

__global__ void assign_dege_cols_kernel(float* dev_I, float* dev_allData, int offset, int width, int radius, int I_rows, int I_cols){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int left_allData_tid = yIndex*width + xIndex + offset;
	//left side
	unsigned int left_I_tid = (yIndex+radius)*I_cols + (radius-1- xIndex);
	dev_I[left_I_tid] = dev_allData[left_allData_tid];
	//right side
	unsigned int right_allData_tid = yIndex*width+ offset+Z_cols-1-xIndex;
	unsigned int right_I_tid = (yIndex + radius)*I_cols + (radius + Z_cols + xIndex);
	dev_I[right_I_tid] = dev_allData[right_allData_tid];
}

__global__ void assign_dege_rows_kernel(float* dev_I, int radius, int I_rows, int I_cols){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int up_allocData_tid = (yIndex+radius)*I_cols + xIndex;
	//up side
	unsigned int up_I_tid =(radius-1-yIndex)*I_cols +xIndex;
	dev_I[up_I_tid] = dev_I[up_allocData_tid];
	//down side
	unsigned int down_allocData_tid = (Z_rows+radius-1-yIndex)*I_cols + xIndex;
	unsigned int down_I_tid = (Z_rows + radius+yIndex)*I_cols +  xIndex;
	dev_I[down_I_tid] = dev_I[down_allocData_tid];
}

__device__ void fetchFromTeture(float &value, int texRef_index, int coord){
	switch (texRef_index)
	{
	case 0:
		value = tex1Dfetch(texRef0, coord);
		break;
	case 1:
		value = tex1Dfetch(texRef1, coord);
		break;
	case 2:
		value = tex1Dfetch(texRef2, coord);
		break;
	case 3:
		value = tex1Dfetch(texRef3, coord);
		break;
	case 4:
		value = tex1Dfetch(texRef4, coord);
		break;
	case 5:
		value = tex1Dfetch(texRef5, coord);
		break;
	case 6:
		value = tex1Dfetch(texRef6, coord);
		break;
	case 7:
		value = tex1Dfetch(texRef7, coord);
		break;
	case 8:
		value = tex1Dfetch(texRef8, coord);
		break;
	case 9:
		value = tex1Dfetch(texRef9, coord);
		break;
	default:
		break;
	}
}

__device__ void no_interpolated(float* result, int texRef_index,int ry,int rx,int I_cols,int center_tid,int tid,int v){
	float center_pixel;
	float point_pixel;
	fetchFromTeture(center_pixel, texRef_index, center_tid);
	fetchFromTeture(point_pixel, texRef_index,(ry - 1)*I_cols + rx - 1);
	if (point_pixel >= center_pixel){
		result[tid] += v;
	}
	else{
		result[tid] += 0;
	}
}

__device__ float roundn(float x, float n){
	float p;
	if (n < 0){
		p = powf(10, -n);
		x = roundf(p*x) / p;
	}
	else if (n>0){
		p = powf(10, n);
		x = p*roundf(x / p);
	}
	else{
		x = roundf(x);
	}
	return x;
}

__device__ void interpolated(float* result, int texRef_index, int fy, int fx,int cy,int cx,int I_cols, int center_tid, int tid, int v,float w1,float w2,float w3,float w4){
	float w1_value, w2_value, w3_value, w4_value;
	fetchFromTeture(w1_value, texRef_index, (fy - 1)*I_cols + fx - 1);
	fetchFromTeture(w2_value, texRef_index, (fy - 1)*I_cols + cx - 1);
	fetchFromTeture(w3_value, texRef_index, (cy - 1)*I_cols + fx - 1);
	fetchFromTeture(w4_value, texRef_index, (cy - 1)*I_cols + cx - 1);
	 w1_value = w1*w1_value;
     w2_value = w2*w2_value;
	 w3_value = w3*w3_value;
	 w4_value = w4*w4_value;
	float center_pixel;
	fetchFromTeture(center_pixel, texRef_index, center_tid);
	float point_pixel = roundn((w1_value + w2_value + w3_value + w4_value), -4);
	if (point_pixel >= center_pixel){
		result[tid] += v;
	}
	else{
		result[tid] += 0;
	}
}


__global__ void LBP_feature_kernel(float* result,int texRef_index,int dx,int dy,float* spoints,int origy,int origx,int I_rows,int I_cols,int neighbors){
	//fetch center pixel
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int tid = yIndex*(dx+1) + xIndex;
	unsigned int center_tid = (origy+yIndex-1)*I_cols + xIndex+origx-1;
#pragma unroll
	for (int i = 0; i < neighbors; i++){
		float y = spoints[i * 2] + origy+yIndex;
		float x = spoints[i * 2 + 1] + origx+xIndex;
		int fy = floorf(y);
		int cy = ceilf(y);
		int ry = roundf(y);
		int fx = floorf(x);
		int cx = ceilf(x);
		int rx = roundf(x);
		//no_interpolated
		int v = 1 << i;
		if (fabsf(x - rx) < 1e-6 && fabsf(y - ry) < 1e-6){
			no_interpolated(result,texRef_index,ry,rx,I_cols,center_tid,tid,v);
		}
		else{
			float ty = y - fy;
			float tx = x - fx;
			//Calculate the interpolation weights.
			float w1 = roundn((1 - tx)*(1 - ty), -6);
			float w2 = roundn(tx*(1 - ty), -6);
			float w3 = roundn((1 - tx)*ty, -6);
			float w4 = roundn(1 - w1 - w2 - w3, -6);
			interpolated(result,texRef_index,fy,fx,cy,cx,I_cols,center_tid,tid,v,w1,w2,w3,w4);
		}
	}
	
}

__global__ void area_feature_kernel(float*histfea, float* dev_X, float*dev_map, int map_cols, int X_cols, int X_rows, uchar* dev_table,int n,int offset){
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	//extern __shared__ int hist[];
	if (xIndex<n){
		if (dev_map[yIndex*map_cols + xIndex] > 0){
			float area;
			int hist[LBP_DIMENSION];
			//init
#pragma unroll
			for (int k = 0; k < LBP_DIMENSION; k++){
				//hist[threadIdx.x*LBP_DIMENSION+k] = 0;
				hist[k] = 0;
			}
#pragma unroll
			for (int i = 0; i < 2 * W + 1; i++){
#pragma unroll
				for (int j = 0; j < 2 * W + 1; j++){
					area = dev_X[(yIndex + i)*X_cols + xIndex + j];
					area = dev_table[(int)area];
					//hist[threadIdx.x*LBP_DIMENSION + (int)area]++;
					hist[(int)area]++;
				}
			}
#pragma unroll
			for (int k = 0; k < LBP_DIMENSION; k++)
				//histfea[yIndex*n*LBP_DIMENSION + xIndex + (k*n)] = hist[k];
				//histfea[yIndex*n*LBP_DIMENSION*BAND_NUM + (offset*n*LBP_DIMENSION) + xIndex + (k*n)] = hist[threadIdx.x*LBP_DIMENSION+k];
				histfea[yIndex*n*LBP_DIMENSION*BAND_NUM + (offset*n*LBP_DIMENSION) + xIndex + (k*n)] = hist[k];
		}
	}
}