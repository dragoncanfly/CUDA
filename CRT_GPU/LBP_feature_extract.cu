#include "LBP_feature_extract.cuh"
#include "Basic_function.cuh"
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include<Math.h>

#ifndef MAX
#define MAX(a,b)((a)>(b)?(a):(b))
#endif
#ifndef MIN
#define MIN(a,b)((a)<(b)?(a):(b))
#endif
void lbp59table(uchar* table, int newMax,int* hopCount,int* dev_hopCount)
{
	int* dev_a;
	checkCudaErrors(cudaMalloc((void**)&dev_a, sizeof(int)* 8*256));
	checkCudaErrors(cudaMemset(dev_hopCount,0,sizeof(int)*256));
	checkCudaErrors(cudaMemset(dev_a, 0, sizeof(int)*8*256));
	hopCount_kernel << <1, 256 >> >(dev_a, dev_hopCount);
	checkCudaErrors(cudaMemcpy(hopCount, dev_hopCount, sizeof(int)* 256,cudaMemcpyDeviceToHost));
	for (int i = 0; i < 256; i++){
		printf("%d\n", hopCount[i]);
	}
	uchar index = 0;
	for (int i = 0; i<256; ++i)
	{
		if (hopCount[i] <= 2)
		{
			table[i] = index;
			index++;
		}
		else{
			table[i] = newMax - 1;
		}
		printf("%d\n", table[i]);
	}
	cudaFree(dev_a);
}

int getHopCount(uchar i)
{
	int a[8] = { 0 };
	int k = 7;
	int cnt = 0;
	while (i)
	{
		a[k] = i & 1;
		i >>= 1;
		--k;
	}
	for (int k = 0; k<8; ++k)
	{
		if (a[k] != a[k + 1 == 8 ? 0 : k + 1])
		{
			++cnt;
		}
	}
	return cnt;
}

void lbp59table(uchar* table, int newMax)
{
	memset(table, 0, 256);
	uchar index = 0;
	for (int i = 0; i<256; ++i)
	{
		if (getHopCount(i) <= 2)
		{
			table[i] = index;
			index++;
		}
		else{
			table[i] = newMax - 1;
		}
		//printf("%d\n", table[i]);
	}
}

EXTERN_C Mapping* getMapping(int samples){
	uchar* table = (uchar*)malloc(sizeof(uchar)* 256);
	int newMax = samples*(samples - 1) + 3;
	/*int* hopCount = (int*)malloc(sizeof(int)* 256);
	int* dev_hopCount;
	checkCudaErrors(cudaMalloc((void**)&dev_hopCount, sizeof(int)* 256));
	lbp59table(table, newMax,hopCount,dev_hopCount);*/
	lbp59table(table, newMax);
	Mapping* mapping = (Mapping*)malloc(sizeof(Mapping));
	mapping->table = table;
	mapping->samples = samples;
	mapping->num = newMax;
	/*cudaFree(dev_hopCount);
	free(hopCount);*/
	return mapping;
}

void padding_elements(cudaStream_t stream,float* dev_I, float* dev_allData, int offset,int width, int radius, int I_rows, int I_cols){
	dim3 grids;
	dim3 blocks;
	grids = dim3(DIV_UP(Z_cols,thread),DIV_UP(Z_rows,thread));
	blocks = dim3(thread, thread);
	assign_center_kernel << <grids, blocks, 0, stream >> >(dev_I,dev_allData,offset,width,radius,I_rows,I_cols);
	grids = dim3(1,Z_rows);
	blocks = dim3(radius, 1);
	assign_dege_cols_kernel << <grids,blocks, 0, stream >> >(dev_I,dev_allData,offset,width,radius,I_rows,I_cols);
	grids = dim3(1,radius);
	blocks = dim3(I_cols,1);
	assign_dege_rows_kernel << <grids, blocks, 0, stream >> >(dev_I, radius, I_rows, I_cols);
	//Test-----------
	/*float* h_I = (float*)malloc(sizeof(float)* 147 * 147);
	checkCudaErrors(cudaMemcpy(h_I, dev_I, sizeof(float)* 147 * 147, cudaMemcpyDeviceToHost));
	writeFile(h_I, 147, 147, "E:\\Test\\I_GPU_no_normalize.mat", "I_GPU_no_normalize");*/
}

void normalize(cudaStream_t stream,float* dev_I,const int I_rows,const int I_cols,float* dev_perBlock){
	dim3 grids;
	dim3 blocks;
	grids = dim3(I_cols,1);
	blocks = dim3(1, DIV_UP(I_rows,2)*2);
	int shareMemorySize = blocks.y*sizeof(float);
	findMaxValue_kernel<< <grids, blocks, shareMemorySize, stream >> >(dev_I,I_cols,I_rows,dev_perBlock);
	//Test
	/*float* maxVaule = (float*)malloc(sizeof(float)*I_cols);
	checkCudaErrors(cudaMemcpy(maxVaule, dev_perBlock, sizeof(float)*I_cols, cudaMemcpyDeviceToHost));
	writeFile(maxVaule, 1, 147, "E:\\Test\\maxValue_GPU.mat", "maxValue_GPU");*/
	grids = dim3(1,1);
	blocks = dim3(1,DIV_UP(I_cols,2)*2);
	shareMemorySize = blocks.y*sizeof(float);
	findMaxValue_kernel<< <grids, blocks, shareMemorySize, stream >> >(dev_perBlock, 1, I_cols, dev_perBlock);
	//Test MAX
	/*float* h_max=(float*)malloc(sizeof(float));
	checkCudaErrors(cudaMemcpy(h_max, &dev_perBlock[0], sizeof(float),cudaMemcpyDeviceToHost));*/
	//printf("%f\n", h_max[0]);
	grids = dim3(DIV_UP(I_cols,thread),DIV_UP(DIV_UP(I_rows,ThreadHandleNum),thread));
	blocks = dim3(thread,thread);
	normalization_kernel << <grids, blocks, 0, stream >> >(dev_I,dev_perBlock,I_cols,I_rows);
	//Test-----------
	/*float* h_I=(float*)malloc(sizeof(float)*147*147);
	checkCudaErrors(cudaMemcpy(h_I, dev_I, sizeof(float)* 147 * 147,cudaMemcpyDeviceToHost));
	writeFile(h_I, 147, 147, "E:\\Test\\I_GPU_normalize.mat", "I_GPU_normalize");*/
}

void calulateSpoints(float* spoints, int neighbors,int radius ,int &bsizey, int &bsizex, int &origy, int &origx){
	float a = (2 * M_PI) / neighbors;
	for (int i = 0; i < neighbors; i++){
		spoints[i * 2] = -radius*sinf(i*a);
		spoints[i * 2 + 1] = radius*cosf(i*a);
	}
	float miny = min(spoints, neighbors, 2, 0);
	float maxy = max(spoints, neighbors, 2, 0);
	float minx = min(spoints, neighbors, 2, 1);
	float maxx = max(spoints, neighbors, 2, 1);
	//block size,each LBP code is computed within a block of size bsizey*bsizex
	 bsizey = ceilf(MAX(maxy, 0)) - floorf(MIN(miny, 0)) + 1;
	 bsizex = ceilf(MAX(maxx, 0)) - floorf(MIN(minx, 0)) + 1;
	 origy = 1 - floorf(MIN(miny, 0));
	 origx = 1 - floorf(MIN(minx, 0));
}

void lbp_HSI(cudaStream_t stream,float* result,int texref_index,int dx,int dy,float* dev_spoints, const int neighbors,int origy,int origx,int I_rows,int I_cols ){
	dim3 grids(1,dy+1);
	dim3 blocks(dx + 1, 1);
	LBP_feature_kernel << <grids, blocks, 0, stream >> >(result,texref_index,dx,dy,dev_spoints,origy,origx,I_rows,I_cols,neighbors);
	//Test
	/*float* resultGPU = (float*)malloc(sizeof(float)*145*145);
	checkCudaErrors(cudaMemcpy(resultGPU, result, sizeof(float)* 145 * 145,cudaMemcpyDeviceToHost));
	writeFile(resultGPU,145,145,"E:\\Test\\result_GPU.mat","result_GPU");*/
}

int getPropThreadPerBlock(int width){
	int thread_num;
	if (width > 192){
		thread_num = 192;
	}
	else{
		thread_num = width;
	}
	return thread_num;
}

void hist_lbp_HSI(cudaStream_t stream,float* histfea,matrix lbp_img,Mapping* mapping,matrix* map,float* dev_map,float* dev_X,int X_rows,int X_cols,uchar* dev_table,int offset){
	int m = lbp_img.rows;
	int n = lbp_img.cols;
	float* dev_result = lbp_img.data;
	//Test
	/*float* resultGPU = (float*)malloc(sizeof(float)*145*145);
	checkCudaErrors(cudaMemcpy(resultGPU, dev_result, sizeof(float)* 145 * 145,cudaMemcpyDeviceToHost));
	writeFile(resultGPU,145,145,"E:\\Test\\result_hisfea_GPU.mat","result_hisfea_GPU");*/
	padding_elements(stream,dev_X,dev_result,0,n,W,X_rows,X_cols);
	/*float* h_X = (float*)malloc(sizeof(float)* X_cols * X_rows);
	checkCudaErrors(cudaMemcpy(h_X, dev_X, sizeof(float)* X_cols * X_rows, cudaMemcpyDeviceToHost));
	writeFile(h_X, X_rows,X_cols, "E:\\Test\\X_GPU.mat", "X_GPU"); */
	dim3 blocks(getPropThreadPerBlock(n), 1);
	dim3 grids(DIV_UP(n,blocks.x),m);
	//int shareMemorySize = blocks.x*mapping->num*sizeof(int);
	area_feature_kernel << <grids, blocks,0, stream >> >(histfea,dev_X, dev_map,map->cols,X_cols,X_rows,dev_table,n,offset);
}

void BindTexture(int index,float* dev_I,int I_cols,int I_rows){
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	switch (index)
	{
	case 0:
		cudaBindTexture(NULL, texRef0, dev_I, desc,sizeof(float)*I_cols*I_rows);
		break;
	case 1:
		cudaBindTexture(NULL, texRef1, dev_I, desc, sizeof(float)*I_cols*I_rows);
		break;
	case 2:
		cudaBindTexture(NULL, texRef2, dev_I, desc, sizeof(float)*I_cols*I_rows);
		break;
	case 3:
		cudaBindTexture(NULL, texRef3, dev_I, desc, sizeof(float)*I_cols*I_rows);
		break;
	case 4:
		cudaBindTexture(NULL, texRef4, dev_I, desc, sizeof(float)*I_cols*I_rows);
		break;
	case 5:
		cudaBindTexture(NULL, texRef5, dev_I, desc, sizeof(float)*I_cols*I_rows);
		break;
	case 6:
		cudaBindTexture(NULL, texRef6, dev_I, desc, sizeof(float)*I_cols*I_rows);
		break;
	case 7:
		cudaBindTexture(NULL, texRef7, dev_I, desc, sizeof(float)*I_cols*I_rows);
		break;
	case 8:
		cudaBindTexture(NULL, texRef8, dev_I, desc, sizeof(float)*I_cols*I_rows);
		break;
	case 9:
		cudaBindTexture(NULL, texRef9, dev_I, desc, sizeof(float)*I_cols*I_rows);
		break;
	default:
		break;
	}
}

void unbindTexture(int texRef_index){
	switch (texRef_index)
	{
	case 0:
		cudaUnbindTexture(texRef0);
		break;
	case 1:
		cudaUnbindTexture(texRef1);
		break;
	case 2:
		cudaUnbindTexture(texRef2);
		break;
	case 3:
		cudaUnbindTexture(texRef3);
		break;
	case 4:
		cudaUnbindTexture(texRef4);
		break;
	case 5:
		cudaUnbindTexture(texRef5);
		break;
	case 6:
		cudaUnbindTexture(texRef6);
		break;
	case 7:
		cudaUnbindTexture(texRef7);
		break;
	case 8:
		cudaUnbindTexture(texRef8);
		break;
	case 9:
		cudaUnbindTexture(texRef9);
		break;
	default:
		break;
	}
}

EXTERN_C void LBP_feature_global(matrix* Feature_P,float* dev_LBP_feature ,matrix* dataSet,int* bsn, Mapping* mapping, int radius, matrix* map, int num_point, int W0){
	float* allData = dataSet->data;
	float* LBP_feature = (float*)malloc(Feature_P->byteSize);
	Feature_P->data = LBP_feature;
	float* dev_allData;
	checkCudaErrors(cudaMalloc((void**)&dev_allData,dataSet->byteSize));
	checkCudaErrors(cudaMemcpy(dev_allData,allData,dataSet->byteSize,cudaMemcpyHostToDevice));
	int I_rows = Z_rows + 2 * radius;
	int I_cols = Z_cols + 2 * radius;
	float** dev_I=(float**)malloc(sizeof(float*)*BAND_NUM);
	float** dev_X = (float**)malloc(sizeof(float*)*BAND_NUM);
	float** dev_perBlock = (float**)malloc(sizeof(float*)*BAND_NUM);
	//calculate LBP weights
	int bsizey,bsizex,origy,origx;
	float* spoints = (float*)malloc(sizeof(float)*num_point * 2);
	calulateSpoints(spoints,num_point,radius,bsizey,bsizex,origy,origx);
	if (I_rows < bsizey || I_cols < bsizex){
		printf("Too small input image. Should be at least (2*radius+1) x (2*radius+1)");
		exit(1);
	}
	float* dev_spoints;
	checkCudaErrors(cudaMalloc((void**)&dev_spoints, sizeof(float)*num_point * 2));
	checkCudaErrors(cudaMemcpy(dev_spoints, spoints, sizeof(float)*num_point * 2,cudaMemcpyHostToDevice));
	int dx = I_cols - bsizex; 
	int dy = I_rows - bsizey;
	int X_rows = dy + 1 + 2 * W0;
	int X_cols = dx + 1 + 2 * W0;
	matrix lbp_img[BAND_NUM];
	//float** histfea = (float**)malloc(sizeof(float*)*BAND_NUM);
	cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*BAND_NUM);
	for (int i = 0; i < BAND_NUM; i++){
		cudaStreamCreate(&streams[i]);
		checkCudaErrors(cudaMalloc((void**)&dev_I[i], sizeof(float)*I_rows*I_cols));
		checkCudaErrors(cudaMalloc((void**)&dev_X[i], sizeof(float)*X_rows*X_cols));
		checkCudaErrors(cudaMalloc((void**)&dev_perBlock[i], sizeof(float)*I_cols));
		checkCudaErrors(cudaMalloc((void**)&lbp_img[i].data,sizeof(float)*(dx+1)*(dy+1)));
		lbp_img[i].rows = dy + 1;
		lbp_img[i].cols = dx + 1;
		checkCudaErrors(cudaMemset(lbp_img[i].data, 0, sizeof(float)*(dx + 1)*(dy + 1)));
		//checkCudaErrors(cudaMalloc((void**)&histfea[i], sizeof(float*)*(dx + 1)*mapping->num*(dy + 1)));
	}
	float* dev_map;
	checkCudaErrors(cudaMalloc((void**)&dev_map,map->byteSize));
	checkCudaErrors(cudaMemcpy(dev_map,map->data,map->byteSize,cudaMemcpyHostToDevice));
	uchar* dev_table;
	checkCudaErrors(cudaMalloc((void**)&dev_table,sizeof(uchar)*256));
	checkCudaErrors(cudaMemcpy(dev_table,mapping->table,sizeof(uchar)*256,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(dev_LBP_feature,0,Feature_P->byteSize));
	for (int i = 0; i < BAND_NUM; i++){
		padding_elements(streams[i],dev_I[i],dev_allData,bsn[i]*Z_cols,dataSet->cols,radius,I_rows,I_cols);
		normalize(streams[i], dev_I[i],I_rows, I_cols,dev_perBlock[i]);
		BindTexture(i, dev_I[i], I_cols, I_rows);
		lbp_HSI(streams[i],lbp_img[i].data,i,dx,dy,dev_spoints,num_point,origy,origx,I_rows,I_cols);
		//hist_lbp_HSI(streams[i],histfea[i],lbp_img[i],mapping,map,dev_map,dev_X[i],X_rows,X_cols,dev_table);
		hist_lbp_HSI(streams[i],dev_LBP_feature, lbp_img[i], mapping, map, dev_map, dev_X[i], X_rows, X_cols, dev_table,i);
	}
	//cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(LBP_feature,dev_LBP_feature,Feature_P->byteSize,cudaMemcpyDeviceToHost));
	//LBP特征从显存向内存拷贝时崩溃，原因未知, 之前I_rows和I_cols的初始化设置是const int I_rows = Z_rows + 2 * radius
	//导致I_rows = 152 ?
	for (int i = 0; i < BAND_NUM; i++){
		cudaStreamDestroy(streams[i]);
		unbindTexture(i);
		cudaFree(dev_I[i]);
		cudaFree(dev_X[i]);
		cudaFree(dev_perBlock[i]);
		cudaFree(lbp_img[i].data);
		//cudaFree(histfea[i]);
	}
	cudaFree(dev_allData);
	cudaFree(dev_map);
	cudaFree(dev_table);
	cudaFree(dev_spoints);
	free(spoints);
	free(dev_I);
	free(dev_perBlock);
	free(streams);
}