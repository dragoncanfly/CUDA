#include "LBP_WCRC.cuh"
#include "Basic.h"
#include "Basic_function.cuh"

#define BAND_NUM 10
//selected training samples are based on percentage of each class
//#define TRAIN_SAMPLE_PERSENT 0.01
//selected training samples are constant.
#define FIX_SAMPLE
#define TRAIN_SAMPLE_CLASS 100

//#define FIX_TEST
#define TEST_SAMPLE_CLASS 1000

#define COMPUTE_SAMPLES_TIME 70000


extern "C" const float CNST1 ;
//const 0
extern "C" const float CNST0;
//const -1
extern "C" const float CNSTm1;

inline void _checkHostStatus(hostError_t status){
	bool ret = true;
	switch (status)
	{
	case hostErrError:
		printf("%s\n", "get or save data failed!");
		ret = false;
		break;
	case hostErrNoError:
		break;
	case hostErrFileOpenFailed:
		printf("%s", "file open failed");
		ret = false;
		break;
	case hostErrMemAlreadyAlloc:
		printf("%s\n", "has alloc memory");
		ret = false;
		break;
	default:
		break;
	}
	if (!ret)
	{
		system("pause");
		exit(EXIT_FAILURE);
	}
}

//-----------------------------------------------------
void printMatrix(matrix *mat){
	int rows = mat->rows;
	int cols = mat->cols;
	float* data = mat->data;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++)
		{
			printf("%f\t", data[i*cols + j]);

		}
		printf("\n");
		break;
	}
}

void printArray(float* data, int rows, int cols){
	for (int i = 0; i < rows; i++){
		for (int j = 0; j <cols; j++)
		{
			printf("%f\t", data[i*cols + j]);
		}
		printf("\n");
		//break;
	}
}

void printArrayInt(int* data, int rows, int cols){
	for (int i = 0; i < rows; i++){
		for (int j = 0; j <cols; j++)
		{
			printf("%d\t", data[i*cols + j]);
		}
		printf("\n");
		break;
	}
}
//--------------------------------------------------------

void normalization(dim3 grids, dim3 blocks, float* dev_A, float* z_max, int width, int height){
	//printf("%f\n", z_max);
	
	normalization_kernel << <grids, blocks >> >(dev_A, z_max, width, height);
	//checkCudaErrors(cudaGetLastError());
}

void reshapeWithCULA(float* dev_A, float* dev_A_2, int width, int height){
	culaDeviceSgeTranspose(height,width,dev_A, width, dev_A_2,height);
	width = Z_rows*Z_cols;
	height = Z_dimension;
	culaDeviceSgeTranspose(height, width, dev_A_2, width, dev_A, height);
}

 void reshape(dim3 grids, dim3 blocks, float* dev_A, float* dev_A_2, int width, int height){
	trans_kernel << <grids, blocks >> >(dev_A, dev_A_2, width, height);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
	width = Z_rows*Z_cols;
	height = Z_dimension;
	dim3 grids2(DIV_UP(width,threads),DIV_UP(height,threads));
	dim3 blocks2(threads, threads);
	trans_kernel << <grids2, blocks2 >> >(dev_A_2, dev_A, width, height);
	checkCudaErrors(cudaGetLastError());
}

 void reshape2(matrix* Z, dim3 grids, dim3 blocks, float* dev_A, float* dev_A_2, int width, int height){
	 trans_kernel << <grids, blocks >> >(dev_A, dev_A_2, width, height);
	 checkCudaErrors(cudaGetLastError());
	 cudaDeviceSynchronize();
	 width = Z_rows*Z_cols;
	 height = Z->cols / Z_cols;
	 dim3 grids2(DIV_UP(width, threads), DIV_UP(height, threads));
	 dim3 blocks2(threads, threads);
	 trans_kernel << <grids2, blocks2 >> >(dev_A_2, dev_A, width, height);
	 checkCudaErrors(cudaGetLastError());
 }


 void reshape2WithCULA(matrix* Z, float* dev_A, float* dev_A_2, int width, int height){
	 culaDeviceSgeTranspose(height, width, dev_A, width, dev_A_2, height);
	 width = Z_rows*Z_cols;
	 height = Z->cols / Z_cols;
	 culaDeviceSgeTranspose(height, width, dev_A_2, width, dev_A, height);
 }

 void transpose(dim3 grids, dim3 blocks, float* dev_A, float* dev_A_2, int width, int height){
	trans_kernel << <grids, blocks >> >(dev_A, dev_A_2, width, height);
	//checkCudaErrors(cudaGetLastError());
}


 //free matrix
 void matrix_free(matrix *p1)
 {
	 free(p1->data);
	 free(p1);
	 p1 = NULL;
 }

 void reshapeZ_data(matrix *Z, float* dev_Z){
	 float* dev_Z_2;
	 //checkCudaErrors(cudaMalloc((void**)&dev_Z,Z->byteSize));
	 checkCudaErrors(cudaMalloc((void**)&dev_Z_2, Z->byteSize));
	 checkCudaErrors(cudaMemcpy(dev_Z, Z->data, Z->byteSize, cudaMemcpyHostToDevice));
	 dim3 grids(DIV_UP(Z->cols,threads), DIV_UP(Z->rows,threads));
	 dim3 blocks(threads, threads);
	 reshape(grids, blocks, dev_Z, dev_Z_2, Z->cols, Z->rows);
	// reshapeWithCULA(dev_Z, dev_Z_2, Z->cols, Z->rows);
	 checkCudaErrors(cudaMemcpy(Z->data, dev_Z, Z->byteSize, cudaMemcpyDeviceToHost));
	 Z->rows = Z_rows*Z_cols;
	 Z->cols = Z_dimension;
	 cudaFree(dev_Z_2);
 }
 void reshapeZ_data2(matrix *Z, float* dev_Z){

	 float* dev_Z_2;
	 //checkCudaErrors(cudaMalloc((void**)&dev_Z,Z->byteSize));
	 checkCudaErrors(cudaMalloc((void**)&dev_Z_2, Z->byteSize));
	 checkCudaErrors(cudaMemcpy(dev_Z, Z->data, Z->byteSize, cudaMemcpyHostToDevice));
	 dim3 grids(DIV_UP(Z->cols, threads), DIV_UP(Z->rows ,threads));
	 dim3 blocks(threads, threads);
	 reshape2(Z,grids, blocks, dev_Z, dev_Z_2, Z->cols, Z->rows);
	 //reshape2WithCULA(Z,dev_Z, dev_Z_2, Z->cols, Z->rows);
	 checkCudaErrors(cudaMemcpy(Z->data, dev_Z, Z->byteSize, cudaMemcpyDeviceToHost));
	 Z->rows = Z_rows*Z_cols;
	 Z->cols = Z->cols/Z_cols;
	 cudaFree(dev_Z_2);
 }

 void matrix_tans(matrix* Mat){
	 float* dev_Mat;
	 float* dev_Mat_trans;
	 int width = Mat->cols;
	 int height = Mat->rows;
	 checkCudaErrors(cudaMalloc((void**)&dev_Mat, Mat->byteSize));
	 checkCudaErrors(cudaMalloc((void**)&dev_Mat_trans, Mat->byteSize));
	 checkCudaErrors(cudaMemcpy(dev_Mat, Mat->data, Mat->byteSize, cudaMemcpyHostToDevice));
	 dim3 grids(DIV_UP(Mat->cols,threads),DIV_UP(Mat->rows,threads));
	 dim3 blocks(threads, threads);
	 transpose(grids, blocks, dev_Mat, dev_Mat_trans, Mat->cols, Mat->rows);
	 cudaMemcpy(Mat->data, dev_Mat_trans, Mat->byteSize, cudaMemcpyDeviceToHost);
	 Mat->rows = width;
	 Mat->cols = height;
	 cudaFree(dev_Mat_trans);
	 cudaFree(dev_Mat);
 }


 array_int* Matlab_find(matrix* map, int class_id){

	 int width = map->cols;
	 int height = map->rows;
	 float* data = map->data;
	 int count = 0;
	 for (int i = 0; i < height; i++){
		 for (int j = 0; j < width; j++){
			 if (data[i*width + j] == class_id)count++;
		 }
	 }
	 array_int* sampleNumSet = (array_int*)malloc(sizeof(array_int));
	 int* position_array = (int*)malloc(sizeof(int)*count);
	 count = 0;
	 for (int i = 0; i < height; i++){
		 for (int j = 0; j < width; j++){
			 if (data[i*width + j] == class_id){
				 position_array[count++] = i*width + j;
			 }
		 }
	 }
	 sampleNumSet->data = position_array;
	 sampleNumSet->num = count;
	 return sampleNumSet;
 }


 void randperm(array_int* sampleNumSet){
	 int n = sampleNumSet->num;
	 int* a = sampleNumSet->data;
	 int index, tmp, i;
	 //srand(time(NULL));
	 srand(1);
	 for (i = 0; i <n; i++)
	 {
		 index = rand() % (n - i) + i;
		 if (index != i)
		 {
			 tmp = a[i];
			 a[i] = a[index];
			 a[index] = tmp;
		 }
	 }
 }

 void getDatabyPosition(matrix* trainSet, matrix* testSet, array_int* sampleNumSet, matrix* src_data, int testOffset, int trainOffset,int trainRow,int testRow){
	 float* data = src_data->data;
	 float* train_sample = trainSet->data;
	 float* test_sample = testSet->data;
	 int* position_array = sampleNumSet->data;
	 int len = src_data->cols;
	 int num = sampleNumSet->num;
	 for (int i = 0; i < num; i++){
		 int rowth = position_array[i];
		 if (i<trainRow)
		 {
			 getDimensionMatrix(data, train_sample, len, rowth, rowth, trainOffset);
			// getDimensionMatrix(data, test_sample, len, rowth, rowth, testOffset);
			 trainOffset++;
		 }
		 else
		 {
			 if (i-trainRow+1>testRow)
			 {
				 break;
			 }
			 getDimensionMatrix(data, test_sample, len, rowth, rowth, testOffset);
			 testOffset++;
		 }

	 }
 }


 //initial gpu and disply info.
 void cudaInitial(int devID = 0)
 {
	 cudaSetDevice(devID);
	 cudaDeviceProp deviceProp;
	 cudaError_t cudaStatus = cudaGetDeviceProperties(&deviceProp, devID);
	 if (cudaStatus != cudaSuccess)
	 {
		 fprintf(stderr, "\n!!Error when initialize Gpu, %s.\n", cudaGetErrorString(cudaStatus));
		 return;
	 }
	 printf("Device %d: <%s>\n"
		 "  >CUDA Capability:%d.%d\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	 printf("  >Multiprocessor(s):%d\n", deviceProp.multiProcessorCount);
	 printf("  >DisplyMemory: %lu MB\n"
		 "  >ConstantMem: %d kB"
		 "\n\n",
		 deviceProp.totalGlobalMem / 1024 / 1024, deviceProp.totalConstMem / 1024);
 }


 void readDataFromMAT(matrix* dataSet, matrix* map){
	 char *file = NULL;
	/* file = "../../Dataset/bloodcellsample.mat";
	 checkHostStatus(readFile(file, "bloodcell", dataSet));*/
	 //file = "../../../Dataset5/BloodCellSampleData.mat";
	 file = "D:\\Nothing\\LPE-LBP\\indian\\indian9.mat";
	 checkHostStatus(readFile(file, "data", dataSet));
	 /*file = "../../Dataset/bloodcellGth.mat";
	 checkHostStatus(readFile(file, "map", map));*/
	 file = "D:\\Nothing\\LPE-LBP\\indian\\indian9_map.mat";
	 checkHostStatus(readFile(file, "map", map));
 }

 void reshapeDataSet(matrix* dataSet, dim3 &grids, dim3 &blocks, float* dev_Z){
	 
	 reshapeZ_data(dataSet, dev_Z);
	 float z_max = findMatrixMaxValue(dataSet);
	 float* dev_z_max;
	 checkCudaErrors(cudaMalloc((void**)&dev_z_max, sizeof(float)));
	 checkCudaErrors(cudaMemcpy(dev_z_max, &z_max, sizeof(float), cudaMemcpyHostToDevice));
	 //per thread process 4 data
	 grids = dim3(DIV_UP(dataSet->cols,threads),DIV_UP(DIV_UP(dataSet->rows,ThreadHandleNum),threads));
	 blocks = dim3(threads, threads);
	 normalization(grids, blocks, dev_Z, dev_z_max, dataSet->cols, dataSet->rows);
	 checkCudaErrors(cudaMemcpy(dataSet->data, dev_Z, dataSet->byteSize, cudaMemcpyDeviceToHost));
	 cudaFree(dev_z_max);
 }

 void reshapeFeature(matrix* dataSet, dim3 &grids, dim3 &blocks, float* dev_Z){
	 reshapeZ_data2(dataSet, dev_Z);
 }


 void chooseSample(matrix* map, const int no_class, matrix* dataSet, matrix* DataTest, matrix* DataTrain, array_int* CTest,array_int* CTrain){
	 matrix_tans(map);
	 //int offset = 0;
	 int testSampleNum = 0;
	 int trainSampleNum = 0;
	 int allocMemory = 0;
	 int trainAllocMemory = 0;
	 //printMatrix(dataSet);
	 for (int i = 1; i <= no_class; i++){
		 array_int* sampleNumSet = Matlab_find(map, i);
		 randperm(sampleNumSet);
#ifdef FIX_SAMPLE
		 int trainRow = TRAIN_SAMPLE_CLASS;
#else
		 int trainRow = DIV_UP(sampleNumSet->num*TRAIN_SAMPLE_PERSENT*100, 100);
#endif
		
#ifdef FIX_TEST
		 int testRow = TEST_SAMPLE_CLASS;
#else
		 int testRow=sampleNumSet->num-trainRow;
#endif
		 int size = testRow*dataSet->cols;
		 int trainSize = trainRow*dataSet->cols;
		 int bytesize = size*sizeof(float);
		 allocMemory += bytesize;
		 trainAllocMemory += trainSize*sizeof(float);
		 if (i == 1){
			 DataTest->data = (float*)malloc(allocMemory);
			 DataTrain->data = (float*)malloc(trainAllocMemory);
		 }
		 else{
			 DataTest->data = (float*)realloc(DataTest->data, allocMemory);
			 DataTrain->data = (float*)realloc(DataTrain->data, trainAllocMemory);
		 }
		 getDatabyPosition(DataTrain, DataTest, sampleNumSet, dataSet, testSampleNum,trainSampleNum,trainRow,testRow);
		 CTest->data[i - 1] = testRow;
		 CTrain->data[i - 1] =trainRow;
		 testSampleNum += testRow;
		 trainSampleNum += trainRow;
		 free(sampleNumSet->data);
		 free(sampleNumSet);
	 }
	 DataTest->rows = testSampleNum;
	 DataTest->cols = dataSet->cols;
	 DataTest->size = DataTest->rows*DataTest->cols;
	 DataTest->byteSize = DataTest->size*sizeof(float);

	 DataTrain->rows = trainSampleNum;
	 DataTrain->cols = dataSet->cols;
	 DataTrain->size = DataTrain->rows*DataTrain->cols;
	 DataTrain->byteSize = DataTrain->size*sizeof(float);
 }

 void nomalizingSample(matrix* DataTrain, matrix* DataTest, float* dev_dataTrain, float* dev_dataTest){
	 float normalize = findMatrixMaxValue(DataTrain);
	 float* dev_z_max;
	 checkCudaErrors(cudaMalloc((void**)&dev_z_max, sizeof(float)));
	 checkCudaErrors(cudaMemcpy(dev_z_max, &normalize, sizeof(float), cudaMemcpyHostToDevice));
	 checkCudaErrors(cudaMemcpy(dev_dataTrain, DataTrain->data, DataTrain->byteSize, cudaMemcpyHostToDevice));
	 checkCudaErrors(cudaMemcpy(dev_dataTest, DataTest->data, DataTest->byteSize, cudaMemcpyHostToDevice));
	 dim3 dataTrain_grids(DIV_UP(DataTrain->cols ,threads), DIV_UP(DIV_UP(DataTrain->rows ,ThreadHandleNum),threads));
	 dim3 blocks(threads, threads);
	 dim3 dataTest_grids(DIV_UP(DataTest->cols,threads), DIV_UP(DIV_UP(DataTest->rows,ThreadHandleNum),threads));
	 normalization(dataTrain_grids, blocks, dev_dataTrain, dev_z_max, DataTrain->cols, DataTrain->rows);
	 normalization(dataTest_grids, blocks, dev_dataTest, dev_z_max, DataTest->cols, DataTest->rows);
	 checkCudaErrors(cudaMemcpy(DataTrain->data, dev_dataTrain, DataTrain->byteSize, cudaMemcpyDeviceToHost));
	 checkCudaErrors(cudaMemcpy(DataTest->data, dev_dataTest, DataTest->byteSize, cudaMemcpyDeviceToHost));
	 cudaFree(dev_z_max);
 }


 void copyMatrix(matrix* dst, matrix* src){
	 int rows = src->rows;
	 int cols = src->cols;
	 /*float* srcdata = src->data;
	 float* dstdata = dst->data;
	 for (int i = 0; i < rows; i++){
		 for (int j = 0; j < cols; j++){
			 dstdata[i*cols + j] = srcdata[i*cols + j];
		 }
	 }*/
	 memcpy(dst->data, src->data, src->byteSize);
	 dst->rows = rows;
	 dst->cols = cols;
	 dst->size = src->size;
	 dst->byteSize = src->byteSize;
 }

 int main(){

#ifdef  EVERAGE_TIME
	 const int TIMES = 30;
#else
	 const int TIMES = 1;
#endif

	 matrix *dataSet = (matrix*)malloc(sizeof(matrix));
	 matrix* map = (matrix*)malloc(sizeof(matrix));
	 readDataFromMAT(dataSet, map);
	 matrix* dataSetCopy = (matrix*)malloc(sizeof(matrix));
	 dataSetCopy->data = (float*)malloc(dataSet->byteSize);
	 copyMatrix(dataSetCopy, dataSet);
	 /*****************************************************
	 *reshape spactral feature
	 *****************************************************/
	 cudaInitial();
	 culaStatus status;
	 status = culaInitialize();
	 checkStatus(status);

	 float* dev_Z;
	 checkCudaErrors(cudaMalloc((void**)&dev_Z, dataSet->byteSize));
	 dim3 grid;
	 dim3 blocks;
	 reshapeDataSet(dataSet, grid, blocks, dev_Z);
	 //writeFile(dataSet, "D:\\Nothing\\LPE-LBP\\DATA\\dataSet.mat", "dataSet_GPU");
	 /******************************************
	 * Band Select
	 ******************************************/
	 cudaEvent_t start_bandSelect, stop_bandSelect;
	 float msecTotal_bandSelect;
	 cudaEventCreate(&start_bandSelect);
	 cudaEventCreate(&stop_bandSelect);
	 printf("bands select using GPU...\n");
	 cudaEventRecord(start_bandSelect, NULL);
	 int* bsn = bandSelect(dataSet,dev_Z);
	 cudaEventRecord(stop_bandSelect, NULL);
	 cudaEventSynchronize(stop_bandSelect);
	 cudaEventElapsedTime(&msecTotal_bandSelect, start_bandSelect, stop_bandSelect);
	 printf("time use=%.3fms\n", msecTotal_bandSelect / TIMES); 
	 cudaEventDestroy(start_bandSelect);
	 cudaEventDestroy(stop_bandSelect);

	 cudaFree(dev_Z);
	 matrix_free(dataSet);
	for (int i = 0; i < BAND_NUM;i++)
		printf("%d\n",bsn[i]);
	 /*****************************************************
	 *extract spatial feature
	 *****************************************************/
	 //extract_spactial_feature(grid, blocks, dataSet, dev_Z);
     const int radis = 1;
	 const int nr = 8;
	 Mapping* mapping = getMapping(nr);
	 matrix* Feature_P = (matrix*)malloc(sizeof(matrix));
	 Feature_P->rows = Z_rows;
	 Feature_P->cols = Z_cols*(mapping->num*BAND_NUM);
	 Feature_P->size = Feature_P->rows*Feature_P->cols;
	 Feature_P->byteSize = Feature_P->size*sizeof(float);
	 float* dev_LBP_feature;
	 checkCudaErrors(cudaMalloc((void**)&dev_LBP_feature, Feature_P->byteSize));

	 cudaEvent_t start_LBP, stop_LBP;
	 float msecTotal_LBP;
	 cudaEventCreate(&start_LBP);
	 cudaEventCreate(&stop_LBP);
	 printf("LBP extraction using GPU...\n");
	 cudaEventRecord(start_LBP, NULL);
	 LBP_feature_global(Feature_P,dev_LBP_feature, dataSetCopy, bsn, mapping, radis, map, nr, 10);
	 cudaEventRecord(stop_LBP, NULL);
	 cudaEventSynchronize(stop_LBP);
	 cudaEventElapsedTime(&msecTotal_LBP, start_LBP, stop_LBP);
	 printf("time use=%.3fms\n", msecTotal_LBP / TIMES);
	 cudaEventDestroy(start_LBP);
	 cudaEventDestroy(stop_LBP);
	 //checkCudaErrors(cudaMemcpy(Feature_P->data, dev_LBP_feature, Feature_P->byteSize, cudaMemcpyDeviceToHost));

	 matrix_free(dataSetCopy);
	// writeFile(Feature_P, "E:\\Feature_P_GPU_noshape1D.mat", "Feature_P_GPU_noshape1D");
	 reshapeFeature(Feature_P, grid, blocks, dev_LBP_feature);
	 cudaFree(dev_LBP_feature);
	 
	 //writeFile(Feature_P, "E:\\Feature_P_GPU.mat", "Feature_P_GPU");
	 /*****************************************************
	 *choose the test samples and train samples
	 *****************************************************/
	 //class's number(class_label 1~8)
	 matrix* DataTrain = (matrix*)malloc(sizeof(matrix));
	 float no_class = findMatrixMaxValue(map);
	/* DataTrain->rows = SAMPLE_NUM * no_class;
	 DataTrain->cols = Feature_P->cols;
	 DataTrain->size = DataTrain->rows*DataTrain->cols;
	 DataTrain->byteSize = DataTrain->size*sizeof(float);
	 DataTrain->data = (float*)malloc(DataTrain->byteSize);*/

	 matrix* DataTest = (matrix*)malloc(sizeof(matrix));

	 array_int* CTest = (array_int*)malloc(sizeof(array_int));
	 CTest->num = no_class;
	 CTest->data = (int*)malloc(sizeof(int)*CTest->num);
	 array_int* CTrain = (array_int*)malloc(sizeof(array_int));
	 CTrain->num = no_class;
	 CTrain->data = (int*)malloc(sizeof(int)*CTrain->num);
	 chooseSample(map, no_class,Feature_P, DataTest, DataTrain, CTest,CTrain);
	 matrix_free(map);
	// matrix_free(dataSet);
	// writeFile(DataTrain, "E:\\dataTrain.mat", "dataTrain");
	 //normalizing sampels
	 float* dev_dataTrain;
	 checkCudaErrors(cudaMalloc((void**)&dev_dataTrain, DataTrain->byteSize));
	 float normalize = findMatrixMaxValue(DataTrain);
	 float* dev_z_max;
	 checkCudaErrors(cudaMalloc((void**)&dev_z_max, sizeof(float)));
	 checkCudaErrors(cudaMemcpy(dev_z_max, &normalize, sizeof(float), cudaMemcpyHostToDevice));
	 checkCudaErrors(cudaMemcpy(dev_dataTrain, DataTrain->data, DataTrain->byteSize, cudaMemcpyHostToDevice));
	 dim3 dataTrain_grids(DIV_UP(DataTrain->cols, threads), DIV_UP(DIV_UP(DataTrain->rows, ThreadHandleNum), threads));
	 //dim3 blocks(threads, threads);
	 blocks = dim3(threads, threads);
	 normalization(dataTrain_grids, blocks, dev_dataTrain, dev_z_max, DataTrain->cols, DataTrain->rows);
	 //computing A=XX';
	 const int Nt = DataTrain->cols;
	 float* dev_Xsq;
	 const int trainSample_rows = DataTrain->rows;
	 free(DataTrain->data);
	 free(DataTrain);
	 checkCudaErrors(cudaMalloc((void**)&dev_Xsq, trainSample_rows*trainSample_rows*sizeof(float)));
	 
	 cudaEvent_t X_start, X_stop;
	 float X_msecTotal;
	 cudaEventCreate(&X_start);
	 cudaEventCreate(&X_stop);
	 printf("Computing Xsq  using GPU...\n");
	 cudaEventRecord(X_start, NULL);
	 //dev_Xsq=X*X'=(X'X)'
	 status = culaDeviceSgemm(
		 'T',
		 'N',
		 trainSample_rows, trainSample_rows, Nt,
		 CNST1,
		 dev_dataTrain, Nt,
		 dev_dataTrain, Nt,
		 CNST0,
		 dev_Xsq, trainSample_rows
		 );
	 checkStatus(status);
	 cudaEventRecord(X_stop, NULL);
	 cudaEventSynchronize(X_stop);
	 cudaEventElapsedTime(&X_msecTotal, X_start,X_stop);
	 printf(" time use=%.3fms\n", X_msecTotal);
	 cudaEventDestroy(X_start);
	 cudaEventDestroy(X_stop);

	 /*
	  *block computing
	  */
	 
	 int blockNum = DIV_UP(DataTest->rows, COMPUTE_SAMPLES_TIME);
	 float* dev_dataTest;
	 int dataTestPartByteSize;
	 int TestNum;
	 int* classlabel = (int*)malloc(sizeof(int)*DataTest->rows);;
	 const float param[9] = { 1e-3,1e-2,1e-1, 1,1e1 };
	 float lamada = param[3];
	 for (int j = 0; j < 1; j++){
		 int TestNumberOffset=0;
		 memset(classlabel, 0, sizeof(int)*DataTest->rows);

		 cudaEvent_t start, stop;
		 float msecTotal;
		 cudaEventCreate(&start);
		 cudaEventCreate(&stop);
		 printf("CRT classification using GPU...\n");
		 cudaEventRecord(start, NULL);

		 for (int i = 0; i < blockNum; i++){
			 if (i == blockNum - 1){
				 TestNum = DataTest->rows - i* COMPUTE_SAMPLES_TIME;
				 dataTestPartByteSize = TestNum*Nt*sizeof(float);
				 checkCudaErrors(cudaMalloc((void**)&dev_dataTest, dataTestPartByteSize));
			 }
			 else{
				 TestNum = COMPUTE_SAMPLES_TIME;
				 dataTestPartByteSize = TestNum*Nt*sizeof(float);
				 checkCudaErrors(cudaMalloc((void**)&dev_dataTest, dataTestPartByteSize));
			 }
			 checkCudaErrors(cudaMemcpy(dev_dataTest, &DataTest->data[i*COMPUTE_SAMPLES_TIME*Nt], dataTestPartByteSize, cudaMemcpyHostToDevice));
			 dim3 dataTest_grids(DIV_UP(Nt, threads), DIV_UP(DIV_UP(TestNum, ThreadHandleNum), threads));
			 normalization(dataTest_grids, blocks, dev_dataTest, dev_z_max, Nt, TestNum);
			 WCRC_Classification_part(CTrain, no_class, TestNum, Nt, dev_dataTrain, dev_dataTest, trainSample_rows, dev_Xsq,lamada,classlabel,TestNumberOffset);
			 TestNumberOffset += TestNum;
			 cudaFree(dev_dataTest);
		 }
		 cudaEventRecord(stop, NULL);
		 cudaEventSynchronize(stop);
		 cudaEventElapsedTime(&msecTotal, start, stop);
		 printf(" time use=%.3fms\n", msecTotal);
		 cudaEventDestroy(start);
		 cudaEventDestroy(stop);

#ifdef FIX_TEST
		// writeFile(classlabel, TEST_SAMPLE_CLASS, 1, "E:\\classification_result_part2.mat", "classification_result_part2");
#else
		 //writeFile(classlabel, DataTest->rows, 1,"E:\\classification_result.mat", "classification_result");
#endif
		 int* c;
		 c = (int*)malloc(sizeof(int)*no_class);
		 memset(c, 0, sizeof(int)*no_class);
		 int offset = 0;
		 for (int i = 0; i < CTest->num; i++){
			 int classTotal = CTest->data[i];
			 for (int j = 0; j < classTotal; j++){
				 if (classlabel[offset + j] == (i + 1))
				 {
					 c[i] += 1;
				 }
			 }
			 offset += classTotal;
			 printf("c[%d]=%d\n", i, c[i]);
		 }
		 //free(classlabel);
		 int sum = 0;
		 for (int i = 0; i < no_class; i++){
			 sum += c[i];
		 }
		 free(c);
		 float accuracy = (float)sum / DataTest->rows;
		 printf("lama=%f,the accuracy is %f\n\n", lamada, accuracy);
		 break;
	 }
	 culaShutdown();
	 free(classlabel);
	 free(DataTest->data);
	 free(DataTest);
	 free(CTest->data);
	 free(CTest);
	 free(CTrain->data);
	 free(CTrain);
	 cudaFree(dev_dataTrain);
	 cudaFree(dev_Xsq);

	// float* dev_dataTrain;
	// float* dev_dataTest;
	// checkCudaErrors(cudaMalloc((void**)&dev_dataTrain, DataTrain->byteSize));
	// checkCudaErrors(cudaMalloc((void**)&dev_dataTest, DataTest->byteSize));
	// nomalizingSample(DataTrain, DataTest, dev_dataTrain, dev_dataTest);
	// int tranSample_rows = DataTrain->rows;
	///* writeFile(DataTest, "E:\\DataTest_GPU.mat", "DataTest_GPU");
	// writeFile(DataTrain, "E:\\DataTrain_GPU.mat", "DataTrain_GPU");*/
	// free(DataTrain->data);
	// free(DataTrain);
	// free(DataTest->data);
	// free(DataTest);
	// cudaFree(dev_dataTrain);
	// cudaFree(dev_Xsq);
	 cudaError_t cudaStatus = cudaDeviceReset();
	 if (cudaStatus != cudaSuccess)
	 {
		 fprintf(stderr, "cudaDeviceReset failed!");
		 return 1;
	 }
	 else
	 {
		 printf("cudaDeviceReset succeeded!\n\n");
	 }

	 getchar();
	 return 0;
 }
