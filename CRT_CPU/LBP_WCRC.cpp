#include <time.h>
#include "WCRC_CLASSIFICATION.h"
#include "BandSelect.h"
#include "LBP_feature_extract.h"

#define BANDNUM 10
#define Z_rows 145 
#define Z_cols 145
#define Z_dimension 200

#define TRAIN_SAMPLE_PERSENT 0.05

#define FIX_SAMPLE
#define TRAIN_SAMPLE_CLASS 100
//#define FIX_TEST
#define TEST_SAMPLE_CLASS 8000

#ifndef DIV_UP
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#endif

#define checkHostStatus(hostStatus) _checkHostStatus(hostStatus)

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

void printMatrix(matrix *mat){
	int rows = mat->rows;
	int cols = mat->cols;
	float* data = mat->data;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++)
		{
			printf("%lf\t", data[i*cols + j]);

		}
		printf("\n");
		break;
	}
}



void reshape(matrix* dataSet){
	float* temp = (float*)malloc(dataSet->byteSize);
	matrix_trans(dataSet->data, temp, dataSet->cols, dataSet->rows);
    int width = Z_rows*Z_cols;
	int height = Z_dimension;
	matrix_trans(temp, dataSet->data, width, height);
	dataSet->rows = Z_rows*Z_cols;
	dataSet->cols = Z_dimension;
	free(temp);
}
void reshape2(matrix* dataSet){
	float* temp = (float*)malloc(dataSet->byteSize);
	matrix_trans(dataSet->data, temp, dataSet->cols, dataSet->rows);
	int width = Z_rows*Z_cols;
	int height = dataSet->cols/Z_cols;
	matrix_trans(temp, dataSet->data, width, height);
	dataSet->rows = width;
	dataSet->cols = height;
	free(temp);
}



array_int* Matlab_find(matrix* map, int class_id){

	int width = map->cols;
	int height = map->rows;
	float* data = map->data;
	int count = 0;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			if (data[i*width + j] == class_id)
				count++;
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
			//getDimensionMatrix(data, test_sample, len, rowth, rowth, testOffset);
			trainOffset++;
		}
		else
		{
			if (i - trainRow + 1>testRow)
			{
				break;
			}
			getDimensionMatrix(data, test_sample, len, rowth, rowth, testOffset);
			testOffset++;
		}

	}
}

void copyMatrix(matrix* dst,matrix* src){
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

	char *file = NULL;
	//file = "../../Dataset/bloodcellsample.mat";
	//file = "../../../Dataset5/BloodCellSampleData.mat";
	file = "D:\\Nothing\\LPE-LBP\\indian\\indian9.mat";
	matrix *dataSet = (matrix*)malloc(sizeof(matrix));
	//checkHostStatus(readFile(file, "bloodcell", dataSet));
	checkHostStatus(readFile(file, "data", dataSet));
	//file = "../../Dataset/bloodcellGth.mat";
	//file = "../../../Dataset5/BloodGth.mat";
	file = "D:\\Nothing\\LPE-LBP\\indian\\indian9_map.mat";
	matrix* map = (matrix*)malloc(sizeof(matrix));
	//matrix* pos_rest = (matrix*)malloc(sizeof(matrix));
	//checkHostStatus(readFile(file, "map", map));
	checkHostStatus(readFile(file, "map", map));
	//checkHostStatus(readFile(file, "pos_rest", pos_rest));
	matrix* dataSetCopy = (matrix*)malloc(sizeof(matrix));
	dataSetCopy->data = (float*)malloc(dataSet->byteSize);
	copyMatrix(dataSetCopy, dataSet);

	/*****************************************************
	*reshape spactral feature,145x145x220=>21025x220
	*****************************************************/
	reshape(dataSet);
	float z_max = findMatrixMaxValue(dataSet);
	nomalizing(dataSet->data, dataSet->cols, dataSet->rows, z_max);
	//writeFile(dataSet, "E:\\Test\\dataSet_C.mat", "dataSet_C");

	/*******************************************************
	 * Band Select
	 *******************************************************/
	printf("bands select\n");
	clock_t start_bandSelect = clock();
	int* bsn=bandSelect(dataSet);
	clock_t end_bandSelect = clock();
	clock_t msecTotal_bandSelect = end_bandSelect - start_bandSelect;
	printf("time use=%ld ms\n", msecTotal_bandSelect / TIMES);
	for (int i = 0; i < BANDNUM;i++)
		printf("%d\n",bsn[i]);
	matrix_free(dataSet);
	/*****************************************************
	*extract spatial feature
	*****************************************************/
	const int radis = 1;
	const int nr = 8;
	Mapping* mapping = getMapping(nr);
	matrix* Feature_P = (matrix*)malloc(sizeof(matrix));

	printf("LBP extraction \n");
	clock_t start_LBP = clock();
	LBP_feature_global(Feature_P, dataSetCopy, bsn, mapping, radis, map, nr, 10);
	clock_t end_LBP = clock();
	clock_t msecTotal_LBP = end_LBP - start_LBP;
	printf("time use=%ld ms\n", msecTotal_LBP / TIMES);

	matrix_free(dataSetCopy);
	//writeFile(Feature_P, "E:\\Test\\Feature_P_noshape.mat", "Feature_P_C_noshape");
	reshape2(Feature_P);
	//writeFile(Feature_P, "E:\\Test\\Feature_P.mat", "Feature_P_C");
	

	/*****************************************************
	*choose the test samples and train samples
	*****************************************************/
	//matrix* Feature_P = dataSet;
	//class's number(class_label 1~8)
	matrix* DataTrain = (matrix*)malloc(sizeof(matrix));
	//DataTrain->data = (float*)malloc(sizeof(float));
	float no_class = findMatrixMaxValue(map);
	/*DataTrain->rows = SAMPLE_NUM * no_class;
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

	matrix_trans(map);
	//printMatrix(dataSet);
	/*const char* savePath = "F:\\test.mat";
	writeFile(dataSet, savePath, "test");*/

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
		int trainRow = DIV_UP(sampleNumSet->num*TRAIN_SAMPLE_PERSENT * 100, 100);
#endif

#ifdef FIX_TEST
		int testRow = TEST_SAMPLE_CLASS;
#else
		int testRow = sampleNumSet->num - trainRow;
#endif
		int size = testRow*Feature_P->cols;
		int trainSize = trainRow*Feature_P->cols;
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
		//int trainSampleNumOffset = (i - 1)*SAMPLE_NUM;
		//getDatabyPosition(DataTrain, DataTest, sampleNumSet, Feature_P, testSampleNum, trainSampleNumOffset);
		getDatabyPosition(DataTrain, DataTest, sampleNumSet, Feature_P, testSampleNum, trainSampleNum, trainRow,testRow);
		CTest->data[i - 1] = testRow;
		CTrain->data[i - 1] = trainRow;
		testSampleNum += testRow;
		trainSampleNum += trainRow;
		free(sampleNumSet->data);
		free(sampleNumSet);
	}

	DataTest->rows = testSampleNum;
	DataTest->cols = Feature_P->cols;
	DataTest->size = DataTest->rows*DataTest->cols;
	DataTest->byteSize = DataTest->size*sizeof(float);

	DataTrain->rows = trainSampleNum;
	DataTrain->cols = Feature_P->cols;
	DataTrain->size = DataTrain->rows*DataTrain->cols;
	DataTrain->byteSize = DataTrain->size*sizeof(float);
	
	matrix_free(Feature_P);
	matrix_free(map);

	//*****************************************Test***************************
	//printf("%d\n", CTest->num);
	/*printf("\n");
	printMatrix(DataTest);
	printf("\n");
	printMatrix(DataTrain);*/
	
	//************************************************************************
	float dataTrain_maxValue = findMatrixMaxValue(DataTrain);
	nomalizing(DataTrain->data, DataTrain->cols, DataTrain->rows, dataTrain_maxValue);
	nomalizing(DataTest->data, DataTest->cols, DataTest->rows, dataTrain_maxValue);
	/*writeFile(DataTrain, "E:\\dataTrain_C.mat", "dataTrain_C");
	writeFile(DataTest, "E:\\dataTest_C.mat", "dataTest_C");*/
	/*****************************************************
	*WCRC classification
	*****************************************************/
	printf("WCRC classification\n");
	//clock_t start=clock();
	for (int i = 0; i < TIMES; i++)
		WCRC_Classification(CTest,CTrain,no_class, DataTrain,DataTest);
	/*clock_t end = clock();
	clock_t msecTotal = end - start;
	printf("Finished.\n");
	printf("time use=%ld ms\n", msecTotal / TIMES);*/
	free(CTest->data);
	free(CTest);
	matrix_free(DataTrain);
	matrix_free(DataTest);
	getchar();
	return 0;
}
