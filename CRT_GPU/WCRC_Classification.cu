#include "WCRC_Classification.cuh"
#include "Basic.h"
#ifndef DIV_UP
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#endif
//const +1
extern "C" const float CNST1;
//const 0
extern "C" const float CNST0;
//const -1
extern "C" const float CNSTm1;

void printMatrix(float* data, int rows, int cols){

	for (int i = 0; i < rows; i++){
		for (int j = 0; j <cols; j++)
		{
			printf("%f\t", data[i*cols + j]);
		}
		printf("\n");
		break;
	}
}




EXTERN_C void LBP_Feature_Extraction(dim3 grids, dim3 blocks, float* dev_Z, float* dev_LBP_data, int width, int height){
	LBP_EXTRACTION << < grids, blocks >> >(dev_Z, dev_LBP_data, width, height);
}



/*****************************************************************
*WCRC_CLASSIFICATION
*****************************************************************/

int getProperSize(int size)
{
	int i = 64;
	while (i < size)i <<= 1;
	return i;
}



void testSample_label(int* classlabel, float** host_dist_Y, int Testnum, int no_class, int Nt){
	float* temp;
	int position;
	float minValue;
	temp = (float*)malloc(sizeof(float)*no_class);

	for (int l = 0; l < Testnum; l++){
		for (int i = 0; i < no_class; i++){
			float maxValue = 0.0;
			for (int j = 0; j < Nt; j++){
				maxValue += host_dist_Y[i][l*Nt + j] * host_dist_Y[i][l*Nt + j];
			}
			temp[i] = sqrtf(maxValue);
		}

		minValue = temp[0];
		position = 0;
		for (int i = 0; i < no_class; i++){
			if (temp[i]<minValue)
			{
				minValue = temp[i];
				position = i;
			}
		}
		classlabel[l] = (position + 1);
	}
	free(temp);
}
static int flag = 0;
void testSample_label_2(int* classlabel, float* host_e, int Testnum, int no_class, int TestNumberOffset){
	
	int position;
	float minValue;
	for (int l = 0; l < Testnum; l++){
		minValue = host_e[l*no_class];
		position = 0;
		for (int i = 0; i < no_class; i++){
			if (host_e[l*no_class+i]<minValue)
			{
				minValue = host_e[l*no_class + i];
				position = i;
			}
		}
		classlabel[TestNumberOffset+l] = (position + 1);
		//printf("Testnum:%d---%d\n", (TestNumberOffset + l), classlabel[TestNumberOffset + l]);
	}
	/*if (!flag){
		writeFile(classlabel, Testnum, 1, "E:\\class_result_forward.mat", "class_result_forward");
		flag = 1;
		printf("flag is %d\n", flag);
	}*/
}

void swap(float *a, float *b){
	float c;
	c = *a;
	*a = *b;
	*b = c;
}

int inv(float *p, int n){
	int *is, *js, i, j, k, l;
	float temp, fmax;
	is = (int *)malloc(n*sizeof(int));
	js = (int *)malloc(n*sizeof(int));
	for (k = 0; k<n; k++)
	{
		fmax = 0.0;
		for (i = k; i<n; i++)
		for (j = k; j<n; j++)
		{
			temp = fabsf(p[i*n + j]);//find max
			if (temp>fmax)
			{
				fmax = temp;
				is[k] = i; js[k] = j;
			}
		}
		if ((fmax + 1.0) == 1.0)
		{
			free(is); free(js);
			printf("no inv");
			return(0);
		}
		if ((i = is[k]) != k)
		for (j = 0; j<n; j++)
			swap(&p[k*n + j], &p[i*n + j]);//swape pointer
		if ((j = js[k]) != k)
		for (i = 0; i<n; i++)
			swap(&p[i*n + k], &p[i*n + j]);  //swape pointer
		p[k*n + k] = 1.0 / p[k*n + k];
		for (j = 0; j<n; j++)
		if (j != k)
			p[k*n + j] *= p[k*n + k];
		for (i = 0; i<n; i++)
		if (i != k)
		for (j = 0; j<n; j++)
		if (j != k)
			p[i*n + j] = p[i*n + j] - p[i*n + k] * p[k*n + j];
		for (i = 0; i<n; i++)
		if (i != k)
			p[i*n + k] *= -p[k*n + k];
	}
	for (k = n - 1; k >= 0; k--)
	{
		if ((j = js[k]) != k)
		for (i = 0; i<n; i++)
			swap(&p[j*n + i], &p[k*n + i]);
		if ((i = is[k]) != k)
		for (j = 0; j<n; j++)
			swap(&p[j*n + i], &p[j*n + k]);
	}
	free(is);
	free(js);
	return 1;
}



void release(float **p, int m, int n)
{
	for (int i = 0; i < m; i++){
		free(p[i]);
		p[i] = NULL;
	}

	free(p);
	p = NULL;
}

void printArray2(float* data, int rows, int cols){
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			printf("%f\t", data[i*cols + j]);
		}
		printf("\n");
		break;
	}
}

//void release_matrix(matrix *p1)
//{
// free(p1->data);
// free(p1);
// p1 = NULL;
//}

//EXTERN_C void WCRC_Classification_stream(array_int* CTest,matrix* dataTest ,float* dev_dataTrain, float* dev_dataTest,int trainSample_rows){
// const float param[9] = { 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0 };
// 
// const int Testnum = dataTest->rows;//=8624
// const int Nt = dataTest->cols;//=220
// 

// int nstreams = 0;
// for (auto i : param)nstreams++;

// cublasHandle_t* handle = (cublasHandle_t*)malloc(nstreams*sizeof(cublasHandle_t));
// cudaStream_t* streams = (cudaStream_t*)malloc(nstreams*sizeof(cudaStream_t));

// for (int i = 0; i < nstreams; i++)
// {
//	 cublasCreate(&(handle[i]));
//	 cudaStreamCreate(&(streams[i]));
//	 cublasSetStream(handle[i], streams[i]);
// }

// float** dev_streamBuf1 = (float**)malloc(nstreams*sizeof(float*));

// //save norm for every data : Testnum*trainSample_rows
// float* dev_normsAll;

// //save X*X':trainSample_rows*trainSample_rows
// float* dev_Xsq;

// //YX':Testnum*trainSample_rows
// float* dev_XY;

// int mProper = getProperSize(trainSample_rows);

// for (int i = 0; i < nstreams; i++)
//	 checkCudaErrors(cudaMalloc((void**)&(dev_streamBuf1[i]), mProper*mProper*sizeof(float)));

// //the three variables are constant;
// checkCudaErrors(cudaMalloc((void**)&dev_normsAll, Testnum*trainSample_rows*sizeof(float)));
// checkCudaErrors(cudaMalloc((void**)&dev_XY, Testnum*trainSample_rows*sizeof(float)));
// checkCudaErrors(cudaMalloc((void**)&dev_Xsq, trainSample_rows*trainSample_rows*sizeof(float)));

// normsKernel << <Testnum, trainSample_rows, 0, streams[0] >> >(dev_dataTest, dev_dataTrain, dev_normsAll, Nt);

// //dev_XY=YX'=((x')'Y')' TestNum*Nt,(m*Nt)'=Test*m
// cublasDgemm(
//	 handle[1],
//	 CUBLAS_OP_T,
//	 CUBLAS_OP_N,
//	 trainSample_rows,Testnum,Nt,
//	 &CNST1,
//	 dev_dataTrain,Nt,
//	 dev_dataTest,Nt,
//	 &CNST0,
//	 dev_XY,trainSample_rows
//	 );

// //dev_Xsq=X*X'
// cublasDgemm(
//	 handle[2],
//	 CUBLAS_OP_T,
//	 CUBLAS_OP_N,
//	 trainSample_rows, trainSample_rows, Nt,
//	 &CNST1,
//	 dev_dataTrain, trainSample_rows,
//	 dev_dataTrain, trainSample_rows,
//	 &CNST0,
//	 dev_Xsq, trainSample_rows
//	 );

// for (int j = 0; j < Testnum; j++){
//	 for (int i = 0; i < nstreams; i++)
//	 {
//		 float lam = param[i];
//		 cudaMemcpy2DAsync(dev_streamBuf1[i], mProper*sizeof(float), dev_Xsq, trainSample_rows, trainSample_rows*sizeof(float), trainSample_rows, cudaMemcpyDeviceToDevice, streams[i]);
//		 geagKernel << <1, mProper, 0, streams[i] >> >(dev_streamBuf1[i], mProper, &(dev_normsAll[trainSample_rows*j]), trainSample_rows, lam);
//	 }
// }
//}

int getProp(int Testnum){
	int blockDimY;
	if (Testnum<1024)
	{
		blockDimY = Testnum;
	}
	else
	{
		blockDimY = 1024;
	}
	return blockDimY;
}

EXTERN_C void WCRC_Classification(array_int* CTest, array_int* CTrain, int no_class, int Testnum, int Nt, float* dev_dataTrain, float* dev_dataTest, int trainSample_rows){
	//const float param[9] = { 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0 };
	const float param[9] = { 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 1e1 };
	culaStatus status;
	int invStatus;
	

	//every param for classification

	//save norm for every data : Testnum*trainSample_rows
	float* dev_normsAll;

	//save X*X':trainSample_rows*trainSample_rows
	float* dev_Xsq;

	//YX':Testnum*trainSample_rows
	float* dev_XY;

	// float* ones;

	//the three variables are constant;
	checkCudaErrors(cudaMalloc((void**)&dev_normsAll, Testnum*trainSample_rows*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_XY, Testnum*trainSample_rows*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_Xsq, trainSample_rows*trainSample_rows*sizeof(float)));
	// checkCudaErrors(cudaMalloc((void**)&ones, trainSample_rows*trainSample_rows*sizeof(float)));

	normsKernel << <Testnum, trainSample_rows >> >(dev_dataTest, dev_dataTrain, dev_normsAll, Nt);

	//dev_XY=YX'=((X')'Y')'
	status = culaDeviceSgemm(
		'T',
		'N',
		trainSample_rows, Testnum, Nt,
		CNST1,
		dev_dataTrain, Nt,
		dev_dataTest, Nt,
		CNST0,
		dev_XY, trainSample_rows
		);
	checkStatus(status);

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


	//store weights
	float* dev_weights;
	checkCudaErrors(cudaMalloc((void**)&dev_weights, Testnum*trainSample_rows*sizeof(float)));


	float* dev_buffer;
	float* dev_ipiv;
	// float* dev_deta;
	checkCudaErrors(cudaMalloc((void**)&dev_buffer, trainSample_rows*trainSample_rows*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_ipiv, trainSample_rows*sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**)&dev_deta, trainSample_rows*sizeof(float)));


	for (int k = 0; k < 9; k++){
		//calculate weight
		float lam = param[k];
		for (int j = 0; j < Testnum; j++){
			geagKernel << <1, trainSample_rows >> >(dev_buffer, &(dev_normsAll[trainSample_rows*j]), trainSample_rows, lam);

			dim3 grids((trainSample_rows + (threads - 1)) / threads, (trainSample_rows + (threads - 1)) / threads);
			dim3 blocks(threads, threads);
			//xx'+G
			addSubKernel << <grids, blocks >> >(dev_Xsq, dev_buffer, trainSample_rows, trainSample_rows, '+');

			// LU factorization
			status = culaDeviceSgetrf(
				trainSample_rows, trainSample_rows,
				dev_buffer, trainSample_rows,
				(culaDeviceInt*)dev_ipiv
				);
			checkStatus(status);

			status = culaDeviceSgetri(
				trainSample_rows,
				dev_buffer,
				trainSample_rows,
				(culaDeviceInt*)dev_ipiv
				);
			checkStatus(status);

			//deta=yx'(xx'+G)inv=(((xx'+G)inv)'(yx')')'
			status = culaDeviceSgemm(
				'N',
				'N',
				trainSample_rows, 1, trainSample_rows,
				CNST1,
				dev_buffer, trainSample_rows,
				&dev_XY[j*trainSample_rows], trainSample_rows,
				CNST0,
				&dev_weights[j*trainSample_rows], trainSample_rows
				);
			checkStatus(status); 

			// sigment2_kernel << <1, trainSample_rows >> >(&dev_weights[j*trainSample_rows],dev_deta,trainSample_rows);
		}


		float** dev_dist_Y;
		//float** host_dist_Y;
		dev_dist_Y = (float**)malloc(no_class*sizeof(float*));
		//host_dist_Y = (float**)malloc(no_class*sizeof(float*)); 
		float** dev_weight_buf;
		dev_weight_buf = (float**)malloc(no_class*sizeof(float*));

		int trainOffset = 0;
		for (int i = 0; i < no_class; i++){
			//host_dist_Y[i] = (float*)malloc(sizeof(float)*Testnum*Nt);
			checkCudaErrors(cudaMalloc((void**)&(dev_dist_Y[i]), Testnum*Nt*sizeof(float)));
			checkCudaErrors(cudaMalloc((void**)&(dev_weight_buf[i]), Testnum*CTrain->data[i]*sizeof(float)));
			
			signment_kernel << <Testnum, CTrain->data[i] >> >(dev_weight_buf[i], dev_weights, i, trainOffset, trainSample_rows);
			trainOffset += CTrain->data[i];
		}

		/*int blockDimY = getProp(Testnum);
		int thread_handle_num =DIV_UP(Testnum, blockDimY);*/
		float* dev_perBlock;
		checkCudaErrors(cudaMalloc((void**)&dev_perBlock,sizeof(float)*Testnum*no_class));
		for (int i = 0; i < no_class; i++){
			//y=ax=(x'a')'
			int offset = 0;
			status = culaDeviceSgemm(
				'N',
				'N',
				Nt, Testnum, CTrain->data[i],
				CNST1,
				&dev_dataTrain[offset*Nt], Nt,
				dev_weight_buf[i], CTrain->data[i],
				CNST0,
				dev_dist_Y[i], Nt
				);
			checkStatus(status);
			offset += CTrain->data[i];
			/*dim3 grids(DIV_UP(Nt,threads),DIV_UP(Testnum ,threads));
			dim3 blocks(threads, threads);
			addSubKernel << <grids, blocks >> >(dev_dist_Y[i], dev_dataTest, Nt, Testnum, '-');
			checkCudaErrors(cudaMemcpy(host_dist_Y[i], dev_dist_Y[i], Testnum*Nt*sizeof(float), cudaMemcpyDeviceToHost));*/

			dim3 grids = dim3(1,Testnum);
			dim3 blocks = dim3(DIV_UP(Nt, 2) * 2,1);
			int shareMemory = blocks.x*sizeof(float);
			distNorms_kernel << <grids, blocks, shareMemory, 0 >> >(dev_dataTest,dev_dist_Y[i],dev_perBlock,Nt,Testnum,no_class,i);
		}
		float* host_e=(float*)malloc(sizeof(float)*Testnum*no_class);
		cudaMemcpy(host_e, dev_perBlock, sizeof(float)*Testnum*no_class, cudaMemcpyDeviceToHost);
		cudaFree(dev_perBlock);
		for (int i = 0; i < no_class; i++){
			cudaFree(dev_dist_Y[i]);
			cudaFree(dev_weight_buf[i]);
		}
		free(dev_dist_Y);
		free(dev_weight_buf);

		//calculate each class error
		int* classlabel;
		// int* dev_classlabel;
		classlabel = (int*)malloc(sizeof(int)*Testnum);
		// checkCudaErrors(cudaMalloc((void**)&dev_classlabel, sizeof(int)*Testnum));
		//testSample_label(classlabel, host_dist_Y, Testnum, no_class, Nt);

		testSample_label_2(classlabel, host_e, Testnum, no_class,0);
		free(host_e);
		//checkCudaErrors(cudaMemcpy(dev_classlabel, classlabel, sizeof(int)*Testnum,cudaMemcpyHostToDevice));
		//release(host_dist_Y, no_class, Testnum*Nt);
		
		//calculate accuracy
		int* c;
		//int* dev_c;
		c = (int*)malloc(sizeof(int)*no_class);
		//checkCudaErrors(cudaMalloc((void**)&dev_c, sizeof(int)*no_class));
		memset(c, 0, sizeof(int)*no_class);
		/*int offset = 0;
		for (int i = 0; i < CTest->num; i++){
		int blocks = (CTest->data[i] + (threads - 1)) / threads;
		histoKernel << < blocks,threads>> >(dev_classlabel+offset,CTest->data[i], &dev_c[i],i+1);
		offset += CTest->data[i];
		}
		checkCudaErrors(cudaMemcpy(c,dev_c,sizeof(int)*no_class,cudaMemcpyDeviceToHost));
		cudaFree(dev_c);
		*/
		//************CPU caclulate accuracy****************************
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

		//**************************************************************
		free(classlabel);
		//cudaFree(dev_classlabel);

		int sum = 0;
		for (int i = 0; i < no_class; i++){
			sum += c[i];
		}
		free(c);
		float accuracy = (float)sum / Testnum;
		printf("lama=%f,the accuracy is %f\n\n", param[k], accuracy);
	}
	
	cudaFree(dev_weights);

	cudaFree(dev_buffer);
	cudaFree(dev_ipiv);
	// cudaFree(dev_deta);

	cudaFree(dev_XY);
	cudaFree(dev_normsAll);
	cudaFree(dev_Xsq);
	// cudaFree(ones);

}

int getProperBlockSize(int Testnum){
	int blockSize = Testnum;
	if (Testnum > 65535){
		blockSize = 65535;
	}
	return blockSize;
}

EXTERN_C void WCRC_Classification_part(array_int* CTrain, int no_class, int Testnum, int Nt, float* dev_dataTrain, float* dev_dataTest, int trainSample_rows, float* dev_Xsq, float lam, int* classlabel,int TestNumberOffset){
	culaStatus status;
	int invStatus;
	//save norm for every data : Testnum*trainSample_rows
	float* dev_normsAll;
	
	//YX':Testnum*trainSample_rows
	float* dev_XY;
	// float* ones;
	//the three variables are constant;
	checkCudaErrors(cudaMalloc((void**)&dev_normsAll, Testnum*trainSample_rows*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_XY, Testnum*trainSample_rows*sizeof(float)));

	//normsKernel << <Testnum, trainSample_rows >> >(dev_dataTest, dev_dataTrain, dev_normsAll, Nt);
	//2 dimension
	dim3 normBlock(threads,threads);
	dim3 normGrid(DIV_UP(trainSample_rows,threads),DIV_UP(Testnum,threads));
	normsKernel2 << <normGrid,normBlock >> >(dev_dataTest, dev_dataTrain, dev_normsAll, Nt,Testnum,trainSample_rows);
	
	//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	/*
	*Test dev_normsAll
	*/
	/*float* host_normsAll = (float*)malloc(sizeof(float)*Testnum*trainSample_rows);
	checkCudaErrors(cudaMemcpy(host_normsAll, dev_normsAll, sizeof(float)*Testnum*trainSample_rows,cudaMemcpyDeviceToHost));
	writeFile(host_normsAll, Testnum, trainSample_rows, "E:\\host_normsAll_100000.mat", "host_normalAll_100000");*/
	//checkCudaErrors(cudaGetLastError());
	//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	//dev_XY=YX'=((X')'Y')'
	status = culaDeviceSgemm(
		'T',
		'N',
		trainSample_rows, Testnum, Nt,
		CNST1,
		dev_dataTrain, Nt,
		dev_dataTest, Nt,
		CNST0,
		dev_XY, trainSample_rows
		);
	checkStatus(status);

	/*float* host_XY = (float*)malloc(Testnum*trainSample_rows*sizeof(float));
	cudaMemcpy(host_XY, dev_XY, Testnum*trainSample_rows*sizeof(float),cudaMemcpyDeviceToHost);
	writeFile(host_XY, Testnum, trainSample_rows, "E:\\host_XY.mat", "host_XY");*/

	//store weights
	float* dev_weights;
	checkCudaErrors(cudaMalloc((void**)&dev_weights, Testnum*trainSample_rows*sizeof(float)));

	float* dev_buffer;
	float* dev_ipiv;
	// float* dev_deta;
	checkCudaErrors(cudaMalloc((void**)&dev_buffer, trainSample_rows*trainSample_rows*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_ipiv, trainSample_rows*sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**)&dev_deta, trainSample_rows*sizeof(float)));

		for (int j = 0; j < Testnum; j++){
			geagKernel << <1, trainSample_rows >> >(dev_buffer, &(dev_normsAll[trainSample_rows*j]), trainSample_rows, lam);

			dim3 grids(DIV_UP(trainSample_rows,threads),DIV_UP(trainSample_rows ,threads));
			dim3 blocks(threads, threads);
			//xx'+G
			addSubKernel << <grids, blocks >> >(dev_Xsq, dev_buffer, trainSample_rows, trainSample_rows, '+');

			// LU factorization
			status = culaDeviceSgetrf(
				trainSample_rows, trainSample_rows,
				dev_buffer, trainSample_rows,
				(culaDeviceInt*)dev_ipiv
				);
			checkStatus(status);

			status = culaDeviceSgetri(
				trainSample_rows,
				dev_buffer,
				trainSample_rows,
				(culaDeviceInt*)dev_ipiv
				);
			checkStatus(status);

			//deta=yx'(xx'+G)inv=(((xx'+G)inv)'(yx')')'
			status = culaDeviceSgemm(
				'N',
				'N',
				trainSample_rows, 1, trainSample_rows,
				CNST1,
				dev_buffer, trainSample_rows,
				&dev_XY[j*trainSample_rows], trainSample_rows,
				CNST0,
				&dev_weights[j*trainSample_rows], trainSample_rows
				);
			checkStatus(status);

			// sigment2_kernel << <1, trainSample_rows >> >(&dev_weights[j*trainSample_rows],dev_deta,trainSample_rows);
		}
		float** dev_dist_Y;
		//float** host_dist_Y;
		dev_dist_Y = (float**)malloc(no_class*sizeof(float*));
		//host_dist_Y = (float**)malloc(no_class*sizeof(float*)); 
		float** dev_weight_buf;
		dev_weight_buf = (float**)malloc(no_class*sizeof(float*));

		int trainOffset = 0;
		for (int i = 0; i < no_class; i++){
			//host_dist_Y[i] = (float*)malloc(sizeof(float)*Testnum*Nt);
			checkCudaErrors(cudaMalloc((void**)&(dev_dist_Y[i]), Testnum*Nt*sizeof(float)));
			checkCudaErrors(cudaMalloc((void**)&(dev_weight_buf[i]), Testnum*CTrain->data[i] * sizeof(float)));

			//signment_kernel << <Testnum, CTrain->data[i] >> >(dev_weight_buf[i], dev_weights, i, trainOffset, trainSample_rows);
			dim3 sigBlock(threads,threads);
			dim3 sigGrid(DIV_UP(CTrain->data[i],threads),DIV_UP(Testnum,threads));
			signment_kernel2 << <sigGrid, sigBlock >> >(dev_weight_buf[i], dev_weights, i, trainOffset, trainSample_rows, CTrain->data[i], Testnum);
			//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
			trainOffset += CTrain->data[i];

			//Test
			/*float* host_weight_buf = (float*)malloc(sizeof(float)*Testnum*CTrain->data[i]);
			checkCudaErrors(cudaMemcpy(host_weight_buf, dev_weight_buf[i], sizeof(float)*Testnum*CTrain->data[i],cudaMemcpyDeviceToHost));
			writeFile(host_weight_buf, Testnum, CTrain->data[i], "E:\\host_weight_buf_65535.mat", "host_weight_buf_65535");*/
		}

		/*int blockDimY = getProp(Testnum);
		int thread_handle_num =DIV_UP(Testnum, blockDimY);*/
		float* dev_perBlock;
		checkCudaErrors(cudaMalloc((void**)&dev_perBlock, sizeof(float)*Testnum*no_class));
		for (int i = 0; i < no_class; i++){
			//y=ax=(x'a')'
			int offset = 0;
			status = culaDeviceSgemm(
				'N',
				'N',
				Nt, Testnum, CTrain->data[i],
				CNST1,
				&dev_dataTrain[offset*Nt], Nt,
				dev_weight_buf[i], CTrain->data[i],
				CNST0,
				dev_dist_Y[i], Nt
				);
			checkStatus(status);
			offset += CTrain->data[i];
			/*dim3 grids(DIV_UP(Nt,threads),DIV_UP(Testnum ,threads));
			dim3 blocks(threads, threads);
			addSubKernel << <grids, blocks >> >(dev_dist_Y[i], dev_dataTest, Nt, Testnum, '-');
			checkCudaErrors(cudaMemcpy(host_dist_Y[i], dev_dist_Y[i], Testnum*Nt*sizeof(float), cudaMemcpyDeviceToHost));*/

			int blockSize = getProperBlockSize(Testnum);
			int blockNum = DIV_UP(Testnum,blockSize);
			dim3 blocks = dim3(DIV_UP(Nt, 2) * 2, 1);
			int shareMemory = blocks.x*sizeof(float);
			dim3 grids;
			int blockSizeOffset = 0;
			for (int k = 0; k < blockNum; k++){
				grids = dim3(1, blockSize);
				distNorms_kernel2 << <grids.y, blocks.x, shareMemory, 0 >> >(&dev_dataTest[blockSizeOffset*Nt], &dev_dist_Y[i][blockSizeOffset*Nt], &dev_perBlock[blockSizeOffset*no_class], Nt, no_class, i);
				//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
				blockSizeOffset += blockSize;
				blockSize = getProperBlockSize(Testnum-blockSizeOffset);
			}

			//distNorms_kernel << <grids, blocks, shareMemory, 0 >> >(dev_dataTest, dev_dist_Y[i], dev_perBlock, Nt, Testnum, no_class, i);
			//distNorms_kernel2 << <grids.y, blocks.x, shareMemory, 0 >> >(dev_dataTest, dev_dist_Y[i], dev_perBlock, Nt, Testnum, no_class, i);
			//checkCudaErrors(cudaGetLastError());

			//Test

		}
		float* host_e = (float*)malloc(sizeof(float)*Testnum*no_class);
		cudaMemcpy(host_e, dev_perBlock, sizeof(float)*Testnum*no_class, cudaMemcpyDeviceToHost);
		//writeFile(host_e, Testnum, no_class, "E:\\host_e_65535.mat", "host_e_65535");
		cudaFree(dev_perBlock);
		for (int i = 0; i < no_class; i++){
			cudaFree(dev_dist_Y[i]);
			cudaFree(dev_weight_buf[i]);
		}
		free(dev_dist_Y);
		free(dev_weight_buf);
		testSample_label_2(classlabel, host_e, Testnum, no_class,TestNumberOffset);
		free(host_e);
		
	

	cudaFree(dev_weights);

	cudaFree(dev_buffer);
	cudaFree(dev_ipiv);
	// cudaFree(dev_deta);

	cudaFree(dev_XY);
	cudaFree(dev_normsAll);
}