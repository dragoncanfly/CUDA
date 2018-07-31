#include"WCRC_CLASSIFICATION.h"
#include<time.h>
//#include <cula_lapack_device.h>
//#include<cula_blas_device.h>

void norms_Tiv(float* dataTest,float* dataTrain,float* norms,int Nt,int Testnum,int tranSample_rows){
	for (int i = 0; i < Testnum; i++){
		for (int j = 0; j < tranSample_rows; j++){
			float sum = 0.0;
			for (int k = 0; k < Nt; k++){
				float temp;
				temp = dataTest[i*Nt + k] - dataTrain[j*Nt + k];
				sum += temp*temp;
			}
			norms[i*tranSample_rows + j] = sum;
		}
	}
}

void compute_G(float* dev_buffer,float* norms,int transample_rows,float lam){
	for (int i = 0; i < transample_rows; i++){
		for (int j = 0; j < transample_rows; j++){
			if (i == j){
				dev_buffer[i*transample_rows + j] =lam*norms[i];
			}
			else{
				dev_buffer[i*transample_rows + j] = 0;
			}
			
		}
	}
}


void copy2Array(float* dst,float* src,int offset,int Testnum,int trainSample_rows){
	for (int i = 0; i < Testnum; i++){
		for (int j = 0; j < SAMPLE_NUM; j++){
			dst[i*SAMPLE_NUM + j] = src[offset*SAMPLE_NUM + i*trainSample_rows + j];
		}
	}
}



void min(int* classlabel,float* dist_Y,const int rows,const int cols){
	float minValue;
	int position;
	for (int i = 0; i < rows; i++){
		minValue = dist_Y[i*cols];
		position = 0;
		for (int j = 0; j < cols; j++){
			if (dist_Y[i*cols + j] < minValue){
				minValue = dist_Y[i*cols + j];
				position = j;
			}
		}
		classlabel[i] = position+1;
	}
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

void compute_accuracy(array_int* CTest,int* classlabel,int no_class,int Testnum,float lam,float* accuracyAll){
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
	int sum = 0;
	for (int i = 0; i < no_class; i++){
		sum += c[i];
	}
	free(c);
	float accuracy = (float)sum / Testnum;
	printf("lama=%lf,the accuracy is %lf\n\n", lam, accuracy);
	accuracyAll[0] = accuracy;
}

void printClassficationResult(float* dist_Y,int rowsth,int rows,int cols){
	float minValue;
	int position = 0;
	for (int i = 0; i < rows; i++){
		minValue = dist_Y[i*cols];
		for (int j = 0; j < cols; j++){
			if (dist_Y[i*cols + j] < minValue){
				minValue = dist_Y[i*cols + j];
				position = j;
			}
		}
		printf("%s%d%s%d\n", "the ", rowsth, " testSample is classificated ", (position + 1));
	}
}

void normal2row(float* data, int cols){
	float sum = 0.0;
	for (int i = 0; i < cols; i++){
		sum += data[i];
	}
	for (int j = 0; j < cols; j++){
		data[j] = data[j] / sum;
	}
}

void WCRC_Classification(array_int* CTest,array_int* CTrain,const int no_class, matrix* DataTrain, matrix* DataTest){
	clock_t start = clock();
	const float param[5] = { 1e-3, 1e-2, 1e-1, 1, 1e1};
	//save accuracy
	float* accuracyAll = (float*)malloc(sizeof(float)* 9);
	const int Testnum = DataTest->rows;
	const int Nt = DataTest->cols;
	//const int trainSample_rows = SAMPLE_NUM*no_class;
	const int trainSample_rows = DataTrain->rows;
	 float* h_dataTest = DataTest->data;
	 float* h_dataTrain = DataTrain->data;

	//save X*X':trainSample_rows*trainSample_rows
	float* h_Xsq;
	float* h_norms;
	float* h_XY;

	int status;
	//float* deta = (float*)malloc(sizeof(float)*trainSample_rows);
	float* weights = (float*)malloc(sizeof(float)*trainSample_rows);
	float* Y_hat = (float*)malloc(sizeof(float)*Nt);
	float* h_dist_Y = (float*)malloc(sizeof(float*)*Testnum*no_class);
	h_Xsq = (float*)malloc(sizeof(float)*trainSample_rows*trainSample_rows);
	h_norms = (float*)malloc(sizeof(float)*trainSample_rows);
	h_XY = (float*)malloc(sizeof(float)*trainSample_rows);
	
	
	//memset(h_normsAll, 0, sizeof(float)*Testnum*trainSample_rows);
	memset(h_Xsq, 0, sizeof(float)*trainSample_rows*trainSample_rows);
	

	float* X_tran = (float*)malloc(DataTrain->byteSize);
	matrix_trans(h_dataTrain,X_tran,Nt,trainSample_rows);

	matrix_multi(h_Xsq,h_dataTrain,X_tran,trainSample_rows,Nt,trainSample_rows);

	//writeFile(h_dataTrain, trainSample_rows, Nt, "F:\\X.mat", "X");
	//writeFile(X_tran, Nt, trainSample_rows, "F:\\X_tran.mat", "X_tran");
	//writeFile(h_XY, Testnum, trainSample_rows, "F:\\h_XY.mat", "h_XY");
	//writeFile(h_Xsq, trainSample_rows, trainSample_rows, "F:\\h_Xsq.mat", "h_Xsq");
	//writeFile(h_norms, 1, trainSample_rows,"F:\\h_normsAll.mat", "h_normsAll");

	int* classlabel;
	classlabel = (int*)malloc(sizeof(int)*Testnum);

	float* dev_buffer = (float*)malloc(sizeof(float)*trainSample_rows*trainSample_rows);
	float lam=param[3];
	for (int k = 0; k < 1; k++){
		 //lam = param[k];
		for (int j = 0; j < Testnum; j++){
			//printf("\n%s%d%s\n", "the ", j+1, " test sample  calculate ");
			norms_Tiv(&h_dataTest[j*Nt], h_dataTrain, h_norms, Nt,1, trainSample_rows);
			//writeFile(h_norms, 1, trainSample_rows, "F:\\h_norms.mat", "h_norms");
			compute_G(dev_buffer,h_norms,trainSample_rows,lam);
			matrix_add_sub(dev_buffer, h_Xsq, trainSample_rows, trainSample_rows,'+');
		//	writeFile(dev_buffer, trainSample_rows, trainSample_rows, "F:\\G_mat.mat", "G_mat");
			/*status=inv2(dev_buffer, trainSample_rows);
			if (!status){
				printf("inv failed");
				return;
			}*/
			inv3(dev_buffer, trainSample_rows); //使用LAPACK库函数优化逆矩阵计算
		//	writeFile(dev_buffer, trainSample_rows, trainSample_rows, "F:\\inv.mat", "inv");

			memset(h_XY, 0, sizeof(float)*trainSample_rows);
			matrix_multi(h_XY,&h_dataTest[j*Nt],X_tran,1,Nt,trainSample_rows);
		//	writeFile(h_XY, 1, trainSample_rows, "F:\\h_XY.mat", "h_XY");

			memset(weights, 0, sizeof(float)*trainSample_rows);
			matrix_multi(weights,h_XY,dev_buffer,1,trainSample_rows,trainSample_rows);
		//	writeFile(weights, 1, trainSample_rows, "F:\\h_weights.mat", "h_weight");
			
			int trainOffset = 0;
			for (int i = 0; i < no_class; i++){
				memset(Y_hat, 0, sizeof(float)*Nt);
				matrix_multi(Y_hat, &weights[trainOffset], &h_dataTrain[trainOffset*Nt], 1, CTrain->data[i], Nt);
				matrix_add_sub(Y_hat, &h_dataTest[j*Nt], Nt, 1, '-');
				//writeFile(Y_hat, 1, Nt, "F:\\h_Y_hat.mat", "h_Y_hat");
				h_dist_Y[j*no_class+i] = norms(Y_hat, 1, Nt, 2);
				trainOffset += CTrain->data[i];
			}
			//normal2row(&h_dist_Y[j*no_class], no_class);
			//writeFile(h_dist_Y, j+1, no_class, "F:\\h_dist_Y.mat", "h_dist_Y");
			//printClassficationResult(&h_dist_Y[j*no_class], j + 1, 1, no_class);
		}

		
		min(classlabel, h_dist_Y, Testnum, no_class);
		//compute_accuracy(CTest,classlabel,no_class,Testnum,lam,&accuracyAll[k]);
		////writeFile(accuracyAll, 1, k + 1, "E:\\Test\\accuracyAll.mat", "accuracyAll");
		//free(classlabel);
		break;
	}
	free(accuracyAll);
	free(h_norms);
	free(h_Xsq);
	free(h_XY);
	free(weights);
	free(Y_hat);
	free(h_dist_Y);
	free(X_tran);
	free(dev_buffer);
	clock_t end = clock();
	clock_t msecTotal = end - start;
	printf("time use=%ld ms\n", msecTotal);
	compute_accuracy(CTest, classlabel, no_class, Testnum, lam, &accuracyAll[0]);
	//writeFile(accuracyAll, 1, k + 1, "E:\\Test\\accuracyAll.mat", "accuracyAll");
	free(classlabel);
}
