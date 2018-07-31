#include "BandSelect.h"

//one dimension
//void bandlp_one(float* e, float* X, float* y, int height){
//	float* X_sq = (float*)malloc(sizeof(float));
//	float* YX = (float*)malloc(sizeof(float));
//	memset(X_sq, 0, sizeof(float));
//	memset(YX, 0, sizeof(float));
//	matrix_multi(X_sq, X, X, 1, height, 1);
//	matrix_multi(YX, X, y, 1, height, 1);
//	float b = (1 / X_sq[0])*YX[0];
//	matrix_dot_malti(b, X, height, 1);
//	matrix_add_sub(X, y, 1, height, '-');
//	e[0] = norms(X, 1, height, 2);
//	free(X_sq);
//	free(YX);
//}

void bandlp_Multi(float* e, float* X,int X_cols ,float* y, int height){
	float* X_sq = (float*)malloc(sizeof(float)*X_cols*X_cols);
	memset(X_sq, 0, sizeof(float)*X_cols*X_cols);
	float* tmp = (float*)malloc(sizeof(float)*X_cols*height);
	memset(tmp, 0, sizeof(float)*X_cols*height);
	float* b = (float*)malloc(sizeof(float)*X_cols);
	memset(b, 0, sizeof(float)*X_cols);
	float* X_tran = (float*)malloc(sizeof(float)*X_cols*height);
	float* y_hat = (float*)malloc(sizeof(float)*height);
	memset(y_hat, 0, sizeof(float)*height);
	matrix_trans(X, X_tran, X_cols, height);
	matrix_multi(X_sq, X_tran,X,X_cols,height, X_cols);
	int status = inv2(X_sq, X_cols);
	if (!status){
		printf("inv failed");   
		return;
	}
	//inv3(X_sq, X_cols);  //LAPACKE library
	matrix_multi(tmp, X_sq,X_tran, X_cols,X_cols,height);
	free(X_tran);
	free(X_sq);

	matrix_multi(b, tmp, y, X_cols, height, 1);
	free(tmp);
	matrix_multi(y_hat, X, b, height, X_cols, 1);
	free(b);
	matrix_add_sub(y_hat, y, 1, height, '-');
	e[0] = norms(y_hat, 1, height, 2);
	free(y_hat);
}



void initArray(int* array,int value){
	for (int i = 0; i < BAND_NUM; i++){
		array[i] = value;
	}
}

int* findStart(matrix* dataSet){
	int rows= dataSet->rows;
	int bandnum = dataSet->cols;
	float* sz = dataSet->data;
	int* bstart = (int*)malloc(sizeof(int)*BAND_NUM);
	int* final = (int*)malloc(sizeof(int)*BAND_NUM);
	initArray(final, -1);
	float* X = (float*)malloc(sizeof(float)*rows);
	float* y = (float*)malloc(sizeof(float)*rows);
	int start;
	float* e = (float*)malloc(sizeof(float)*(bandnum));
	
	for (int k = 0; k< BAND_NUM; k++){
		memset(bstart, 0, sizeof(int)*BAND_NUM);
		bstart[0] = k;
		start = k;
		for (int i = 0; i < BAND_NUM; i++){
			getMatrixCol(sz,X,rows,bandnum,start);
			memset(e, 0, sizeof(float)*(bandnum));
			for (int j = 0; j < bandnum; j++){
				if (j != start){
					getMatrixCol(sz, y, rows, bandnum, j);
					bandlp_Multi(&e[j], X,1,y, rows);
				}
			}
			float MAX_VALUE=0;
			int index ;
			//find initial error position
			for (int j = 0; j < bandnum; j++){
				if (j != start){
					index = j;
					break;
				}
			}
			//find max(e)
			
			for (int j = 0; j < bandnum; j++){
				if (e[j]>MAX_VALUE){
					MAX_VALUE = e[j];
					index = j;
				}
			}
			start = index;
			bool finish_this_band_select = false;
			for (int j = 0; j < BAND_NUM; j++){
				if (bstart[j] == start){
					final[k] = start;
					finish_this_band_select = true;
					break;
				}
			}
			if (finish_this_band_select){
				break;
			}
			bstart[i + 1] = start;
		}
	}
	free(bstart);
	free(X);
	free(y);
	free(e);
	return final;
}

int* bs1(matrix* dataSet,int ind){
	const int height = dataSet->rows;
	const int bandnum = dataSet->cols;
	float* sz = dataSet->data;
	int* bands = (int*)malloc(sizeof(int)*BAND_NUM);
	bands[0] = ind;
	int selectedCount = 1;
	float* y = (float*)malloc(sizeof(float)*height);
	for (int i = 0; i < BAND_NUM - 1; i++){
		float* X = (float*)malloc(sizeof(float)*height*(i+1));
		getMatrixCol(sz, X, height, bandnum,bands,selectedCount);
		float e = 0;
		int position = 0;
		bool hasSelected;
		//select init position
		for (int j = 0; j < bandnum; j++){
			hasSelected = false;
			for (int k = 0; k < selectedCount; k++){
				if (j == bands[k]){
					hasSelected = true;
					break;
				}
			}
			if (hasSelected)continue;
			position = j;
			break;
		}
		
		//calculate errors
		for (int j = 0; j < bandnum; j++){
			hasSelected = false;
			float ne = 0;
			for (int k = 0; k < selectedCount; k++){
				if (j == bands[k]){
					hasSelected = true;
					break;
				}
			}
			if (hasSelected)continue;
			getMatrixCol(sz, y, height, bandnum, j);
			bandlp_Multi(&ne, X, selectedCount, y, height);
			if (ne>e){
				e = ne;
				position = j;
			}
		}

		bands[i + 1] = position;
		selectedCount++;
		free(X);
	}
	free(y);
	return bands;
}

int* bandSelect(matrix* dataSet){
	int* ind = findStart(dataSet);
	//make elements of ind +1
	for (int i = 0; i < BAND_NUM; i++){
		ind[i] += 1;
	}
	int* a = (int*)malloc(sizeof(int)*(dataSet->cols+1));
	memset(a, 0, sizeof(int)*(dataSet->cols+1));
	for (int i = 0; i < BAND_NUM; i++){
		a[ind[i]]++;
	}
	int MAX_COUNT = a[0];
	int position = 0;
	for (int i = 0; i < dataSet->cols+1; i++){
		if (a[i]>MAX_COUNT){
			a[i] = MAX_COUNT;
			position = i;
		}
	}
	free(a);
	int firstBand = position - 1;
	int* bsn;
	if (firstBand==-1)
	{
		bsn=bs1(dataSet, BAND_NUM-1);
	}
	else{
		bsn=bs1(dataSet, firstBand);
	}
	return bsn;
}