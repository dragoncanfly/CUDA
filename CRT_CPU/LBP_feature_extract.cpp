#include "LBP_feature_extract.h"
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include<Math.h>
#include<stdio.h>

#define Z_rows 145 
#define Z_cols 145
#define BAND_NUM 10

#define MAX(a,b)((a)>(b)?(a):(b))
#define MIN(a,b)((a)<(b)?(a):(b))

float roundn(float x, float n){
	float p;
	if (n < 0){
		p = powf(10,-n);
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



//uniform
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

void lbp59table(uchar* table,int newMax)
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
			table[i] = newMax-1;
		}
	}
}


Mapping* getMapping(int samples){
	uchar* table = (uchar*)malloc(sizeof(uchar)*256);
	int newMax = samples*(samples - 1) + 3;
	lbp59table(table, newMax);
	Mapping* mapping = (Mapping*)malloc(sizeof(Mapping));
	mapping->table = table;
	mapping->samples = samples;
	mapping->num = newMax;
	return mapping;
}

float* padding_elements(float* data,int offset,int W,int width,int m,int n){
	int X_rows = m + 2 * W;
	int X_cols = n + 2 * W;
	float* X = (float*)malloc(sizeof(float)*(X_rows*X_cols));
	//(W+1:m+W, W+1:n+W)
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			X[(i + W)*X_cols + (j + W)] = data[offset + i*width + j];
		}
	}
	//(W+1:m+W, 1:W)
	for (int i = 0; i < m; i++){
		for (int j = W; j>0; j--){
			X[(i + W)*X_cols + j-1] = data[offset + i*width + (W - j)];
		}
	}
	//(W+1:m+W, n+W+1:n+2*W)
	for (int i = 0; i < m; i++){
		for (int j = n; j>n -W; j--){
			X[(i + W)*X_cols + (W+n+n-j)] = data[offset + i*width + (j - 1)];
		}
	}
	//(1:W, :)
	for (int i = 0; i < W; i++){
		for (int j = 0; j < X_cols; j++){
			X[i*X_cols + j] = X[(2 * W - 1 - i)*X_cols + j];
		}
	}
	//(m+(W+1):m+2*W, :)
	for (int i = m + W; i < m + 2 * W; i++){
		for (int j = 0; j < X_cols; j++){
			X[i*X_cols + j] = X[(2*(m+W)-i-1)*X_cols+j];
		}
	}
	return  X;
}


void no_interpolated(float* D,float* image,int image_cols,int dx,int dy,int rx,int ry,int origx,int origy){
	for (int i = 0; i <= dy; i++){
		for (int j = 0; j <= dx; j++){
			if (image[((int)ry + i - 1)*image_cols + ((int)rx + j - 1)] >= image[(origy + i - 1)*image_cols + (origx + j - 1)]){
				D[i*(dx + 1) + j] = 1;
			}
			else{
				D[i*(dx + 1) + j] = 0;
			}
		}
	}
	//writeFile(D, dy + 1, dx + 1, "E:\\Test\\D.mat", "D_C");
}


void interpolated(float* image,int image_cols,float* D,float* N,int dx,int dy,int fx,int fy,int cx,int cy,float w1,float w2,float w3,float w4,int origx,int origy){
	float w1_value;
	float w2_value;
	float w3_value;
	float w4_value;
	for (int i = 0; i <= dy; i++){
		for (int j = 0; j <= dx; j++){
			w1_value = w1*image[(fy + i - 1)*image_cols + (fx + j - 1)];
			w2_value = w2*image[(fy + i - 1)*image_cols + (cx + j - 1)];
			w3_value = w3*image[(cy + i - 1)*image_cols + (fx + j - 1)];
			w4_value = w4*image[(cy + i - 1)*image_cols + (cx + j - 1)];
			N[i*(dx + 1) + j] =roundn((w1_value + w2_value + w3_value + w4_value),-4);
		}
	}

	for (int i = 0; i <= dy; i++){
		for (int j = 0; j <= dx; j++){
			if (N[i*(dx + 1) + j] >= image[(origy + i - 1)*image_cols + (origx + j - 1)]){
				D[i*(dx + 1) + j] = 1;
			}
			else{
				D[i*(dx + 1) + j] = 0;
			}
		}
	}
	//writeFile(D, dy + 1, dx + 1, "E:\\Test\\D.mat", "D_C");

}

matrix* lbp_HSI(float* image,const int image_rows,const int image_cols,const int radius,const int neighbors,Mapping* mapping){
	//Angle step
	float a = (2 * M_PI)/neighbors;
	float* spoints = (float*)malloc(sizeof(float)*neighbors * 2);
	for (int i = 0; i < neighbors; i++){
		spoints[i * 2] = -radius*sinf(i*a);
		spoints[i * 2 + 1] = radius*cosf(i*a);
	}
	int xsize = image_cols;
	int ysize = image_rows;
	float miny = min(spoints, neighbors, 2, 0);
	float maxy = max(spoints, neighbors, 2, 0);
	float minx = min(spoints, neighbors, 2, 1);
	float maxx = max(spoints, neighbors, 2, 1);
	//block size,each LBP code is computed within a block of size bsizey*bsizex
	int bsizey = ceilf(MAX(maxy, 0)) - floorf(MIN(miny,0))+1;
	int bsizex = ceilf(MAX(maxx, 0)) - floorf(MIN(minx, 0)) + 1;
	int origy = 1 - floorf(MIN(miny, 0));
	int origx = 1 - floorf(MIN(minx, 0));
	if (ysize < bsizey || xsize < bsizex){
		printf("Too small input image. Should be at least (2*radius+1) x (2*radius+1)");
		exit(1);
	}
	int dx = xsize - bsizex;
	int dy = ysize - bsizey;
	matrix* lbp_img = (matrix*)malloc(sizeof(matrix));
	float* result = (float*)malloc(sizeof(float)*(dy + 1)*(dx + 1));
	lbp_img->data = result;
	lbp_img->rows = dy + 1;
	lbp_img->cols = dx + 1;
	memset(result, 0, sizeof(float)*(dy + 1)*(dx + 1));
	float* D = (float*)malloc(sizeof(float)*(dy + 1)*(dx + 1));
	float* N = (float*)malloc(sizeof(float)*(dy + 1)*(dx + 1));
	for (int i = 0; i < neighbors; i++){
		float y = spoints[i * 2] + origy;
		float x = spoints[i * 2 + 1] + origx;
		int fy = floorf(y);
		int cy = ceilf(y);
		float ry = round(y);
		int fx = floorf(x);
		int cx = ceilf(x);
		float rx = roundf(x);
		// Check if interpolation is needed.
		memset(D, 0, sizeof(float)*(dy + 1)*(dx + 1));
		if (fabsf(x - rx) < 1e-6 && fabsf(y - ry) < 1e-6){
			no_interpolated(D,image,image_cols,dx,dy,rx,ry,origx,origy);
		}
		else{
			float ty = y - fy;
			float tx = x - fx;
			//Calculate the interpolation weights.
			float w1 = roundn((1-tx)*(1-ty),-6);
			float w2 = roundn(tx*(1-ty),-6);
			float w3 = roundn((1 - tx)*ty, -6);
			float w4 = roundn(1-w1-w2-w3,-6);
			interpolated(image,image_cols,D,N,dx,dy,fx,fy,cx,cy,w1,w2,w3,w4,origx,origy);

		}
		float v = 1 <<i;
		matrix_dot_multi(v, D, (dy + 1), (dx + 1));
		matrix_add_sub(result, D, (dx + 1), (dy + 1),'+');
		//writeFile(result, dy + 1, dx + 1, "E:\\Test\\result.mat", "result_C");
	}
	//writeFile(result, dy + 1, dx + 1, "E:\\Test\\result.mat", "result_C");
	free(spoints);
	free(D);
	free(N);
	return lbp_img;
} 

void assigment_area(float* X,float* result,int pp,int qq,int W,int width,int height,int X_cols){
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			result[i*width + j] = X[(pp-W+i)*X_cols+(qq-W+j)];
		}
	}
}

 void hist(float* histCount,float* result,int width,int height,int scale){
	memset(histCount, 0, sizeof(float)*scale);
	for (int i = 0; i < width*height; i++){
		histCount[(int)result[i]]++;
	}
}

float* hist_lbp_HSI(matrix* lbp_img,Mapping* mapping,int W,matrix* map){
	int m = lbp_img->rows;
	int n = lbp_img->cols;
	float* result = lbp_img->data;
	//writeFile(result, m, n, "E:\\Test\\result_hisfea.mat", "result_hisfea");
	int X_rows = m + 2 * W;
	int X_cols = n + 2 * W;
	float* X=padding_elements(result,0,W,n,m,n);
	//writeFile(X, X_rows, X_cols, "E:\\Test\\X_hisfea.mat", "X_hisfea");
	int area_width = 2 * W + 1;
	int area_height = 2 * W + 1;
	int map_rows = map->rows;
	int map_cols = map->cols;
	float* area = (float*)malloc(sizeof(float)*area_height*area_width);
	float* histfea = (float*)malloc(sizeof(float)*m*(n*mapping->num));
	float* histCount = (float*)malloc(sizeof(float)*mapping->num);
	for (int pp = W; pp <= m + W - 1; pp++){
		for (int qq = W; qq <= n + W - 1; qq++){
			if (map->data[(pp - W)*map_cols + (qq - W)]>0){
				assigment_area(X, area, pp, qq, W, area_width, area_height,X_cols);
				//writeFile(area, area_height, area_width, "E:\\Test\\area.mat", "area_C");
				for (int i = 0; i < area_height; i++){
					for (int j = 0; j < area_width; j++){
						area[i*area_width + j] = mapping->table[(int)area[i*area_width + j]];
					}
				}
				//writeFile(area, area_height, area_width, "E:\\Test\\area.mat", "area_C");
				hist(histCount, area, area_width, area_height, mapping->num);
				//3 dimension data show as matrix
				for (int k = 0; k < mapping->num; k++){
					histfea[(pp - W)*(n*mapping->num) + (qq - W) + (k*n)] = histCount[k];
				}
			}
		}
	}
	free(X);
	free(area);
	free(histCount);
	return histfea;
}

void cat(float* LBP_feature,float* histfea,int dimension,int offset){
	for (int i = 0; i < Z_rows; i++){
		for (int j = 0; j < Z_cols*dimension; j++){
			LBP_feature[i*(Z_cols*dimension*BAND_NUM) + j + (offset*Z_cols*dimension)] = histfea[i*(Z_cols*dimension) + j];
		}
	}
}

void LBP_feature_global(matrix* Feature_P,matrix* dataSet, int* bsn,Mapping* mapping, int radius, matrix* map, int num_point, int W0){
	float* allData = dataSet->data;
	float* LBP_feature = (float*)malloc(sizeof(float)*Z_rows*Z_cols*(mapping->num*BAND_NUM));
	Feature_P->data = LBP_feature;
	Feature_P->rows = Z_rows;
	Feature_P->cols = Z_cols*(mapping->num*BAND_NUM);
	Feature_P->size = Feature_P->rows*Feature_P->cols;
	Feature_P->byteSize = Feature_P->size*sizeof(float);
	for (int i = 0; i < BAND_NUM; i++){
		int bandIndex = bsn[i];
		int offset = bandIndex*Z_cols;
		float* I = padding_elements(allData,offset, radius,dataSet->cols,Z_rows,Z_cols);
		const int I_rows = Z_rows + 2 * radius;
		const int I_cols = Z_cols + 2 * radius;
		//writeFile(I, I_rows, I_cols, "E:\\Test\\I_C_no_nomalize.mat", "I_C_no_nomalize");
		float ImaxValue = findMatrixMaxValue(I, I_rows, I_cols);
		nomalizing(I,I_cols,I_rows,ImaxValue);
		//writeFile(I, I_rows, I_cols, "E:\\Test\\I.mat", "I_C");
		matrix* lbp_img=lbp_HSI(I, I_rows,I_cols,radius, num_point, mapping);
		//writeFile(lbp_img,"E:\\Test\\lbp_image.mat","lbp_image_C");
		float* histfea=hist_lbp_HSI(lbp_img, mapping, W0, map);
		//writeFile(histfea,lbp_img->rows,(lbp_img->cols*mapping->num),"E:\\Test\\histfea.mat","histfea_C");
		cat(LBP_feature, histfea, mapping->num, i);
		matrix_free(lbp_img);
		free(histfea);
	}
}

