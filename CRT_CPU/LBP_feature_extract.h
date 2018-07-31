#ifndef LBP_FEATURE_EXTRACT_H
#define LBP_FEATURE_EXTRACT_H
#ifndef MATOPERATION_H
#include "MatOperation.h"
#endif
#endif
typedef unsigned char uchar;
typedef struct{
	uchar* table;
	int samples;
	int num;
}Mapping;

Mapping* getMapping(int samples);

void LBP_feature_global(matrix* Feature_P,matrix* dataSet, int* bsn, Mapping* mapping, int radius, matrix* map, int num_point, int W0);
