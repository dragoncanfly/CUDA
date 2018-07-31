#ifndef WCRC_CLASSIFICATION_H
#define WCRC_CLASSIFICATION_H
#ifndef MATOPERATION_H
#include "MatOperation.h"
#endif
#endif
#include <stdio.h>

#define SAMPLE_NUM 100

void WCRC_Classification(array_int* CTest,array_int* CTrain,const int no_class,matrix* DataTrain,matrix* DataTest);

