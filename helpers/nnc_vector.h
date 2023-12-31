#ifndef CMATRIX_NNC_VECTOR_H
#define CMATRIX_NNC_VECTOR_H

#include "nnc_config.h"

void NNCVectorPrint(nnc_vector vector, nnc_uint len);

nnc_mtype NNCVectorMean(nnc_vector vector, nnc_uint len);

nnc_vector NNCVectorAdd(nnc_vector a, nnc_vector b, nnc_uint len);

nnc_mtype NNCVectorAccuracy(nnc_vector a, nnc_vector b, nnc_uint len);

void NNCVectorPrintTargetPrediction(nnc_vector a, nnc_vector b, nnc_uint len);

#endif //CMATRIX_NNC_VECTOR_H
