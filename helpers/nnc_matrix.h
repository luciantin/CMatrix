#ifndef CMATRIX_NNC_MATRIX_H
#define CMATRIX_NNC_MATRIX_H

#include <stdbool.h>
#include "nnc_config.h"

#ifndef NNC_MATRIX_MULTIPLY_SQUARE_ALGO
    #define NNC_MATRIX_MULTIPLY_SQUARE_ALGO NNC_MATRIX_MULTIPLY_SQUARE_ALGO_ITERATIVE
#endif

typedef struct NNCMatrixType
{
    nnc_matrix matrix;
    nnc_uint    x;
    nnc_uint    y;
}
NNCMatrixType;

#define NNCIMatrixType NNCMatrixType*


NNCIMatrixType NNCMatrixAlloc(nnc_uint x, nnc_uint y);
NNCIMatrixType NNCMatrixAllocBaseValue(nnc_uint x, nnc_uint y, nnc_mtype base_value);
NNCIMatrixType NNCMatrixAllocRandom(nnc_uint x, nnc_uint y);
NNCIMatrixType NNCMatrixAllocBernoulli(nnc_uint x, nnc_uint y, nnc_mtype p, nnc_mtype divider);
NNCIMatrixType NNCMatrixAllocDiagonal(nnc_uint x, nnc_uint y, nnc_mtype base_value);

NNCIMatrixType NNCMatrixAllocLine(nnc_uint x, nnc_uint y);
NNCIMatrixType NNCMatrixAllocSum(nnc_uint x, nnc_uint y);

void NNCMatrixDeAlloc(NNCIMatrixType matrix);
void NNCMatrixPrint(NNCIMatrixType matrix);

NNCIMatrixType NNCMatrixProduct(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b);
NNCIMatrixType NNCMatrixProductNumber(NNCIMatrixType matrix, nnc_mtype num);
NNCIMatrixType NNCMatrixQuotient(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b);
NNCIMatrixType NNCMatrixQuotientNumber(NNCIMatrixType matrix, nnc_mtype num);
NNCIMatrixType NNCMatrixQuotientNumberReverse(nnc_mtype num, NNCIMatrixType matrix);

nnc_mtype NNCMatrixSumAll(NNCIMatrixType matrix);
nnc_mtype NNCMatrixSumAllAbs(NNCIMatrixType matrix);
NNCIMatrixType NNCMatrixSum(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b);
NNCIMatrixType NNCMatrixSumSingle(NNCIMatrixType matrix, bool axis);
NNCIMatrixType NNCMatrixSumNumber(NNCIMatrixType matrix, nnc_mtype number);
NNCIMatrixType NNCMatrixSub(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b);

NNCIMatrixType NNCMatrixSqrt(NNCIMatrixType matrix);

nnc_mtype NNCMatrixMean(NNCIMatrixType matrix);

NNCIMatrixType NNCMatrixClip(NNCIMatrixType matrix, nnc_mtype min_value, nnc_mtype max_value);

nnc_vector NNCMatrixDotProduct(NNCIMatrixType matrix, nnc_vector vector);
NNCIMatrixType NNCMatrixAddVector(NNCIMatrixType matrix, nnc_vector vector);
NNCIMatrixType NNCMatrixTranspose(NNCIMatrixType matrix);
nnc_vector NNCMatrixArgMax(NNCIMatrixType matrix);
nnc_vector NNCMatrixToVector(NNCIMatrixType matrix, bool axis);



typedef nnc_uint (*NNCMatrixIterationOperationSingle)(nnc_uint x, nnc_uint y, nnc_uint value);
typedef nnc_uint (*NNCMatrixIterationOperationDouble)(nnc_uint a_x, nnc_uint a_y, nnc_uint a_value, nnc_uint b_x, nnc_uint b_y, nnc_uint b_value);

// Inplace always in first matrix

NNCIMatrixType NNCMatrixIterationOperationSingleMatrix(NNCIMatrixType matrix, NNCMatrixIterationOperationSingle operation, nnc_bool inplace);
NNCIMatrixType NNCMatrixIterationOperationDoubleMatrix(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b, NNCMatrixIterationOperationDouble operation, nnc_bool inplace);

NNCIMatrixType NNCMatrixIterationOperationSingleMatrix_LazyIteration(NNCIMatrixType matrix, NNCMatrixIterationOperationSingle operation, nnc_bool inplace);
NNCIMatrixType NNCMatrixIterationOperationDoubleMatrix_LazyIteration(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b, NNCMatrixIterationOperationDouble operation, nnc_bool inplace);

NNCIMatrixType NNCMatrixIterationOperationSingleMatrix_ParallelIteration(NNCIMatrixType matrix, NNCMatrixIterationOperationSingle operation, nnc_bool inplace);
NNCIMatrixType NNCMatrixIterationOperationDoubleMatrix_ParallelIteration(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b, NNCMatrixIterationOperationDouble operation, nnc_bool inplace);





#endif //CMATRIX_NNC_MATRIX_H
