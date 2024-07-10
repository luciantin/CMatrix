#include <malloc.h>
#include "nnc_config.h"
#include "nnc_matrix.h"
#include <math.h>


/////////////////////////////////////////////////////////////
//              ALLOC / DEALLOC                            //
/////////////////////////////////////////////////////////////


NNCIMatrixType NNCMatrixAllocBaseValue(nnc_uint x, nnc_uint y, nnc_mtype base_value) {
    NNCIMatrixType matrix = malloc(sizeof(NNCMatrixType));
    matrix->matrix = malloc(sizeof(nnc_mtype*) * y);
    matrix->x = x; matrix->y = y;
    for(int _y = 0; _y < y; _y ++) matrix->matrix[_y] = malloc(sizeof(nnc_mtype) * x);
    for(int _y = 0; _y < matrix->y; _y ++) for(int _x = 0; _x < matrix->x; _x ++) matrix->matrix[_y][_x] = base_value;
    return matrix;
}

NNCIMatrixType NNCMatrixAlloc(nnc_uint x, nnc_uint y) {
    NNCIMatrixType matrix = malloc(sizeof(NNCMatrixType));
    matrix->matrix = malloc(sizeof(nnc_mtype*) * y);
    matrix->x = x; matrix->y = y;
    for(int _y = 0; _y < y; _y ++) matrix->matrix[_y] = malloc(sizeof(nnc_mtype) * x);
    return matrix;
}

NNCIMatrixType NNCMatrixAllocRandom(nnc_uint x, nnc_uint y) {
    NNCIMatrixType matrix = malloc(sizeof(NNCMatrixType));
    matrix->matrix = malloc(sizeof(nnc_mtype*) * y);
    matrix->x = x; matrix->y = y;
    for(int _y = 0; _y < y; _y ++) matrix->matrix[_y] = malloc(sizeof(nnc_mtype) * x);
    for(int _y = 0; _y < matrix->y; _y ++)
        for(int _x = 0; _x < matrix->x; _x ++)
            matrix->matrix[_y][_x] = NNCGetRandomMType();
    return matrix;
}

NNCIMatrixType NNCMatrixAllocBernoulli(nnc_uint x, nnc_uint y, nnc_mtype p, nnc_mtype divider){
    NNCIMatrixType matrix = malloc(sizeof(NNCMatrixType));
    matrix->matrix = malloc(sizeof(nnc_mtype*) * y);
    matrix->x = x; matrix->y = y;
    for(int _y = 0; _y < y; _y ++) matrix->matrix[_y] = malloc(sizeof(nnc_mtype) * x);
    for(int _y = 0; _y < matrix->y; _y ++)
        for(int _x = 0; _x < matrix->x; _x ++){
            nnc_mtype r = NNCGetRandomUnsignedMType();
            matrix->matrix[_y][_x] = r <= p ? 1 : 0;
            if(divider != 0) matrix->matrix[_y][_x] /= divider;
        }

    return matrix;
}

NNCIMatrixType NNCMatrixAllocLine(unsigned int x, unsigned int y) {
    NNCIMatrixType matrix = malloc(sizeof(NNCMatrixType));
    matrix->matrix = malloc(sizeof(nnc_mtype*) * y);
    matrix->x = x; matrix->y = y;
    for(int _y = 0; _y < y; _y ++) matrix->matrix[_y] = malloc(sizeof(nnc_mtype) * x);
    for(int _y = 0; _y < matrix->y; _y ++) for(int _x = 0; _x < matrix->x; _x ++) matrix->matrix[_y][_x] = _y;
    return matrix;
}

NNCIMatrixType NNCMatrixAllocSum(unsigned int x, unsigned int y) {
    NNCIMatrixType matrix = malloc(sizeof(NNCMatrixType));
    matrix->matrix = malloc(sizeof(nnc_mtype*) * y);
    matrix->x = x; matrix->y = y;
    for(int _y = 0; _y < y; _y ++) matrix->matrix[_y] = malloc(sizeof(nnc_mtype) * x);
    for(int _y = 0; _y < matrix->y; _y ++) for(int _x = 0; _x < matrix->x; _x ++) matrix->matrix[_y][_x] = _y + _x;
    return matrix;
}

NNCIMatrixType NNCMatrixAllocDiagonal(nnc_uint x, nnc_uint y, nnc_mtype base_value){
    NNCIMatrixType matrix = malloc(sizeof(NNCMatrixType));
    matrix->matrix = malloc(sizeof(nnc_mtype*) * y);
    matrix->x = x; matrix->y = y;
    for(int _y = 0; _y < y; _y ++) matrix->matrix[_y] = malloc(sizeof(nnc_mtype) * x);
    for(int _y = 0; _y < matrix->y; _y ++)
        for(int _x = 0; _x < matrix->x; _x ++){
            if(_y == _x) matrix->matrix[_y][_x] = base_value;
            else matrix->matrix[_y][_x] = 0;
        }

    return matrix;
}

void NNCMatrixDeAlloc(NNCIMatrixType matrix) {
    for(int _y = 0; _y < matrix->y; _y ++) free(matrix->matrix[_y]);
    free(matrix->matrix);
    free(matrix);
}

/////////////////////////////////////////////////////////////
//              HELPERS                                    //
/////////////////////////////////////////////////////////////

void NNCMatrixPrint(NNCMatrixType *matrix) {
    if(matrix == nnc_null) return;
    for(int _y = 0; _y < matrix->y; _y ++){
        for(int _x = 0; _x < matrix->x; _x ++){
            nnc_vector vec = matrix->matrix[_y];
            dprintf(" %.9g", matrix->matrix[_y][_x]);
        }
        dprintf("\n");
    }
}

/////////////////////////////////////////////////////////////
//              ALGO                                       //
/////////////////////////////////////////////////////////////

NNCIMatrixType NNCMatrixProduct(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b) {
//#if NNC_MATRIX_MULTIPLY_SQUARE_ALGO == NNC_MATRIX_MULTIPLY_SQUARE_ALGO_ITERATIVE
    if(matrix_a->x == matrix_b->x && matrix_a->y == matrix_b->y){
        NNCIMatrixType matrix_out = NNCMatrixAllocBaseValue(matrix_a->x, matrix_a->y, 0);
        for (int y = 0; y < matrix_out->y; y++)
                for (int x = 0; x < matrix_out->x; x++)
                    matrix_out->matrix[y][x] += matrix_a->matrix[y][x] * matrix_b->matrix[y][x];
        return matrix_out;
    }else {//if(matrix_a->x < matrix_b->x && matrix_a->y == matrix_b->y){
        NNCIMatrixType matrix_out = NNCMatrixAllocBaseValue(matrix_b->x, matrix_a->y, 0);
        for (int i = 0; i < matrix_a->y; i++)
            for (int j = 0; j < matrix_b->x; j++)
                for (int k = 0; k < matrix_b->y; k++) // FIXME a < b
                    matrix_out->matrix[i][j] += matrix_a->matrix[i][k] * matrix_b->matrix[k][j];
        return matrix_out;
    }
//    else if(matrix_a->x > matrix_b->x && matrix_a->y == matrix_b->y){
//        NNCIMatrixType matrix_out = NNCMatrixAllocBaseValue(matrix_b->x, matrix_a->y, 0);
//        for (int i = 0; i < matrix_b->y; i++)
//            for (int j = 0; j < matrix_a->x; j++)
//                for (int k = 0; k < matrix_a->y; k++) // FIXME a < b
//                    matrix_out->matrix[i][j] += matrix_b->matrix[i][k] * matrix_a->matrix[k][j];
//        return matrix_out;
//    }
//#endif
}

NNCIMatrixType NNCMatrixQuotient(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b) {
    NNCIMatrixType matrix_b_neg_exp = NNCMatrixQuotientNumberReverse(1, matrix_b);
    if(matrix_a->x == matrix_b->x && matrix_a->y == matrix_b->y){
        NNCIMatrixType matrix_out = NNCMatrixAllocBaseValue(matrix_a->x, matrix_a->y, 0);
        for (int y = 0; y < matrix_out->y; y++)
            for (int x = 0; x < matrix_out->x; x++)
                matrix_out->matrix[y][x] += matrix_a->matrix[y][x] * matrix_b_neg_exp->matrix[y][x];
        NNCMatrixDeAlloc(matrix_b_neg_exp);
        return matrix_out;
    }else{
        NNCIMatrixType matrix_out = NNCMatrixAllocBaseValue(matrix_b->x, matrix_a->y, 0);
        for (int i = 0; i < matrix_a->y; i++)
            for (int j = 0; j < matrix_b->x; j++)
                for (int k = 0; k < matrix_b->y; k++) // FIXME a < b
                    matrix_out->matrix[i][j] += matrix_a->matrix[i][k] * matrix_b_neg_exp->matrix[k][j];
        NNCMatrixDeAlloc(matrix_b_neg_exp);
        return matrix_out;
    }
}

nnc_vector NNCMatrixDotProduct(NNCIMatrixType matrix, nnc_vector vector) {
    nnc_vector vector_out = malloc(sizeof(nnc_mtype)*matrix->x);
    for(int _y = 0; _y < matrix->y; _y ++) {
        vector_out[_y] = 0;
        for(int _x = 0; _x < matrix->x; _x ++)
            vector_out[_y] += matrix->matrix[_y][_x] * vector[_x];
    }
    return vector_out;
}

NNCIMatrixType NNCMatrixAddVector(NNCIMatrixType matrix, nnc_vector vector) {
    NNCIMatrixType matrix_out = NNCMatrixAlloc(matrix->x, matrix->y);
    for(int _y = 0; _y < matrix->y; _y ++)
        for(int _x = 0; _x < matrix->x; _x ++)
            matrix_out->matrix[_y][_x] = matrix->matrix[_y][_x] + vector[_x];
    return matrix_out;
}

NNCIMatrixType NNCMatrixTranspose(NNCIMatrixType matrix) {
    NNCIMatrixType matrix_out = NNCMatrixAlloc(matrix->y, matrix->x);
    for(int _y = 0; _y < matrix->y; _y ++)
        for(int _x = 0; _x < matrix->x; _x ++)
            matrix_out->matrix[_x][_y] = matrix->matrix[_y][_x];
    return matrix_out;
}

nnc_vector NNCMatrixArgMax(NNCIMatrixType matrix) {
    nnc_vector vector_out = malloc(sizeof(nnc_mtype) * matrix->y);

    for(int _y = 0; _y < matrix->y; _y ++){
        nnc_uint max_index = 0;
        nnc_mtype max_value = matrix->matrix[_y][0];
        for(int _x = 0; _x < matrix->x; _x ++){
            if(matrix->matrix[_y][_x] > max_value){
                max_value = matrix->matrix[_y][_x];
                max_index = _x;
            }
        }
        vector_out[_y] = max_index;
    }

    return vector_out;
}

NNCIMatrixType NNCMatrixSum(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b) {
    if(matrix_a->y == matrix_b->y && matrix_a->x == matrix_b->x){
        NNCIMatrixType matrix_c = NNCMatrixAlloc(matrix_a->x, matrix_a->y);
        for(nnc_uint x = 0; x < matrix_a->x; x++) for(nnc_uint y = 0; y < matrix_a->y; y++) matrix_c->matrix[y][x] = matrix_a->matrix[y][x] + matrix_b->matrix[y][x];
        return matrix_c;
    } else if(matrix_a->x == matrix_b->x && matrix_b->y == 1){
        NNCIMatrixType matrix_c = NNCMatrixAlloc(matrix_a->x, matrix_a->y);
        for(nnc_uint x = 0; x < matrix_a->x; x++) for(nnc_uint y = 0; y < matrix_a->y; y++) matrix_c->matrix[y][x] = matrix_a->matrix[y][x] + matrix_b->matrix[0][x];
        return matrix_c;
    } else if(matrix_a->y == matrix_b->y && matrix_b->x == 1){
        NNCIMatrixType matrix_c = NNCMatrixAlloc(matrix_a->x, matrix_a->y);
        for(nnc_uint x = 0; x < matrix_a->x; x++) for(nnc_uint y = 0; y < matrix_a->y; y++) matrix_c->matrix[y][x] = matrix_a->matrix[y][x] + matrix_b->matrix[y][0];
        return matrix_c;
    }
    return nnc_null;
}

NNCIMatrixType NNCMatrixSumSingle(NNCIMatrixType matrix, bool axis) {
    if(axis){
        NNCIMatrixType matrix_out = NNCMatrixAllocBaseValue(1, matrix->y, 0);
        for(int _y = 0; _y < matrix->y; _y ++) for(int _x = 0; _x < matrix->x; _x ++) matrix_out->matrix[_y][0] += matrix->matrix[_y][_x];
        return matrix_out;
    }else{
        NNCIMatrixType matrix_out = NNCMatrixAllocBaseValue(matrix->x, 1, 0);
        for(int _x = 0; _x < matrix->x; _x ++) for(int _y = 0; _y < matrix->y; _y ++) matrix_out->matrix[0][_x] += matrix->matrix[_y][_x];
        return matrix_out;
    }
}

nnc_mtype NNCMatrixSumAll(NNCIMatrixType matrix) {
    nnc_mtype sum = 0;
    for (int _y = 0; _y < matrix->y; _y++)
        for (int _x = 0; _x < matrix->x; _x++)
            sum += matrix->matrix[_y][_x];
    return sum;
}

nnc_mtype NNCMatrixSumAllAbs(NNCIMatrixType matrix) {
    nnc_mtype sum = 0;
    for (int _y = 0; _y < matrix->y; _y++)
        for (int _x = 0; _x < matrix->x; _x++)
            sum += fabsf(matrix->matrix[_y][_x]);
    return sum;
}

NNCIMatrixType NNCMatrixProductNumber(NNCIMatrixType matrix, nnc_mtype num) {
    NNCIMatrixType matrix_out = NNCMatrixAlloc(matrix->x, matrix->y);
    for(int _y = 0; _y < matrix->y; _y ++) for(int _x = 0; _x < matrix->x; _x ++) matrix_out->matrix[_y][_x] = matrix->matrix[_y][_x] * num;
    return matrix_out;
}

NNCIMatrixType NNCMatrixQuotientNumber(NNCIMatrixType matrix, nnc_mtype num) {
    NNCIMatrixType matrix_out = NNCMatrixAlloc(matrix->x, matrix->y);
    for(int _y = 0; _y < matrix->y; _y ++) for(int _x = 0; _x < matrix->x; _x ++) matrix_out->matrix[_y][_x] = matrix->matrix[_y][_x] / num;
    return matrix_out;
}

NNCIMatrixType NNCMatrixQuotientNumberReverse(nnc_mtype num, NNCIMatrixType matrix){
    NNCIMatrixType matrix_out = NNCMatrixAlloc(matrix->x, matrix->y);
    for(int _y = 0; _y < matrix->y; _y ++) for(int _x = 0; _x < matrix->x; _x ++) matrix_out->matrix[_y][_x] = num / matrix->matrix[_y][_x];
    return matrix_out;
}

// ako je vrijednost matrice > max_value onda postavi elem. na max_value
// ako je vrijednost matrice < min_value onda postavi elem. na min_value
NNCIMatrixType NNCMatrixClip(NNCIMatrixType matrix, nnc_mtype min_value, nnc_mtype max_value){
    NNCIMatrixType matrix_out = NNCMatrixAlloc(matrix->x, matrix->y);
    for(int _y = 0; _y < matrix->y; _y ++) for(int _x = 0; _x < matrix->x; _x ++)
        matrix_out->matrix[_y][_x] = matrix->matrix[_y][_x] > max_value ? max_value : matrix->matrix[_y][_x] < min_value ? min_value : matrix->matrix[_y][_x];
    return matrix_out;
}

NNCIMatrixType NNCMatrixSub(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b){
    if(matrix_a->y == matrix_b->y && matrix_a->x == matrix_b->x){
        NNCIMatrixType matrix_c = NNCMatrixAlloc(matrix_a->x, matrix_a->y);
        for(nnc_uint x = 0; x < matrix_a->x; x++) for(nnc_uint y = 0; y < matrix_a->y; y++) matrix_c->matrix[y][x] = matrix_a->matrix[y][x] - matrix_b->matrix[y][x];
        return matrix_c;
    } else if(matrix_a->x == matrix_b->x && matrix_b->y == 1){
        NNCIMatrixType matrix_c = NNCMatrixAlloc(matrix_a->x, matrix_a->y);
        for(nnc_uint x = 0; x < matrix_a->x; x++) for(nnc_uint y = 0; y < matrix_a->y; y++) matrix_c->matrix[y][x] = matrix_a->matrix[y][x] - matrix_b->matrix[0][x];
        return matrix_c;
    } else if(matrix_a->y == matrix_b->y && matrix_b->x == 1){
        NNCIMatrixType matrix_c = NNCMatrixAlloc(matrix_a->x, matrix_a->y);
        for(nnc_uint x = 0; x < matrix_a->x; x++) for(nnc_uint y = 0; y < matrix_a->y; y++) matrix_c->matrix[y][x] = matrix_a->matrix[y][x] - matrix_b->matrix[y][0];
        return matrix_c;
    }
    return nnc_null;
}

NNCIMatrixType NNCMatrixSqrt(NNCIMatrixType matrix){
    NNCIMatrixType output = NNCMatrixAlloc(matrix->x, matrix->y);
    for(nnc_uint y = 0; y < matrix->y; y++) for(nnc_uint x = 0; x < matrix->x; x++) output->matrix[y][x] = sqrtf(matrix->matrix[y][x]);
    return output;
}

NNCIMatrixType NNCMatrixSumNumber(NNCIMatrixType matrix, nnc_mtype number){
    NNCIMatrixType output = NNCMatrixAlloc(matrix->x, matrix->y);
    for(nnc_uint y = 0; y < matrix->y; y++) for(nnc_uint x = 0; x < matrix->x; x++) output->matrix[y][x] = matrix->matrix[y][x] + number;
    return output;
}

nnc_mtype NNCMatrixMean(NNCIMatrixType matrix){
    nnc_mtype mean = 0;
    for(nnc_uint y = 0; y < matrix->y; y++) for(nnc_uint x = 0; x < matrix->x; x++) mean += matrix->matrix[y][x];
    mean = mean / (matrix->x * matrix->y);
    return mean;
}

nnc_vector NNCMatrixToVector(NNCIMatrixType matrix, bool axis){
    if(!axis){
        nnc_vector vector_out = malloc(sizeof(nnc_mtype) * matrix->x);
        for(nnc_uint x = 0; x < matrix->x; x++) vector_out[x] = matrix->matrix[0][x];
        return vector_out;
    }else{
        nnc_vector vector_out = malloc(sizeof(nnc_mtype) * matrix->y);
        for(nnc_uint y = 0; y < matrix->y; y++) vector_out[y] = matrix->matrix[y][0];
        return vector_out;
    }
}






NNCIMatrixType NNCMatrixIterationOperationSingleMatrix(NNCIMatrixType matrix, NNCMatrixIterationOperationSingle operation, nnc_bool inplace){
#if NNC_PARALLEL == 1
    return NNCMatrixIterationOperationSingleMatrix_ParallelIteration(matrix, operation, inplace);
#else
    return NNCMatrixIterationOperationSingleMatrix_LazyIteration(matrix, operation, inplace);
#endif
}

NNCIMatrixType NNCMatrixIterationOperationDoubleMatrix(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b, NNCMatrixIterationOperationDouble operation, nnc_bool inplace){
#if NNC_PARALLEL == 1
    return NNCMatrixIterationOperationDoubleMatrix_ParallelIteration(matrix_a, matrix_b, operation, inplace);
#else
    return NNCMatrixIterationOperationDoubleMatrix_LazyIteration(matrix_a, matrix_b, operation, inplace);
#endif
}

NNCIMatrixType NNCMatrixIterationOperationSingleMatrix_LazyIteration(NNCIMatrixType matrix, NNCMatrixIterationOperationSingle operation, nnc_bool inplace){

}

NNCIMatrixType NNCMatrixIterationOperationDoubleMatrix_LazyIteration(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b, NNCMatrixIterationOperationDouble operation, nnc_bool inplace){
    if(matrix_a->x == matrix_b->x && matrix_a->y == matrix_b->y){
        NNCIMatrixType matrix_out = NNCMatrixAllocBaseValue(matrix_a->x, matrix_a->y, 0);
        for (int y = 0; y < matrix_out->y; y++)
            for (int x = 0; x < matrix_out->x; x++)
                matrix_out->matrix[y][x] += matrix_a->matrix[y][x] * matrix_b->matrix[y][x];
        return matrix_out;
    }else {//if(matrix_a->x < matrix_b->x && matrix_a->y == matrix_b->y){
        NNCIMatrixType matrix_out = NNCMatrixAllocBaseValue(matrix_b->x, matrix_a->y, 0);
        for (int i = 0; i < matrix_a->y; i++)
            for (int j = 0; j < matrix_b->x; j++)
                for (int k = 0; k < matrix_b->y; k++) // FIXME a < b
                    matrix_out->matrix[i][j] += matrix_a->matrix[i][k] * matrix_b->matrix[k][j];
        return matrix_out;
    }
}

NNCIMatrixType NNCMatrixIterationOperationSingleMatrix_ParallelIteration(NNCIMatrixType matrix, NNCMatrixIterationOperationSingle operation, nnc_bool inplace){
#if NNC_PARALLEL == 1
#else
    return nnc_null;
#endif
}

NNCIMatrixType NNCMatrixIterationOperationDoubleMatrix_ParallelIteration(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b, NNCMatrixIterationOperationDouble operation, nnc_bool inplace){
#if NNC_PARALLEL == 1
#else
    return nnc_null;
#endif
}



