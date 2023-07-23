#include <malloc.h>
#include <stdlib.h>
#include "nnc_config.h"
#include "nnc_matrix.h"

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

void NNCMatrixDeAlloc(NNCIMatrixType matrix) {
    for(int _y = 0; _y < matrix->y; _y ++) free(matrix->matrix[_y]);
    free(matrix->matrix);
    free(matrix);
}

void NNCMatrixPrint(NNCMatrixType *matrix) {
    for(int _y = 0; _y < matrix->y; _y ++){
        for(int _x = 0; _x < matrix->x; _x ++){
            nnc_vector vec = matrix->matrix[_y];
            printf(" %f", matrix->matrix[_y][_x]);
        }
        printf("\n");
    }
}

NNCIMatrixType NNCMatrixProduct(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b) {
    NNCIMatrixType matrix_out = NNCMatrixAllocBaseValue(matrix_b->x, matrix_a->y, 0);
    for (int i = 0; i < matrix_a->y; i++)
        for (int j = 0; j < matrix_b->x; j++)
            for (int k = 0; k < matrix_b->y; k++)
                matrix_out->matrix[i][j] += matrix_a->matrix[i][k] * matrix_b->matrix[k][j];
    return matrix_out;
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
    nnc_vector vector_out = malloc(sizeof(nnc_mtype)*matrix->x);

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

NNCIMatrixType NNCMatrixAllocSampleLine(unsigned int x, unsigned int y) {
    NNCIMatrixType matrix = malloc(sizeof(NNCMatrixType));
    matrix->matrix = malloc(sizeof(nnc_mtype*) * y);
    matrix->x = x; matrix->y = y;
    for(int _y = 0; _y < y; _y ++) matrix->matrix[_y] = malloc(sizeof(nnc_mtype) * x);
    for(int _y = 0; _y < matrix->y; _y ++) for(int _x = 0; _x < matrix->x; _x ++) matrix->matrix[_y][_x] = _y;
    return matrix;
}

NNCIMatrixType NNCMatrixAllocSampleSum(unsigned int x, unsigned int y) {
    NNCIMatrixType matrix = malloc(sizeof(NNCMatrixType));
    matrix->matrix = malloc(sizeof(nnc_mtype*) * y);
    matrix->x = x; matrix->y = y;
    for(int _y = 0; _y < y; _y ++) matrix->matrix[_y] = malloc(sizeof(nnc_mtype) * x);
    for(int _y = 0; _y < matrix->y; _y ++) for(int _x = 0; _x < matrix->x; _x ++) matrix->matrix[_y][_x] = _y + _x;
    return matrix;
}

