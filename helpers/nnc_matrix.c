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
    return NULL;
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
