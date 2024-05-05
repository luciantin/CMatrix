#include "nnc_vector.h"

void NNCVectorPrint(nnc_vector vector, nnc_uint len) {
    for(int _x = 0; _x < len; _x ++) dprintf(" %.2g ", vector[_x]);
    dputs("");
}

nnc_mtype NNCVectorMean(nnc_vector vector, nnc_uint len) {
    nnc_mtype sum = 0;
    for(nnc_uint _x = 0; _x < len; _x ++) sum += vector[_x];
    return sum/len;
}

nnc_vector NNCVectorAdd(nnc_vector a, nnc_vector b, nnc_uint len) {
    return 0;
}

nnc_mtype NNCVectorAccuracy(nnc_vector a, nnc_vector b, nnc_uint len){
    nnc_mtype correct = 0;
    for(nnc_uint _x = 0; _x < len; _x ++) correct += a[_x] == b[_x] ? 1 : 0;
    correct = correct / len;
    return correct;
}

void NNCVectorPrintTargetPrediction(nnc_vector a, nnc_vector b, nnc_uint len){
    for(nnc_uint _x = 0; _x < len; _x ++) dprintf("Prediction : %f %f \n", a[_x], b[_x]);
}