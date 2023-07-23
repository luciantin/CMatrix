#include <stdio.h>
#include "nnc_vector.h"

void NNCVectorPrint(nnc_vector vector, nnc_uint len) {
    for(int _x = 0; _x < len; _x ++) printf(" %f ", vector[_x]);
    puts("");
}

nnc_mtype NNCVectorMean(nnc_vector vector, nnc_uint len) {
    nnc_mtype sum = 0;
    for(nnc_uint _x = 0; _x < len; _x ++) sum += vector[_x];
    return sum/len;
}
