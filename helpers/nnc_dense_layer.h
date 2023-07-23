#ifndef CMATRIX_NNC_DENSE_LAYER_H
#define CMATRIX_NNC_DENSE_LAYER_H


#include "nnc_matrix.h"

typedef struct NNCDenseLayerType
{
    NNCIMatrixType weights;     // [x - # inputa][y - # neurona]
    nnc_vector biases;          // isti kao # neurona
    nnc_uint num_inputs;
    nnc_uint num_neurons;
}
NNCDenseLayerType;

#define NNCIDenseLayerType NNCDenseLayerType*


NNCIDenseLayerType NNCDenseLayerAlloc(nnc_uint num_inputs, nnc_uint num_neurons);
void NNCDenseLayerDeAlloc(NNCIDenseLayerType layer);
//void NNCNeuronLayerPrint(NNCIMatrixType layer);

NNCIMatrixType NNCDenseLayerForward(NNCIMatrixType inputs, NNCIDenseLayerType layer);


#endif //CMATRIX_NNC_DENSE_LAYER_H
