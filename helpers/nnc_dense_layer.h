#ifndef CMATRIX_NNC_DENSE_LAYER_H
#define CMATRIX_NNC_DENSE_LAYER_H


#include "nnc_matrix.h"

typedef struct NNCDenseLayerType
{
    NNCIMatrixType  weights;      // [x - # inputa][y - # neurona]
    NNCIMatrixType  inputs;       //
    NNCIMatrixType  biases;       // isti kao # neurona

    NNCIMatrixType  dweights;     // [x - # inputa][y - # neurona]
    NNCIMatrixType  dinputs;      //
    NNCIMatrixType  dbiases;      // isti kao # neurona

    NNCIMatrixType  mweights;     // momentum of weights
    NNCIMatrixType  mbiases;      // momentum of biases

    NNCIMatrixType cweights;      // cache of weights
    NNCIMatrixType cbiases;       // cache of biases

    nnc_uint        num_inputs;
    nnc_uint        num_neurons;
}
NNCDenseLayerType;

#define NNCIDenseLayerType NNCDenseLayerType*


NNCIDenseLayerType NNCDenseLayerAlloc(nnc_uint num_inputs, nnc_uint num_neurons);
void NNCDenseLayerDeAlloc(NNCIDenseLayerType layer);
//void NNCNeuronLayerPrint(NNCIMatrixType layer);

NNCIMatrixType NNCDenseLayerForward(NNCIMatrixType inputs, NNCIDenseLayerType layer);
void NNCDenseLayerBackward(NNCIMatrixType dvalues, NNCIDenseLayerType layer);

#endif //CMATRIX_NNC_DENSE_LAYER_H
