#ifndef CMATRIX_NNC_LAYER_H
#define CMATRIX_NNC_LAYER_H


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

    NNCIMatrixType  cweights;     // cache of weights
    NNCIMatrixType  cbiases;      // cache of biases

    nnc_mtype       l1r_weights;
    nnc_mtype       l2r_weights;

    nnc_mtype       l1r_biases;
    nnc_mtype       l2r_biases;

    nnc_uint        num_inputs;
    nnc_uint        num_neurons;
}
NNCDenseLayerType;

#define NNCIDenseLayerType NNCDenseLayerType*


NNCIDenseLayerType NNCDenseLayerAlloc(nnc_uint num_inputs, nnc_uint num_neurons);
void NNCDenseLayerDeAlloc(NNCIDenseLayerType layer);

void NNCDenseLayerSetRegularizationParameters(NNCIDenseLayerType layer, nnc_mtype l1r_weights, nnc_mtype l2r_weights, nnc_mtype l1r_biases, nnc_mtype l2r_biases);
nnc_mtype NNCDenseLayerCalculateRegularizationLoss(NNCIDenseLayerType layer);
NNCIMatrixType NNCDenseLayerForward(NNCIMatrixType inputs, NNCIDenseLayerType layer);
void NNCDenseLayerBackward(NNCIMatrixType dvalues, NNCIDenseLayerType layer);
void NNCDenseLayerWithRegularizationBackward(NNCIMatrixType dvalues, NNCIDenseLayerType layer);

//

typedef struct NNCDropoutLayerType
{
    NNCIMatrixType  binary_mask;      // [x - # inputa][y - # neurona]
    NNCIMatrixType  dinputs;      //

    nnc_mtype       dropout_rate;
//    nnc_uint        num_inputs;
//    nnc_uint        num_neurons;
}
NNCDropoutLayerType;

#define NNCIDropoutLayerType NNCDropoutLayerType*


NNCIDropoutLayerType NNCDropoutLayerAlloc(nnc_mtype dropout_rate);
void NNCDropoutLayerDeAlloc(NNCIDropoutLayerType layer);

NNCIMatrixType NNCDropoutLayerForward(NNCIMatrixType inputs, NNCIDropoutLayerType layer);
void NNCDropoutLayerBackward(NNCIMatrixType dvalues, NNCIDropoutLayerType layer);


#endif //CMATRIX_NNC_LAYER_H
