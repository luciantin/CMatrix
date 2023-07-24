#include <malloc.h>
#include "nnc_dense_layer.h"
#include "nnc_vector.h"

NNCIDenseLayerType NNCDenseLayerAlloc(nnc_uint num_inputs, nnc_uint num_neurons) {
    NNCIDenseLayerType layer = malloc(sizeof(NNCIDenseLayerType));

    layer->biases = NNCMatrixAllocBaseValue(num_neurons, 1, 1);
    layer->weights = NNCMatrixAllocSum(num_neurons, num_inputs);
    layer->num_neurons = num_neurons;
    layer->num_inputs = num_inputs;

    return layer;
}

void NNCDenseLayerDeAlloc(NNCIDenseLayerType layer) {
    NNCMatrixDeAlloc(layer->weights);
    NNCMatrixDeAlloc(layer->biases);
    free(layer);
}

NNCIMatrixType NNCDenseLayerForward(NNCIMatrixType inputs, NNCIDenseLayerType layer) {
    layer->inputs = inputs;
    NNCIMatrixType output  = NNCMatrixProduct(inputs, layer->weights);
    NNCIMatrixType _output = NNCMatrixSum(output, layer->biases);
    NNCMatrixDeAlloc(output);
    return _output;
}

void NNCDenseLayerBackward(NNCIMatrixType dvalues, NNCIDenseLayerType layer) {
    NNCIMatrixType _inputsT  = NNCMatrixTranspose(layer->inputs);
    NNCIMatrixType _dweights = NNCMatrixProduct(_inputsT, dvalues);
    NNCIMatrixType _dbiases  = NNCMatrixSumSingle(dvalues, 0);
    NNCIMatrixType _weightsT = NNCMatrixTranspose(layer->weights);
    NNCIMatrixType _dinputs  = NNCMatrixSum(_weightsT, dvalues);

    layer->dweights = _dweights;
    layer->dinputs  = _dinputs;
    layer->dbiases  = _dbiases;

    NNCMatrixDeAlloc(_inputsT);
    NNCMatrixDeAlloc(_weightsT);
}
