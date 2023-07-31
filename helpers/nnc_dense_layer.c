#include <malloc.h>
#include "nnc_dense_layer.h"
#include "nnc_vector.h"

NNCIDenseLayerType NNCDenseLayerAlloc(nnc_uint num_inputs, nnc_uint num_neurons) {
    NNCIDenseLayerType layer = malloc(sizeof(NNCDenseLayerType));

    layer->biases = NNCMatrixAllocBaseValue(num_neurons, 1, 0);
    layer->weights = NNCMatrixAllocSum(num_neurons, num_inputs);
//    layer->weights = NNCMatrixAllocRandom(num_neurons, num_inputs);
//    layer->weights = NNCMatrixAllocBaseValue(num_neurons, num_inputs, 1);
    layer->num_neurons = num_neurons;
    layer->num_inputs = num_inputs;
    layer->dinputs = NULL;
    layer->dbiases = NULL;
    layer->dweights = NULL;

    return layer;
}

void NNCDenseLayerDeAlloc(NNCIDenseLayerType layer) {
    NNCMatrixDeAlloc(layer->weights);
    NNCMatrixDeAlloc(layer->biases);

    if(layer->dweights != NULL) NNCMatrixDeAlloc(layer->dweights);
    if(layer->dbiases != NULL) NNCMatrixDeAlloc(layer->dbiases);
    if(layer->dinputs != NULL) NNCMatrixDeAlloc(layer->dinputs);

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

    NNCIMatrixType _dinputs  = NNCMatrixProduct(dvalues, _weightsT);

    if(layer->dweights != NULL) NNCMatrixDeAlloc(layer->dweights);
    if(layer->dbiases != NULL) NNCMatrixDeAlloc(layer->dbiases);
    if(layer->dinputs != NULL) NNCMatrixDeAlloc(layer->dinputs);

    layer->dweights = _dweights;
    layer->dinputs  = _dinputs;
    layer->dbiases  = _dbiases;

    NNCMatrixDeAlloc(_inputsT);
    NNCMatrixDeAlloc(_weightsT);
}
