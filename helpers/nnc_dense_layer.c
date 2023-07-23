#include <malloc.h>
#include "nnc_dense_layer.h"

NNCIDenseLayerType NNCDenseLayerAlloc(nnc_uint num_inputs, nnc_uint num_neurons) {
    NNCIDenseLayerType layer = malloc(sizeof(NNCIDenseLayerType));
    layer->biases = malloc(sizeof(nnc_mtype) * num_neurons);
    for(nnc_uint _x = 0; _x < num_neurons; _x ++) layer->biases[_x] = 0;
//    layer->weights = NNCMatrixAllocBaseValue(num_neurons, num_inputs, 1);
    layer->weights = NNCMatrixAllocRandom(num_neurons, num_inputs);
    layer->num_neurons = num_neurons;
    layer->num_inputs = num_inputs;
    return layer;
}

void NNCDenseLayerDeAlloc(NNCIDenseLayerType layer) {
    NNCMatrixDeAlloc(layer->weights);
    free(layer->biases);
    free(layer);
}

NNCIMatrixType NNCDenseLayerForward(NNCIMatrixType inputs, NNCIDenseLayerType layer) {
    NNCIMatrixType output  = NNCMatrixProduct(inputs, layer->weights);
    NNCIMatrixType _output = NNCMatrixAddVector(output, layer->biases);
    NNCMatrixDeAlloc(output);
    return _output;
}
