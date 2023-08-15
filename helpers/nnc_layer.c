#include <malloc.h>
#include "nnc_layer.h"
#include "nnc_vector.h"

NNCIDenseLayerType NNCDenseLayerAlloc(nnc_uint num_inputs, nnc_uint num_neurons) {
    NNCIDenseLayerType layer = malloc(sizeof(NNCDenseLayerType));

    layer->biases = NNCMatrixAllocBaseValue(num_neurons, 1, 0);

//    layer->weights = NNCMatrixAllocSum(num_neurons, num_inputs);
    layer->weights = NNCMatrixAllocRandom(num_neurons, num_inputs);
//    layer->weights = NNCMatrixAllocBaseValue(num_neurons, num_inputs, 1);

    layer->num_neurons = num_neurons;
    layer->num_inputs = num_inputs;

    layer->dinputs = nnc_null;
    layer->dbiases = nnc_null;
    layer->dweights = nnc_null;

    layer->mbiases = nnc_null;
    layer->mweights = nnc_null;

    layer->cbiases = nnc_null;
    layer->cweights = nnc_null;

    layer->l1r_weights = 0;
    layer->l2r_weights = 0;
    layer->l1r_biases  = 0;
    layer->l2r_biases  = 0;

    return layer;
}

void NNCDenseLayerDeAlloc(NNCIDenseLayerType layer) {
    NNCMatrixDeAlloc(layer->weights);
    NNCMatrixDeAlloc(layer->biases);

    if(layer->dweights != nnc_null) NNCMatrixDeAlloc(layer->dweights);
    if(layer->dbiases != nnc_null) NNCMatrixDeAlloc(layer->dbiases);
    if(layer->dinputs != nnc_null) NNCMatrixDeAlloc(layer->dinputs);

    if(layer->mbiases != nnc_null) NNCMatrixDeAlloc(layer->mbiases);
    if(layer->mweights != nnc_null) NNCMatrixDeAlloc(layer->mweights);
    if(layer->cbiases != nnc_null) NNCMatrixDeAlloc(layer->cbiases);
    if(layer->cweights != nnc_null) NNCMatrixDeAlloc(layer->cweights);

    free(layer);
}

void NNCDenseLayerSetRegularizationParameters(NNCIDenseLayerType layer, nnc_mtype l1r_weights, nnc_mtype l2r_weights, nnc_mtype l1r_biases, nnc_mtype l2r_biases){
    layer->l1r_weights = l1r_weights;
    layer->l2r_weights = l2r_weights;
    layer->l1r_biases  = l1r_biases;
    layer->l2r_biases  = l2r_biases;
}

nnc_mtype NNCDenseLayerCalculateRegularizationLoss(NNCIDenseLayerType layer){
    nnc_mtype regularization_loss = 0;

    if(layer->l1r_weights > 0) regularization_loss += NNCMatrixSumAllAbs(layer->weights) * layer->l1r_weights;
    if(layer->l2r_weights > 0) {
        NNCIMatrixType sqr = NNCMatrixProduct(layer->weights, layer->weights);
        regularization_loss += NNCMatrixSumAll(sqr) * layer->l2r_weights;
        NNCMatrixDeAlloc(sqr);
    }
    if(layer->l1r_biases > 0) regularization_loss += NNCMatrixSumAllAbs(layer->biases) * layer->l1r_biases;
    if(layer->l2r_biases > 0) {
        NNCIMatrixType sqr = NNCMatrixProduct(layer->biases, layer->biases);
        regularization_loss += NNCMatrixSumAll(sqr) * layer->l2r_biases;
        NNCMatrixDeAlloc(sqr);
    }

    return regularization_loss;
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

    if(layer->dweights != nnc_null) NNCMatrixDeAlloc(layer->dweights);
    if(layer->dbiases != nnc_null) NNCMatrixDeAlloc(layer->dbiases);
    if(layer->dinputs != nnc_null) NNCMatrixDeAlloc(layer->dinputs);

    layer->dweights = _dweights;
    layer->dinputs  = _dinputs;
    layer->dbiases  = _dbiases;

    NNCMatrixDeAlloc(_inputsT);
    NNCMatrixDeAlloc(_weightsT);
}

void NNCDenseLayerWithRegularizationBackward(NNCIMatrixType dvalues, NNCIDenseLayerType layer) {
    NNCIMatrixType _inputsT  = NNCMatrixTranspose(layer->inputs);
    NNCIMatrixType _dweights = NNCMatrixProduct(_inputsT, dvalues);
    NNCIMatrixType _dbiases  = NNCMatrixSumSingle(dvalues, 0);
    NNCIMatrixType _weightsT = NNCMatrixTranspose(layer->weights);
    NNCIMatrixType _dinputs  = NNCMatrixProduct(dvalues, _weightsT);

    if(layer->dweights != nnc_null) NNCMatrixDeAlloc(layer->dweights);
    if(layer->dbiases != nnc_null) NNCMatrixDeAlloc(layer->dbiases);
    if(layer->dinputs != nnc_null) NNCMatrixDeAlloc(layer->dinputs);

    layer->dweights = _dweights;
    layer->dinputs  = _dinputs;
    layer->dbiases  = _dbiases;

    NNCMatrixDeAlloc(_inputsT);
    NNCMatrixDeAlloc(_weightsT);

    if(layer->l1r_weights > 0){
        NNCIMatrixType dl1r_w = NNCMatrixAllocBaseValue(layer->weights->x, layer->weights->y, 1);
        for(int _y = 0; _y < dl1r_w->y; _y ++) for(int _x = 0; _x < dl1r_w->x; _x ++) if(layer->weights->matrix[_y][_x] < 0) dl1r_w->matrix[_y][_x] = -1;
        NNCIMatrixType dl1r_wp = NNCMatrixProductNumber(dl1r_w, layer->l1r_weights);
        NNCIMatrixType dl1r_ws = NNCMatrixSum(layer->dweights, dl1r_wp);
        NNCMatrixDeAlloc(layer->dweights);
        NNCMatrixDeAlloc(dl1r_w);
        NNCMatrixDeAlloc(dl1r_wp);
        layer->dweights = dl1r_ws;
    }

    if(layer->l2r_weights){
        NNCIMatrixType dl2r_w = NNCMatrixProductNumber(layer->weights, 2*layer->l2r_weights);
        NNCIMatrixType dl2r_ws = NNCMatrixSum(layer->dweights, dl2r_w);
        NNCMatrixDeAlloc(dl2r_w);
        NNCMatrixDeAlloc(layer->dweights);
        layer->dweights = dl2r_ws;
    }

    if(layer->l1r_biases > 0){
        NNCIMatrixType dl1r_b = NNCMatrixAllocBaseValue(layer->biases->x, layer->biases->y, 1);
        for(int _y = 0; _y < dl1r_b->y; _y ++) for(int _x = 0; _x < dl1r_b->x; _x ++) if(layer->biases->matrix[_y][_x] < 0) dl1r_b->matrix[_y][_x] = -1;
        NNCIMatrixType dl1r_bp = NNCMatrixProductNumber(dl1r_b, layer->l1r_biases);
        NNCIMatrixType dl1r_bs = NNCMatrixSum(layer->dbiases, dl1r_bp);
        NNCMatrixDeAlloc(layer->dbiases);
        NNCMatrixDeAlloc(dl1r_b);
        NNCMatrixDeAlloc(dl1r_bp);
        layer->dbiases = dl1r_bs;
    }

    if(layer->l2r_biases){
        NNCIMatrixType dl2r_b = NNCMatrixProductNumber(layer->biases, 2*layer->l2r_biases);
        NNCIMatrixType dl2r_bs = NNCMatrixSum(layer->dbiases, dl2r_b);
        NNCMatrixDeAlloc(dl2r_b);
        NNCMatrixDeAlloc(layer->dbiases);
        layer->dbiases = dl2r_bs;
    }

}


//


NNCIDropoutLayerType NNCDropoutLayerAlloc(nnc_mtype dropout_rate) {
    NNCIDropoutLayerType layer = malloc(sizeof(NNCDropoutLayerType));

    layer->binary_mask = nnc_null;
    layer->dinputs = nnc_null;
    layer->dropout_rate = 1 - dropout_rate;

    return layer;
}

void NNCDropoutLayerDeAlloc(NNCIDropoutLayerType layer) {
    if(layer->binary_mask != nnc_null) NNCMatrixDeAlloc(layer->binary_mask);
    if(layer->dinputs != nnc_null) NNCMatrixDeAlloc(layer->dinputs);
    free(layer);
}

NNCIMatrixType NNCDropoutLayerForward(NNCIMatrixType inputs, NNCIDropoutLayerType layer) {
    if(layer->binary_mask != nnc_null) NNCMatrixDeAlloc(layer->binary_mask);
    layer->binary_mask = NNCMatrixAllocBernoulli(inputs->x, inputs->y, layer->dropout_rate, layer->dropout_rate);
//    layer->binary_mask = NNCMatrixAllocDiagonal(inputs->x, inputs->y, 1);
    return NNCMatrixProduct(inputs, layer->binary_mask);
}

void NNCDropoutLayerBackward(NNCIMatrixType dvalues, NNCIDropoutLayerType layer) {
    if(layer->dinputs != nnc_null) NNCMatrixDeAlloc(layer->dinputs);
    layer->dinputs = NNCMatrixProduct(dvalues, layer->binary_mask);
}


