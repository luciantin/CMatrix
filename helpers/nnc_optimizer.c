#include <malloc.h>
#include "nnc_optimizer.h"


NNCIOptimizerSGDType NNCOptimizerSGDAlloc(nnc_mtype learning_rate){
    NNCIOptimizerSGDType opt = malloc(sizeof(NNCOptimizerSGDType));
    opt->learning_rate = learning_rate;
    return opt;
}

void NNCOptimizerSGDDeAlloc(NNCIOptimizerSGDType opt){
    free(opt);
}

void NNCOptimizerSGDUpdateParams(NNCIOptimizerSGDType opt, NNCIDenseLayerType layer){
    NNCIMatrixType _weights = NNCMatrixProductNumber(layer->dweights, -opt->learning_rate);
    NNCIMatrixType _biases  = NNCMatrixProductNumber(layer->dbiases,  -opt->learning_rate);

    NNCIMatrixType __weights = NNCMatrixSum(layer->weights, _weights);
    NNCIMatrixType __biases  = NNCMatrixSum(layer->biases, _biases);

    NNCMatrixDeAlloc(_weights);
    NNCMatrixDeAlloc(_biases);
    NNCMatrixDeAlloc(layer->weights);
    NNCMatrixDeAlloc(layer->biases);

    layer->weights = __weights;
    layer->biases  = __biases;
}