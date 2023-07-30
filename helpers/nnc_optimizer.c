#include <malloc.h>
#include "nnc_optimizer.h"


NNCIOptimizerSGDType NNCOptimizerSGDAlloc(nnc_mtype learning_rate, nnc_mtype decay){
    NNCIOptimizerSGDType opt = malloc(sizeof(NNCOptimizerSGDType));
    opt->learning_rate = learning_rate;
    opt->current_learning_rate = learning_rate;
    opt->decay = decay;
    opt->iteration = 0;
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

void NNCOptimizerSGDPreUpdateParams(NNCIOptimizerSGDType opt){
    if(opt->decay != 0){
        opt->current_learning_rate = opt->learning_rate * ( 1.0 / ( 1.0 + opt->decay * opt->iteration));
    }
}

void NNCOptimizerSGDPostUpdateParams(NNCIOptimizerSGDType opt){
    opt->iteration += 1;
}
