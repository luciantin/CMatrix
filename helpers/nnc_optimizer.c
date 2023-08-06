#include <malloc.h>
#include "nnc_optimizer.h"
#include <stdlib.h>
#include <math.h>

NNCIOptimizerSGDType NNCOptimizerSGDAlloc(nnc_mtype learning_rate, nnc_mtype decay, nnc_mtype momentum){
    NNCIOptimizerSGDType opt = malloc(sizeof(NNCOptimizerSGDType));
    opt->learning_rate = learning_rate;
    opt->current_learning_rate = learning_rate;
    opt->decay = decay;
    opt->iteration = 0;
    opt->momentum = momentum;
    return opt;
}

void NNCOptimizerSGDDeAlloc(NNCIOptimizerSGDType opt){
    free(opt);
}

void NNCOptimizerSGDUpdateParams(NNCIOptimizerSGDType opt, NNCIDenseLayerType layer){
    if(opt->momentum != 0){

        // FIXME 1
        if(layer->mweights == nnc_null) layer->mweights = NNCMatrixAllocBaseValue(layer->num_neurons, layer->num_inputs, 0);
        if(layer->mbiases == nnc_null) layer->mbiases = NNCMatrixAllocBaseValue(layer->num_neurons, 1, 0);

        // weights = momentum * mweights - current_learning_rate * dweights
        // biases  = momentum * mbiases  - current_learning_rate * dbiases

        NNCIMatrixType _dweights = NNCMatrixProductNumber(layer->dweights, opt->current_learning_rate);
        NNCIMatrixType _dbiases  = NNCMatrixProductNumber(layer->dbiases,  opt->current_learning_rate);

        NNCIMatrixType _mweights = NNCMatrixProductNumber(layer->dweights, opt->momentum);
        NNCIMatrixType _mbiases  = NNCMatrixProductNumber(layer->dbiases,  opt->momentum);

        NNCIMatrixType __mweights = NNCMatrixSub(_mweights, _dweights);
        NNCIMatrixType __mbiases  = NNCMatrixSub(_mbiases,  _dbiases);

        // FIXME 1
        layer->mweights = __mweights;
        layer->mbiases  = __mbiases;

        NNCIMatrixType __weights = NNCMatrixSum(layer->weights, __mweights);
        NNCIMatrixType __biases  = NNCMatrixSum(layer->biases, __mbiases);

        NNCMatrixDeAlloc(_mweights);
        NNCMatrixDeAlloc(_mbiases);
        NNCMatrixDeAlloc(_dweights);
        NNCMatrixDeAlloc(_dbiases);
        NNCMatrixDeAlloc(__mweights);
        NNCMatrixDeAlloc(__mbiases);
        NNCMatrixDeAlloc(layer->weights);
        NNCMatrixDeAlloc(layer->biases);

        layer->weights = __weights;
        layer->biases  = __biases;
    }
    else{ // without momentum
        NNCIMatrixType _weights = NNCMatrixProductNumber(layer->dweights, -opt->current_learning_rate);
        NNCIMatrixType _biases  = NNCMatrixProductNumber(layer->dbiases,  -opt->current_learning_rate);

        NNCIMatrixType __weights = NNCMatrixSum(layer->weights, _weights);
        NNCIMatrixType __biases  = NNCMatrixSum(layer->biases, _biases);

        NNCMatrixDeAlloc(_weights);
        NNCMatrixDeAlloc(_biases);
        NNCMatrixDeAlloc(layer->weights);
        NNCMatrixDeAlloc(layer->biases);

        layer->weights = __weights;
        layer->biases  = __biases;
    }
}

void NNCOptimizerSGDPreUpdateParams(NNCIOptimizerSGDType opt){
    if(opt->decay != 0){
        opt->current_learning_rate = opt->learning_rate * ( 1.0 / ( 1.0 + opt->decay * opt->iteration));
    }
}

void NNCOptimizerSGDPostUpdateParams(NNCIOptimizerSGDType opt){
    opt->iteration += 1;
}

/// ADAGRAD


NNCIOptimizerAdaGradType NNCOptimizerAdaGradAlloc(nnc_mtype learning_rate, nnc_mtype decay){
    NNCIOptimizerAdaGradType opt = malloc(sizeof(NNCOptimizerAdaGradType));
    opt->learning_rate = learning_rate;
    opt->current_learning_rate = learning_rate;
    opt->decay = decay;
    opt->iteration = 0;
    opt->epsilon = (nnc_mtype)1e-7;
    return opt;
}

void NNCOptimizerAdaGradDeAlloc(NNCIOptimizerAdaGradType opt){
    free(opt);
}

void NNCOptimizerAdaGradUpdateParams(NNCIOptimizerAdaGradType opt, NNCIDenseLayerType layer){

    // Cache
    // cweights = cweights + dweights * dweights

    // samo prvi put kad se tek inicijalizira
    if(layer->cweights == nnc_null) layer->cweights = NNCMatrixAllocBaseValue(layer->num_neurons, layer->num_inputs, 0);
    if(layer->cbiases == nnc_null) layer->cbiases = NNCMatrixAllocBaseValue(layer->num_neurons, 1, 0);

    NNCIMatrixType cweights = NNCMatrixProduct(layer->dweights, layer->dweights);
    NNCIMatrixType cbiases = NNCMatrixProduct(layer->dbiases, layer->dbiases);

    NNCIMatrixType _cweights = NNCMatrixSum(layer->cweights, cweights);
    NNCIMatrixType _cbiases = NNCMatrixSum(layer->cbiases, cbiases);

    if(layer->cweights != nnc_null) NNCMatrixDeAlloc(layer->cweights);
    if(layer->cbiases != nnc_null) NNCMatrixDeAlloc(layer->cbiases);

    NNCMatrixDeAlloc(cbiases);
    NNCMatrixDeAlloc(cweights);

    layer->cweights = _cweights;
    layer->cbiases = _cbiases;

    NNCIMatrixType _weights = NNCMatrixProductNumber(layer->dweights, -opt->current_learning_rate);
    NNCIMatrixType _biases  = NNCMatrixProductNumber(layer->dbiases,  -opt->current_learning_rate);

    NNCIMatrixType _sqrt_cweights = NNCMatrixSqrt(layer->cweights);
    NNCIMatrixType _sqrt_cbiases  = NNCMatrixSqrt(layer->cbiases);

    NNCIMatrixType _epsilon_sqrt_cweights = NNCMatrixSumNumber(_sqrt_cweights, opt->epsilon);
    NNCIMatrixType _epsilon_sqrt_cbiases  = NNCMatrixSumNumber(_sqrt_cbiases, opt->epsilon);

    NNCMatrixDeAlloc(_sqrt_cweights);
    NNCMatrixDeAlloc(_sqrt_cbiases);

    NNCIMatrixType _calc_weights = NNCMatrixQuotient(_weights, _epsilon_sqrt_cweights);
    NNCIMatrixType _calc_biases  = NNCMatrixQuotient(_biases,  _epsilon_sqrt_cbiases);

    NNCIMatrixType _sum_weights = NNCMatrixSum(_calc_weights, layer->weights);
    NNCIMatrixType _sum_biases  = NNCMatrixSum(_calc_biases,  layer->biases);

    NNCMatrixDeAlloc(_epsilon_sqrt_cweights);
    NNCMatrixDeAlloc(_epsilon_sqrt_cbiases);

    NNCMatrixDeAlloc(_weights);
    NNCMatrixDeAlloc(_biases);

    NNCMatrixDeAlloc(layer->weights);
    NNCMatrixDeAlloc(layer->biases);

    NNCMatrixDeAlloc(_calc_weights);
    NNCMatrixDeAlloc(_calc_biases);

    layer->weights = _sum_weights;
    layer->biases  = _sum_biases;
}

void NNCOptimizerAdaGradPreUpdateParams(NNCIOptimizerAdaGradType opt){
    if(opt->decay != 0){
        opt->current_learning_rate = opt->learning_rate * ( 1.0 / ( 1.0 + opt->decay * opt->iteration));
    }
}

void NNCOptimizerAdaGradPostUpdateParams(NNCIOptimizerAdaGradType opt){
    opt->iteration += 1;
}
