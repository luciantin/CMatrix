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

// RMSProp


NNCIOptimizerRMSPropType NNCOptimizerRMSPropAlloc(nnc_mtype learning_rate, nnc_mtype decay){
    NNCIOptimizerRMSPropType opt = malloc(sizeof(NNCOptimizerRMSPropType));
    opt->learning_rate = learning_rate;
    opt->current_learning_rate = learning_rate;
    opt->decay = decay;
    opt->iteration = 0;
    opt->rho = (nnc_mtype)0.9;
    opt->epsilon = (nnc_mtype)1e-7;
    return opt;
}

void NNCOptimizerRMSPropDeAlloc(NNCIOptimizerRMSPropType opt){
    free(opt);
}

void NNCOptimizerRMSPropUpdateParams(NNCIOptimizerRMSPropType opt, NNCIDenseLayerType layer){

    // Cache
    // cweights = cweights + dweights * dweights

    // samo prvi put kad se tek inicijalizira
    if(layer->cweights == nnc_null) layer->cweights = NNCMatrixAllocBaseValue(layer->num_neurons, layer->num_inputs, 0);
    if(layer->cbiases == nnc_null) layer->cbiases = NNCMatrixAllocBaseValue(layer->num_neurons, 1, 0);

    NNCIMatrixType cweights = NNCMatrixProduct(layer->dweights, layer->dweights);
    NNCIMatrixType cbiases = NNCMatrixProduct(layer->dbiases, layer->dbiases);

    NNCIMatrixType _rho_cweights = NNCMatrixProductNumber(cweights, 1 - opt->rho);
    NNCIMatrixType _rho_cbiases = NNCMatrixProductNumber(cbiases, 1 - opt->rho);

    NNCIMatrixType _rho__cweights = NNCMatrixProductNumber(layer->cweights, opt->rho);
    NNCIMatrixType _rho__cbiases = NNCMatrixProductNumber(layer->cbiases, opt->rho);

    NNCIMatrixType _cweights = NNCMatrixSum(_rho__cweights, _rho_cweights);
    NNCIMatrixType _cbiases = NNCMatrixSum(_rho__cbiases, _rho_cbiases);

    if(layer->cweights != nnc_null) NNCMatrixDeAlloc(layer->cweights);
    if(layer->cbiases != nnc_null) NNCMatrixDeAlloc(layer->cbiases);

    NNCMatrixDeAlloc(cbiases);
    NNCMatrixDeAlloc(cweights);

    NNCMatrixDeAlloc(_rho_cweights);
    NNCMatrixDeAlloc(_rho_cbiases);

    NNCMatrixDeAlloc(_rho__cweights);
    NNCMatrixDeAlloc(_rho__cbiases);

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

void NNCOptimizerRMSPropPreUpdateParams(NNCIOptimizerRMSPropType opt){
    if(opt->decay != 0){
        opt->current_learning_rate = opt->learning_rate * ( 1.0 / ( 1.0 + opt->decay * opt->iteration));
    }
}

void NNCOptimizerRMSPropPostUpdateParams(NNCIOptimizerRMSPropType opt){
    opt->iteration += 1;
}


// Adam

NNCIOptimizerAdamType NNCOptimizerAdamAlloc(nnc_mtype learning_rate, nnc_mtype decay,
                                            nnc_mtype epsilon, nnc_mtype beta_1, nnc_mtype beta_2 ){
    NNCIOptimizerAdamType opt = malloc(sizeof(NNCOptimizerAdamType));
    opt->learning_rate = learning_rate;
    opt->current_learning_rate = learning_rate;
    opt->decay = decay;
    opt->iteration = 0;
    opt->epsilon = (nnc_mtype)epsilon;
    opt->beta_1 = (nnc_mtype)beta_1;
    opt->beta_2 = (nnc_mtype)beta_2;
    return opt;
}

void NNCOptimizerAdamDeAlloc(NNCIOptimizerAdamType opt){
    free(opt);
}

void NNCOptimizerAdamUpdateParams(NNCIOptimizerAdamType opt, NNCIDenseLayerType layer){

    // Cache
    // cweights = cweights + dweights * dweights

    // samo prvi put kad se tek inicijalizira
    if(layer->cweights == nnc_null) layer->cweights = NNCMatrixAllocBaseValue(layer->num_neurons, layer->num_inputs, 0);
    if(layer->cbiases == nnc_null) layer->cbiases = NNCMatrixAllocBaseValue(layer->num_neurons, 1, 0);
    if(layer->mweights == nnc_null) layer->mweights = NNCMatrixAllocBaseValue(layer->num_neurons, layer->num_inputs, 0);
    if(layer->mbiases == nnc_null) layer->mbiases = NNCMatrixAllocBaseValue(layer->num_neurons, 1, 0);
    ///////////////////////////////////////////////

    NNCIMatrixType _beta_1_mweights = NNCMatrixProductNumber(layer->mweights, opt->beta_1);
    NNCIMatrixType _beta_1_mbiases = NNCMatrixProductNumber(layer->mbiases, opt->beta_1);

    NNCIMatrixType _beta_1__mweights = NNCMatrixProductNumber(layer->dweights, 1- opt->beta_1);
    NNCIMatrixType _beta_1__mbiases = NNCMatrixProductNumber(layer->dbiases, 1- opt->beta_1);

    NNCIMatrixType _mweights = NNCMatrixSum(_beta_1__mweights, _beta_1_mweights);
    NNCIMatrixType _mbiases = NNCMatrixSum(_beta_1__mbiases, _beta_1_mbiases);

    if(layer->mweights != nnc_null) NNCMatrixDeAlloc(layer->mweights);
    if(layer->mbiases != nnc_null) NNCMatrixDeAlloc(layer->mbiases);

    NNCMatrixDeAlloc(_beta_1_mweights);
    NNCMatrixDeAlloc(_beta_1_mbiases);

    NNCMatrixDeAlloc(_beta_1__mweights);
    NNCMatrixDeAlloc(_beta_1__mbiases);

    layer->mweights = _mweights;
    layer->mbiases = _mbiases;

    ///////////////////////////////////////////////

    NNCIMatrixType dweights = NNCMatrixProduct(layer->dweights, layer->dweights);
    NNCIMatrixType dbiases = NNCMatrixProduct(layer->dbiases, layer->dbiases);

    NNCIMatrixType _beta_2_cweights = NNCMatrixProductNumber(layer->cweights, opt->beta_2);
    NNCIMatrixType _beta_2_cbiases = NNCMatrixProductNumber(layer->cbiases, opt->beta_2);

    NNCIMatrixType _beta_2__cweights = NNCMatrixProductNumber(dweights, 1- opt->beta_2);
    NNCIMatrixType _beta_2__cbiases = NNCMatrixProductNumber(dbiases, 1 - opt->beta_2);

    NNCIMatrixType _cweights = NNCMatrixSum(_beta_2__cweights, _beta_2_cweights);
    NNCIMatrixType _cbiases = NNCMatrixSum(_beta_2__cbiases, _beta_2_cbiases);

    if(layer->cweights != nnc_null) NNCMatrixDeAlloc(layer->cweights);
    if(layer->cbiases != nnc_null) NNCMatrixDeAlloc(layer->cbiases);

    NNCMatrixDeAlloc(_beta_2_cweights);
    NNCMatrixDeAlloc(_beta_2_cbiases);

    NNCMatrixDeAlloc(_beta_2__cweights);
    NNCMatrixDeAlloc(_beta_2__cbiases);

    NNCMatrixDeAlloc(dweights);
    NNCMatrixDeAlloc(dbiases);

    layer->cweights = _cweights;
    layer->cbiases = _cbiases;

//    NNCMatrixPrint(layer->cweights);
//    puts("--------");
//    NNCMatrixPrint(layer->cbiases);
//    puts("--------");
    ///////////////////////////////////////

    nnc_mtype b1 = 1 - powf(opt->beta_1, opt->iteration + 1);
    NNCIMatrixType _mweight_corrected = NNCMatrixQuotientNumber(layer->mweights, b1);
    NNCIMatrixType _mbias_corrected = NNCMatrixQuotientNumber(layer->mbiases, b1);

//    NNCMatrixPrint(_mweight_corrected);
//    puts("--------");
//    NNCMatrixPrint(_mbias_corrected);
//    puts("--------");

    nnc_mtype b2 = 1 - powf( opt->beta_2, opt->iteration + 1);
    NNCIMatrixType _cweight_corrected = NNCMatrixQuotientNumber(layer->cweights, b2);
    NNCIMatrixType _cbias_corrected = NNCMatrixQuotientNumber(layer->cbiases, b2);

    ///////////////////////////////////////

    NNCIMatrixType _weights = NNCMatrixProductNumber(_mweight_corrected, -opt->current_learning_rate);
    NNCIMatrixType _biases  = NNCMatrixProductNumber(_mbias_corrected,  -opt->current_learning_rate);

    NNCIMatrixType _sqrt_cweights = NNCMatrixSqrt(_cweight_corrected);
    NNCIMatrixType _sqrt_cbiases  = NNCMatrixSqrt(_cbias_corrected);

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

    NNCMatrixDeAlloc(_mweight_corrected);
    NNCMatrixDeAlloc(_mbias_corrected);

    NNCMatrixDeAlloc(_cweight_corrected);
    NNCMatrixDeAlloc(_cbias_corrected);

    layer->weights = _sum_weights;
    layer->biases  = _sum_biases;
}

void NNCOptimizerAdamPreUpdateParams(NNCIOptimizerAdamType opt){
    if(opt->decay != 0){
        opt->current_learning_rate = opt->learning_rate * ( 1.0 / ( 1.0 + opt->decay * opt->iteration));
    }
}

void NNCOptimizerAdamPostUpdateParams(NNCIOptimizerAdamType opt){
    opt->iteration += 1;
}