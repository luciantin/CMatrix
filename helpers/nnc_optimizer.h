#ifndef CMATRIX_NNC_OPTIMIZER_H
#define CMATRIX_NNC_OPTIMIZER_H


#include "nnc_config.h"
#include "nnc_dense_layer.h"

typedef struct NNCOptimizerSGDType
{
    nnc_mtype   learning_rate;
}
NNCOptimizerSGDType;

#define NNCIOptimizerSGDType NNCOptimizerSGDType*


NNCIOptimizerSGDType NNCOptimizerSGDAlloc(nnc_mtype learning_rate);
void NNCOptimizerSGDDeAlloc(NNCIOptimizerSGDType opt);
void NNCOptimizerSGDUpdateParams(NNCIOptimizerSGDType opt, NNCIDenseLayerType layer);

#endif //CMATRIX_NNC_OPTIMIZER_H
