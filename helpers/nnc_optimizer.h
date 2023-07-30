#ifndef CMATRIX_NNC_OPTIMIZER_H
#define CMATRIX_NNC_OPTIMIZER_H


#include "nnc_config.h"
#include "nnc_dense_layer.h"

typedef struct NNCOptimizerSGDType
{
    nnc_mtype   learning_rate;
    nnc_mtype   current_learning_rate;
    nnc_mtype   decay;
    nnc_mtype   iteration;
}
NNCOptimizerSGDType;

#define NNCIOptimizerSGDType NNCOptimizerSGDType*


NNCIOptimizerSGDType NNCOptimizerSGDAlloc(nnc_mtype learning_rate, nnc_mtype decay);
void NNCOptimizerSGDDeAlloc(NNCIOptimizerSGDType opt);

void NNCOptimizerSGDPreUpdateParams(NNCIOptimizerSGDType opt);
void NNCOptimizerSGDUpdateParams(NNCIOptimizerSGDType opt, NNCIDenseLayerType layer);
void NNCOptimizerSGDPostUpdateParams(NNCIOptimizerSGDType opt);

#endif //CMATRIX_NNC_OPTIMIZER_H
