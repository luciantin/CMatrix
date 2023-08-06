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
    nnc_mtype   momentum;
}
NNCOptimizerSGDType;

#define NNCIOptimizerSGDType NNCOptimizerSGDType*


NNCIOptimizerSGDType NNCOptimizerSGDAlloc(nnc_mtype learning_rate, nnc_mtype decay, nnc_mtype momentum);
void NNCOptimizerSGDDeAlloc(NNCIOptimizerSGDType opt);

void NNCOptimizerSGDPreUpdateParams(NNCIOptimizerSGDType opt);
void NNCOptimizerSGDUpdateParams(NNCIOptimizerSGDType opt, NNCIDenseLayerType layer);
void NNCOptimizerSGDPostUpdateParams(NNCIOptimizerSGDType opt);



typedef struct NNCOptimizerAdaGradType
{
    nnc_mtype   learning_rate;
    nnc_mtype   current_learning_rate;
    nnc_mtype   decay;
    nnc_mtype   iteration;
    nnc_mtype   epsilon;
}
NNCOptimizerAdaGradType;

#define NNCIOptimizerAdaGradType NNCOptimizerAdaGradType*


NNCIOptimizerAdaGradType NNCOptimizerAdaGradAlloc(nnc_mtype learning_rate, nnc_mtype decay);
void NNCOptimizerAdaGradDeAlloc(NNCIOptimizerAdaGradType opt);

void NNCOptimizerAdaGradPreUpdateParams(NNCIOptimizerAdaGradType opt);
void NNCOptimizerAdaGradUpdateParams(NNCIOptimizerAdaGradType opt, NNCIDenseLayerType layer);
void NNCOptimizerAdaGradPostUpdateParams(NNCIOptimizerAdaGradType opt);





#endif //CMATRIX_NNC_OPTIMIZER_H
