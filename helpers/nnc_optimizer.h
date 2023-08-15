#ifndef CMATRIX_NNC_OPTIMIZER_H
#define CMATRIX_NNC_OPTIMIZER_H


#include "nnc_config.h"
#include "nnc_layer.h"

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

typedef struct NNCOptimizerRMSPropType
{
    nnc_mtype   learning_rate;
    nnc_mtype   current_learning_rate;
    nnc_mtype   decay;
    nnc_mtype   iteration;
    nnc_mtype   epsilon;
    nnc_mtype   rho;
}
NNCOptimizerRMSPropType;

#define NNCIOptimizerRMSPropType NNCOptimizerRMSPropType*


NNCIOptimizerRMSPropType NNCOptimizerRMSPropAlloc(nnc_mtype learning_rate, nnc_mtype decay);
void NNCOptimizerRMSPropDeAlloc(NNCIOptimizerRMSPropType opt);

void NNCOptimizerRMSPropPreUpdateParams(NNCIOptimizerRMSPropType opt);
void NNCOptimizerRMSPropUpdateParams(NNCIOptimizerRMSPropType opt, NNCIDenseLayerType layer);
void NNCOptimizerRMSPropPostUpdateParams(NNCIOptimizerRMSPropType opt);


typedef struct NNCOptimizerAdamType
{
    nnc_mtype   learning_rate;
    nnc_mtype   current_learning_rate;
    nnc_mtype   decay;
    nnc_mtype   iteration;
    nnc_mtype   epsilon;
    nnc_mtype   beta_1;
    nnc_mtype   beta_2;
}
NNCOptimizerAdamType;

#define NNCIOptimizerAdamType NNCOptimizerAdamType*


NNCIOptimizerAdamType NNCOptimizerAdamAlloc(nnc_mtype learning_rate, nnc_mtype decay,
                                            nnc_mtype epsilon, nnc_mtype beta_1, nnc_mtype beta_2 );
void NNCOptimizerAdamDeAlloc(NNCIOptimizerAdamType opt);

void NNCOptimizerAdamPreUpdateParams(NNCIOptimizerAdamType opt);
void NNCOptimizerAdamUpdateParams(NNCIOptimizerAdamType opt, NNCIDenseLayerType layer);
void NNCOptimizerAdamPostUpdateParams(NNCIOptimizerAdamType opt);




#endif //CMATRIX_NNC_OPTIMIZER_H
