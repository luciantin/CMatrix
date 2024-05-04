#ifndef CMATRIX_NNC_TRAINER_H
#define CMATRIX_NNC_TRAINER_H


#include "nnc_config.h"
#include "nnc_model.h"
#include "nnc_statistics.h"


enum NNCTrainerTypeStrategy{
    NNCTrainerTypeStrategy_Iterative,
};

typedef struct NNCTrainerType
{
    char*                       tag;
    nnc_uint                    max_epoch;
    nnc_uint                    current_epoch;
    enum NNCTrainerTypeStrategy strategy;
//    NNCIMatrixType* (*LayerForwardPass)  (NNCIModelType model, NNCIMatrixType input);
//    NNCIMatrixType* (*LayerBackwardPass) (NNCIModelType model, NNCIMatrixType target, NNCIMatrixType* output_forward_lst);
//    void (*OptimizerPass)(NNCIModelType model);
}
NNCTrainerType;

#define NNCITrainerType NNCTrainerType*

NNCITrainerType NNCTrainerAlloc(char* tag, enum NNCTrainerTypeStrategy strategy, nnc_uint max_epoch);
void NNCTrainerDeAlloc(NNCITrainerType trainer);

//void NNCTrainerLinkModel(NNCITrainerType trainer, NNCIModelType model);

NNCIModelStatistics NNCTrainerTrain(NNCITrainerType trainer, NNCIModelType model, NNCIMatrixType input, NNCIMatrixType target);
NNCIModelStatistics NNCTrainerTest(NNCITrainerType trainer, NNCIModelType model, NNCIMatrixType input, NNCIMatrixType target);
NNCIMatrixType NNCTrainerPredict(NNCITrainerType trainer, NNCIModelType model, NNCIMatrixType input);

#endif //CMATRIX_NNC_TRAINER_H
