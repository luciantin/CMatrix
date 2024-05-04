#ifndef CMATRIX_NNC_MODEL_H
#define CMATRIX_NNC_MODEL_H

#include "nnc_config.h"
#include "nnc_matrix.h"
#include "nnc_trainer.h"

#define NNC_MODEL_SERIALIZED_START              '#'
#define NNC_MODEL_SERIALIZED_START_LEN          3
#define NNC_MODEL_SERIALIZED_END                '$'
#define NNC_MODEL_SERIALIZED_END_LEN            3
#define NNC_MODEL_SERIALIZED_ELEMENT_SEPARATOR  '|'

//////

typedef struct NNCModelSerializedType
{
    nnc_uint    len;
    char*       model;
}
NNCModelSerializedType;

#define NNCIModelSerializedType NNCModelSerializedType*

///////

typedef struct NNCModelElementSerializedType
{
    nnc_uint    len;
    char*       element;
}
NNCModelElementSerializedType;

#define NNCIModelElementSerializedType NNCModelElementSerializedType*

///////

enum NNCModelLayerElementType{
    NNCLayerType_Layer_Dense,
    NNCLayerType_Layer_Dropout,

    NNCLayerType_Activation_ReLU,
    NNCLayerType_Activation_SoftMax,

    NNCLayerType_Loss_CCEL,

    NNCLayerType_Optimizer_SGD,
    NNCLayerType_Optimizer_AdaGrad,
    NNCLayerType_Optimizer_RMSProp,
    NNCLayerType_Optimizer_Adam,
};

static const char * const NNCModelLayerElementTypeToString[] =
{
    [NNCLayerType_Layer_Dense] = "NNCLayerType_Layer_Dense",
    [NNCLayerType_Layer_Dropout] = "NNCLayerType_Layer_Dropout",
    [NNCLayerType_Activation_ReLU]  = "NNCLayerType_Activation_ReLU",
    [NNCLayerType_Activation_SoftMax]  = "NNCLayerType_Activation_SoftMax",
    [NNCLayerType_Loss_CCEL]  = "NNCLayerType_Loss_CCEL",
    [NNCLayerType_Optimizer_SGD]  = "NNCLayerType_Optimizer_SGD",
    [NNCLayerType_Optimizer_AdaGrad]  = "NNCLayerType_Optimizer_AdaGrad",
    [NNCLayerType_Optimizer_RMSProp]  = "NNCLayerType_Optimizer_RMSProp",
    [NNCLayerType_Optimizer_Adam]  = "NNCLayerType_Optimizer_Adam",
};

typedef struct NNCModelLayerType
{
    enum NNCModelLayerElementType   type;
    void*                           layer;
    char*                           tag;
}
NNCModelLayerType;

#define NNCIModelLayerType NNCModelLayerType*


NNCIModelLayerType NNCModelLayerAlloc(void* layer, enum NNCModelLayerElementType type, char* tag);
void NNCModelLayerDeAlloc(NNCIModelLayerType layer);

NNCIModelElementSerializedType NNCModelLayerSerialize(NNCIModelLayerType element);
NNCIModelLayerType NNCModelLayerDeserialize(NNCIModelElementSerializedType model_serilized);

////////
typedef struct NNCModelStatistics
{
    nnc_mtype learning_rate;
    nnc_uint sample_len;
    nnc_uint current_epoch;
    nnc_uint total_epoch;
    nnc_mtype regularization_loss;
    nnc_mtype accuracy;
    nnc_mtype mean;
}
NNCModelStatistics;

#define NNCIModelStatistics NNCModelStatistics*

typedef struct NNCModelType
{
    char*                   tag;
    NNCIModelLayerType*     layers;
    nnc_uint                layer_len;
    NNCIModelLayerType      optimizer;
}
NNCModelType;

#define NNCIModelType NNCModelType*

NNCIModelType NNCModelAlloc(char* tag);
void NNCModelDeAlloc(NNCIModelType model);
void NNCModelDeAllocAll(NNCIModelType model);

NNCIMatrixType NNCModelTrain(NNCIModelType model, NNCITrainerType trainer, NNCIMatrixType input, NNCIMatrixType target);
NNCIMatrixType NNCModelTest(NNCIModelType model, NNCIMatrixType input, NNCIMatrixType target);
NNCIMatrixType NNCModelPredict(NNCIModelType model, NNCIMatrixType input);

void NNCModelSetOptimizer(NNCIModelType model, NNCIModelLayerType optimizer);
void NNCModelLayerRemove(NNCIModelType model, const char* tag);
void NNCModelLayerAdd(NNCIModelType model, NNCIModelLayerType layer);
void NNCModelPrintLayers(NNCIModelType model);

NNCIMatrixType* NNCModelLayerForwardPass(NNCIModelType model, NNCIMatrixType input);
NNCIMatrixType  NNCModelLayerForwardStep(NNCIModelLayerType layer, NNCIMatrixType input);
NNCIMatrixType* NNCModelLayerBackwardPass(NNCIModelType model, NNCIMatrixType target, NNCIMatrixType* output_forward_lst);
NNCIMatrixType  NNCModelLayerBackwardStep(NNCIModelLayerType layer, NNCIMatrixType dvalues, NNCIMatrixType layer_output_forward);

void NNCModelOptimizerPass(NNCIModelType model);

NNCIModelSerializedType NNCModelSerialize(NNCIModelType model);
NNCIModelType NNCModelDeserialize(NNCIModelSerializedType model_serialized);


#endif //CMATRIX_NNC_MODEL_H
