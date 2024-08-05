#ifndef CMATRIX_NNC_MODEL_H
#define CMATRIX_NNC_MODEL_H

#include "nnc_config.h"
#include "nnc_matrix.h"

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
    NNCLayerType_Layer_Dense_With_Regularization,
    NNCLayerType_Layer_Dropout,

    NNCLayerType_Activation_ReLU,
    NNCLayerType_Activation_SoftMax,

    NNCLayerType_Loss_CCEL,

    NNCLayerType_Optimizer_SGD,
    NNCLayerType_Optimizer_AdaGrad,
    NNCLayerType_Optimizer_RMSProp,
    NNCLayerType_Optimizer_Adam,

    NNCLayerType_NONE
};

static const char * const NNCModelLayerElementTypeToString[] =
{
    [NNCLayerType_Layer_Dense] = "NNCLayerType_Layer_Dense",
    [NNCLayerType_Layer_Dense_With_Regularization] = "NNCLayerType_Layer_Dense_With_Regularization",
    [NNCLayerType_Layer_Dropout] = "NNCLayerType_Layer_Dropout",
    [NNCLayerType_Activation_ReLU]  = "NNCLayerType_Activation_ReLU",
    [NNCLayerType_Activation_SoftMax]  = "NNCLayerType_Activation_SoftMax",
    [NNCLayerType_Loss_CCEL]  = "NNCLayerType_Loss_CCEL",
    [NNCLayerType_Optimizer_SGD]  = "NNCLayerType_Optimizer_SGD",
    [NNCLayerType_Optimizer_AdaGrad]  = "NNCLayerType_Optimizer_AdaGrad",
    [NNCLayerType_Optimizer_RMSProp]  = "NNCLayerType_Optimizer_RMSProp",
    [NNCLayerType_Optimizer_Adam]  = "NNCLayerType_Optimizer_Adam",
    [NNCLayerType_NONE]  = "NNCLayerType_NONE",
};

enum NNCModelLayerElementType NNCModelLayerElementTypeFromString(char* string);

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
void NNCModelLayerDeAllocAll(NNCIModelLayerType layer);

NNCIModelElementSerializedType NNCModelLayerSerialize(NNCIModelLayerType element);
NNCIModelLayerType NNCModelLayerDeserialize(NNCIModelElementSerializedType model_serilized);

////////

typedef struct NNCModelLayerOutputType
{
    NNCIMatrixType              data;
    nnc_bool                    must_not_deallocate;
}
NNCModelLayerOutputType;

#define NNCIModelLayerOutputType NNCModelLayerOutputType*

NNCIModelLayerOutputType NNCModelLayerOutputAlloc(NNCIMatrixType output, nnc_bool must_not_deallocate);
void NNCIModelLayerOutputDeAlloc(NNCIModelLayerOutputType output);
void NNCIModelLayerOutputDeAllocForced(NNCIModelLayerOutputType output);

////////


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

//NNCIModelStatistics NNCModelTrain(NNCIModelType model, NNCITrainerType trainer, NNCIMatrixType input, NNCIMatrixType target);
//NNCIModelStatistics NNCModelTest(NNCIModelType model, NNCIMatrixType input, NNCIMatrixType target);
//NNCIMatrixType NNCModelPredict(NNCIModelType model, NNCIMatrixType input);

void NNCModelSetOptimizer(NNCIModelType model, NNCIModelLayerType optimizer);
void NNCModelLayerRemove(NNCIModelType model, const char* tag);
void NNCModelLayerAdd(NNCIModelType model, NNCIModelLayerType layer);
void NNCModelPrintLayers(NNCIModelType model);

NNCIModelLayerOutputType* NNCModelLayerForwardPass(NNCIModelType model, NNCIMatrixType input);
NNCIModelLayerOutputType  NNCModelLayerForwardStep(NNCIModelLayerType layer, NNCIModelLayerOutputType input);
NNCIModelLayerOutputType* NNCModelLayerBackwardPass(NNCIModelType model, NNCIMatrixType target, NNCIModelLayerOutputType* output_forward_lst);
NNCIModelLayerOutputType NNCModelLayerBackwardStep(NNCIModelLayerType layer, NNCIModelLayerOutputType layer_output_previous, NNCIModelLayerOutputType layer_output_forward);

void NNCModelOptimizerPass(NNCIModelType model);



NNCIModelSerializedType NNCModelSerialize(NNCIModelType model);
NNCIModelType NNCModelDeserialize(NNCIModelSerializedType model_serialized);


#endif //CMATRIX_NNC_MODEL_H
