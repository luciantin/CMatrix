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

    NNCLayerType_Optimizer_Sgd,
    NNCLayerType_Optimizer_AdaGrad,
    NNCLayerType_Optimizer_RmsProp,
    NNCLayerType_Optimizer_Adam,
};

typedef struct NNCModelLayerType
{
    enum NNCModelLayerElementType   type;
    void*                           layer;
    char*                           tag;
}
NNCModelLayerType;

#define NNCIModelLayerType NNCModelLayerType*


NNCIModelLayerType NNCModelElementAlloc(void* layer, enum NNCModelLayerElementType type, char* tag);
void NNCModelElementDeAlloc(NNCIModelLayerType element);

NNCIModelElementSerializedType NNCModelElementSerialize(NNCIModelLayerType element);
NNCIModelLayerType NNCModelElementDeserialize(NNCIModelElementSerializedType model_serilized);

////////

typedef struct NNCModelType
{
    char*                   tag;
    NNCIModelLayerType*     layers;
    nnc_uint                layer_len;
    nnc_uint                layer_index;    // FIXME solution without layer_index, with nullptr
}
NNCModelType;

#define NNCIModelType NNCModelType*

NNCIModelType NNCModelAlloc(char* tag, int layer_len);
void NNCModelDeAlloc(NNCIModelType model);
void NNCModelDeAllocAll(NNCIModelType model);

NNCIMatrixType NNCModelTrain(NNCIModelType model, NNCITrainerType trainer, NNCIMatrixType input, NNCIMatrixType target);
NNCIMatrixType NNCModelTest(NNCIModelType model, NNCIMatrixType input, NNCIMatrixType target);
NNCIMatrixType NNCModelPredict(NNCIModelType model, NNCIMatrixType input);

void NNCModelLayerRemove(NNCIModelType model, char* tag); // TODO
void NNCModelLayerAdd(NNCIModelType model, NNCIModelLayerType layer);

NNCIMatrixType NNCModelLayerForward(NNCIModelType model, NNCITrainerType trainer, NNCIModelLayerType layer, NNCIMatrixType input);
NNCIMatrixType NNCModelLayerBackward(NNCIModelType model, NNCITrainerType trainer, NNCIModelLayerType layer, NNCIMatrixType input);

NNCIModelSerializedType NNCModelSerialize(NNCIModelType model);
NNCIModelType NNCModelDeserialize(NNCIModelSerializedType model_serialized);


#endif //CMATRIX_NNC_MODEL_H
