#ifndef CMATRIX_NNC_MODEL_H
#define CMATRIX_NNC_MODEL_H

#include <stdbool.h>
#include "nnc_config.h"
#include "nnc_matrix.h"

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

enum NNCModelElementLayerType{
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

typedef struct NNCModelElementType
{
    enum NNCModelElementLayerType   type;
    void*                           layer;
    char*                           tag;
}
NNCModelElementType;

#define NNCIModelElementType NNCModelElementType*


NNCIModelElementType NNCModelElementAlloc(void* layer, enum NNCModelElementLayerType type, char* tag);
void NNCModelElementDeAlloc(NNCIModelElementType element);

NNCIModelElementSerializedType NNCModelElementSerialize(NNCIModelElementType element);
NNCIModelElementType NNCModelElementDeserialize(NNCIModelElementSerializedType model_serilized);

////////

typedef struct NNCModelType
{
    char*                   tag;
    NNCIModelElementType*   layers;
    nnc_uint                layer_len;
}
NNCModelType;

#define NNCIModelType NNCModelType*

NNCIModelType NNCModelAlloc(char* tag);
void NNCModelDeAlloc(NNCIModelType model);
void NNCModelDeAllocAll(NNCIModelType model);

NNCIMatrixType NNCModelTrain(NNCIMatrixType input, NNCIMatrixType target);
NNCIMatrixType NNCModelTest(NNCIMatrixType input, NNCIMatrixType target);
NNCIMatrixType NNCModelPredict(NNCIMatrixType input);

NNCIMatrixType NNCModelLayerRemove(NNCIModelType model, char* tag);
NNCIMatrixType NNCModelLayerAdd(NNCIModelType model, void* layer, enum NNCModelElementLayerType type, char* tag);

NNCIModelSerializedType NNCModelSave(NNCIModelType model);
NNCIModelType NNCModelDeserialize(NNCIModelSerializedType model_serialized);


#endif //CMATRIX_NNC_MODEL_H
