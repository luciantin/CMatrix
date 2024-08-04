#ifndef CMATRIX_NNC_SERIALIZER_H
#define CMATRIX_NNC_SERIALIZER_H


#include "nnc_model.h"
#include "nnc_trainer.h"
#include "nnc_list.h"

#define NNC_LAYER_START_BYTE ':'
#define NNC_LAYER_END_BYTE '#'
#define NNC_VALUE_END_BYTE ';'
#define NNC_VALUE_NOT_DEFINED '-'
#define NNC_VALUE_EMPTY_BYTE '*'

enum NNCSerializerType{
    NNCSerializer_Model,
    NNCSerializer_ModelWithTrainer,
    NNCSerializer_TrainedModel,
    NNCSerializer_NONE
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////


typedef struct NNCSerializedLayerType
{
    NNCIList                            layer;
    nnc_uint                            len;
    enum NNCModelLayerElementType       type;
}
NNCSerializedLayerType;

#define NNCISerializedLayerType NNCSerializedLayerType*

NNCISerializedLayerType NNCSerializedLayerTypeAlloc(NNCIList layer, nnc_uint len, enum NNCModelLayerElementType type);
void NNCSerializedLayerTypeDeAlloc(NNCISerializedLayerType layer);

NNCISerializedLayerType NNCSerializedLayerTypeSerializeLayer(enum NNCSerializerType type, NNCIModelLayerType layer);
NNCIModelLayerType NNCSerializedLayerTypeDeSerializeLayer(NNCISerializedLayerType layer);

NNCIList NNCSerializedLayerTypeToList(NNCISerializedLayerType layer);
NNCISerializedLayerType NNCSerializedLayerTypeFromList_Destructive(NNCIList* list);
////////////////////////////////////////////////////////////////////////////////////////////////////////////



static const nnc_uint NNCSerializerTypeToStringLen = 3;
static const char * const NNCSerializerTypeToString[] =
{
    [NNCSerializer_Model] = "NNCSerializer_Model",
    [NNCSerializer_ModelWithTrainer] = "NNCSerializer_ModelWithTrainer",
    [NNCSerializer_TrainedModel] = "NNCSerializer_TrainedModel",
};


typedef struct NNCDeSerializedModelType
{
    enum NNCSerializerType              type;
    NNCIModelType                       model;
    NNCITrainerType                     trainer;
}
NNCDeSerializedModelType;

#define NNCIDeSerializedModelType NNCDeSerializedModelType*

NNCIDeSerializedModelType NNCDeSerializedModelAlloc(enum NNCSerializerType type, NNCIModelType model, NNCITrainerType trainer);
void NNCDeSerializedModelDeAlloc(NNCIDeSerializedModelType model);


////////////////////////////////////////////////////////////////////////////////////////////////////////////


typedef struct NNCSerializedModelType
{
    enum NNCSerializerType              type;
    NNCIList                            tag;
    NNCISerializedLayerType             trainer;
    NNCISerializedLayerType             optimizer;
    NNCISerializedLayerType*            layers;
    nnc_uint                            layers_len;
}
NNCSerializedModelType;

#define NNCISerializedModelType NNCSerializedModelType*

NNCISerializedModelType NNCSerializedModelAlloc(enum NNCSerializerType type, NNCIList tag, NNCISerializedLayerType trainer, NNCISerializedLayerType optimizer, NNCISerializedLayerType* layers, nnc_uint layers_len);
void NNCSerializedModelDeAlloc(NNCISerializedModelType model);

void NNCSerializedModelAddLayer(NNCISerializedModelType model, NNCISerializedLayerType layer);

NNCIList NNCSerializedModelMinify(NNCISerializedModelType model);
NNCISerializedModelType NNCSerializedModelMaxify_Destructive(NNCIList list);

void NNCSerializedModelSaveToFile(NNCISerializedModelType model, char* file_name);
NNCISerializedModelType NNCSerializedModelLoadFromFile(char* file_name);

NNCISerializedModelType   NNCDeSerializedModelSerialize(enum NNCSerializerType type, NNCIDeSerializedModelType model);
NNCIDeSerializedModelType NNCSerializedModelDeSerialize(NNCISerializedModelType model);


////////////////////////////////////////////////////////////////////////////////////////////////////////////


// file structure :
// lenX;lenY;val1;val2;...;valN;
NNCIMatrixType NNCImportMatrixFromFile(char* fileName);

NNCISerializedModelType NNCSerialize(enum NNCSerializerType type, NNCIModelType model, NNCITrainerType trainer);

typedef struct NNCSerializedMatrixType
{
    char*       matrix;
    nnc_uint    len;
}
        NNCSerializedMatrixType;

#define NNCISerializedMatrixType NNCSerializedMatrixType*

void NNCISerializedMatrixDeAlloc(NNCISerializedMatrixType smatrix);
NNCISerializedMatrixType NNCIMatrixSerialize(NNCIMatrixType matrix);



#endif //CMATRIX_NNC_SERIALIZER_H
