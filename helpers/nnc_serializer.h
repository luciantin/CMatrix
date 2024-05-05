#ifndef CMATRIX_NNC_SERIALIZER_H
#define CMATRIX_NNC_SERIALIZER_H


#include "nnc_model.h"
#include "nnc_trainer.h"
#include "nnc_list.h"

typedef struct NNCSerializedLayerType
{
    Node*                               layer;
    nnc_uint                            len;
    enum NNCModelLayerElementType       type;
}
NNCSerializedLayerType;

#define NNCISerializedLayerType NNCSerializedLayerType*

NNCISerializedLayerType NNCSerializedLayerTypeAlloc(Node* layer, nnc_uint len, enum NNCModelLayerElementType type);
void NNCSerializedLayerTypeDeAlloc(NNCISerializedLayerType layer);

NNCISerializedLayerType NNCSerializedLayerTypeSerializeLayer(NNCIModelLayerType layer);
NNCIModelLayerType NNCSerializedLayerTypeDeSerializeLayer(NNCISerializedLayerType layer);

enum NNCSerializerType{
    NNCSerializer_Model,
    NNCSerializer_ModelWithTrainer,
    NNCSerializer_TrainedModel
};

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


typedef struct NNCSerializedModelType
{
    enum NNCSerializerType              type;
    Node*                               model;
    nnc_uint                            model_len;
    Node*                               trainer;
    nnc_uint                            trainer_len;
    NNCISerializedLayerType*            layers;
    nnc_uint                            layers_len;
    NNCISerializedLayerType             optimizer;
}
NNCSerializedModelType;

#define NNCISerializedModelType NNCSerializedModelType*

NNCISerializedModelType NNCSerializedModelAlloc(enum NNCSerializerType type, Node* model, nnc_uint model_len, Node* trainer, nnc_uint trainer_len, NNCISerializedLayerType* layers, nnc_uint layers_len, NNCISerializedLayerType optimizer);
void NNCSerializedModelDeAlloc(NNCISerializedModelType model);

void NNCSerializedModelAddLayer(NNCISerializedModelType model, NNCISerializedLayerType layer);

void NNCSerializedModelSaveToFile(NNCISerializedModelType model, char* file_name);
NNCISerializedModelType NNCSerializedModelLoadFromFile(char* file_name);

NNCISerializedModelType   NNCDeSerializedModelSerialize(NNCIDeSerializedModelType model);
NNCIDeSerializedModelType NNCSerializedModelDeSerialize(NNCISerializedModelType model);

// file structure :
// lenX;lenY;val1;val2;...;valN;
NNCIMatrixType NNCImportMatrixFromFile(char* fileName);



#endif //CMATRIX_NNC_SERIALIZER_H
