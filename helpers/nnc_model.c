#include <malloc.h>
#include "nnc_model.h"

NNCIModelType NNCModelAlloc(char* tag, int layer_len){
    NNCIModelType model = malloc(sizeof(NNCModelType));
    model->layer_len = layer_len;
    model->layers = malloc(sizeof(NNCModelLayerType) * layer_len);
    for(int _x = 0; _x < layer_len; _x ++) model->layers[_x] = nnc_null;
    model->tag = tag;
    model->layer_index = 0;
    return model;
}

void NNCModelDeAlloc(NNCIModelType model){
    free(model->layers);
    free(model->tag);
    free(model);
}

void NNCModelDeAllocAll(NNCIModelType model){
    // TODO free layers
    free(model->layers);
    free(model->tag);
    free(model);
}

void NNCModelLayerAdd(NNCIModelType model, NNCIModelLayerType layer){
    if(model->layer_index >= model->layer_len) return;
    else model->layers[model->layer_index] = layer;
    model->layer_index += 1;
}

NNCIMatrixType NNCModelTrain(NNCIModelType model, NNCITrainerType trainer, NNCIMatrixType input, NNCIMatrixType target){
    NNCIMatrixType output = NNCMatrixAllocBaseValue(target->x, 0, 0);

    for(nnc_uint layer_index = 0; layer_index <= model->layer_len; layer_index ++){
        NNCIMatrixType _output_forward =
    }

    return output;
}

NNCIMatrixType NNCModelLayerForward(NNCIModelType model, NNCIModelLayerType layer, NNCITrainerType trainer, NNCIMatrixType input){

}

NNCIMatrixType NNCModelTest(NNCIModelType model, NNCIMatrixType input, NNCIMatrixType target);
NNCIMatrixType NNCModelPredict(NNCIModelType model, NNCIMatrixType input);
