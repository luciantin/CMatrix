#include <malloc.h>
#include "nnc_model.h"
#include "nnc_layer.h"
#include "nnc_activation_layer.h"
#include "nnc_loss_function.h"
#include "nnc_optimizer.h"
#include "nnc_vector.h"

NNCIModelType NNCModelAlloc(char* tag){
    NNCIModelType model = malloc(sizeof(NNCModelType));
    model->layer_len = 0;
    model->layers = nnc_null;
    model->optimizer = nnc_null;
    model->tag = tag;
    return model;
}

void NNCModelDeAlloc(NNCIModelType model){
    for(nnc_uint layer_index = 0; layer_index < model->layer_len; layer_index ++) NNCModelLayerDeAlloc(model->layers[layer_index]);
    free(model->layers);
    free(model->tag);
    free(model);
}

void NNCModelDeAllocAll(NNCIModelType model){
    for(nnc_uint layer_index = 0; layer_index < model->layer_len; layer_index ++) NNCModelLayerDeAllocAll(model->layers[layer_index]);
    free(model->layers);
    free(model->tag);
    free(model);
}

void NNCModelSetOptimizer(NNCModelType *model, NNCIModelLayerType optimizer) {
    model->optimizer = optimizer;
}

void NNCModelLayerAdd(NNCIModelType model, NNCIModelLayerType layer){

    NNCIModelLayerType* tmp_layers = malloc(sizeof(NNCIModelLayerType) * (model->layer_len + 1));

    if(model->layers != nnc_null) for(int _x = 0; _x < model->layer_len; _x ++) tmp_layers[_x] = model->layers[_x];
    tmp_layers[model->layer_len] = layer; // layer_len starts from 1, [layer_len] is 1 past last element

    if(model->layers != nnc_null) free(model->layers);
    model->layers = tmp_layers;
    model->layer_len += 1;
}

void NNCModelLayerRemove(NNCModelType *model, const char *tag) {
    if(model->layer_len == 0) return;
    for(nnc_uint layer_index = 0; layer_index < model->layer_len; layer_index ++){
        if(model->layers[layer_index]->tag == tag){

            if(model->layer_len - 1 == 0){
                free(model->layers);
                return;
            }

            NNCIModelLayerType* tmp_layers = malloc(sizeof(NNCModelLayerType) * model->layer_len - 1);

            for(int _x = 0; _x < model->layer_len; _x ++) {
                if(_x < layer_index) tmp_layers[_x] = model->layers[_x];
                else if(_x > layer_index) tmp_layers[_x-1] = model->layers[_x];
            }

            free(model->layers);
            model->layers = tmp_layers;
            model->layer_len -= 1;
        }
    }
}

NNCIModelLayerOutputType* NNCModelLayerForwardPass(NNCIModelType model, NNCIMatrixType input){
    NNCIModelLayerOutputType* _output_forward_lst = malloc(sizeof(NNCIMatrixType) * model->layer_len);
    NNCIModelLayerOutputType _output_forward = NNCModelLayerOutputAlloc(input, nnc_true);;

    for(int layer_index = 0; layer_index < model->layer_len; layer_index ++){
        dprintf("%d. - %s - ", layer_index + 1, model->tag);
        _output_forward = NNCModelLayerForwardStep(model->layers[layer_index], _output_forward);
        _output_forward_lst[layer_index] = _output_forward;
    }
    return _output_forward_lst;
}

NNCIModelLayerOutputType NNCModelLayerForwardStep(NNCIModelLayerType layer, NNCIModelLayerOutputType input){
    dprintf("forward - %s - %s\n", layer->tag, NNCModelLayerElementTypeToString[layer->type]);

    if(layer->type == NNCLayerType_Layer_Dense || layer->type == NNCLayerType_Layer_Dense_With_Regularization){
        return NNCModelLayerOutputAlloc(NNCDenseLayerForward(input->data, layer->layer), nnc_false);
    }
    else if(layer->type == NNCLayerType_Activation_ReLU){
        return NNCModelLayerOutputAlloc(NNCActivationReLUForward(input->data), nnc_false);
    }
    else if(layer->type == NNCLayerType_Layer_Dropout){
        return NNCModelLayerOutputAlloc(NNCDropoutLayerForward(input->data, layer->layer), nnc_false);
    }
    else if(layer->type == NNCLayerType_Activation_SoftMax){
        return NNCModelLayerOutputAlloc(NNCActivationSoftMaxForward(input->data), nnc_false);
    }
    else return nnc_null;
}

NNCIModelLayerOutputType* NNCModelLayerBackwardPass(NNCIModelType model, NNCIMatrixType target, NNCIModelLayerOutputType* output_forward_lst){
    NNCIModelLayerOutputType* _output_backward_lst = malloc(sizeof(NNCIMatrixType) * model->layer_len);
    NNCIModelLayerOutputType _output_backward = NNCModelLayerOutputAlloc(target, nnc_true);

    for(int layer_index = model->layer_len - 1; layer_index >= 0; layer_index -= 1){
        dprintf("%d. - %s - backward - %s - %s\n", layer_index + 1, model->tag, model->layers[layer_index]->tag, NNCModelLayerElementTypeToString[model->layers[layer_index]->type]);

        _output_backward = NNCModelLayerBackwardStep(model->layers[layer_index], _output_backward, output_forward_lst[layer_index]);
        _output_backward_lst[layer_index] = _output_backward;
    }

    return _output_backward_lst;
}

NNCIModelLayerOutputType NNCModelLayerBackwardStep(NNCIModelLayerType layer, NNCIModelLayerOutputType layer_output_previous, NNCIModelLayerOutputType layer_output_forward){
    if(layer->type == NNCLayerType_Layer_Dense){
        NNCDenseLayerBackward(layer_output_previous->data, layer->layer);
        return NNCModelLayerOutputAlloc(((NNCIDenseLayerType)(layer->layer))->dinputs, nnc_true);
    }
    if(layer->type == NNCLayerType_Layer_Dense_With_Regularization){
        NNCDenseLayerWithRegularizationBackward(layer_output_previous->data, layer->layer);
        return NNCModelLayerOutputAlloc(((NNCIDenseLayerType)(layer->layer))->dinputs, nnc_true);
    }
    else if(layer->type == NNCLayerType_Activation_ReLU){
        return NNCModelLayerOutputAlloc(NNCActivationReLUBackward(layer_output_forward->data, layer_output_previous->data), nnc_false);
    }
    else if(layer->type == NNCLayerType_Layer_Dropout){
        NNCDropoutLayerBackward(layer_output_previous->data, layer->layer);
        return NNCModelLayerOutputAlloc(((NNCIDropoutLayerType)(layer->layer))->dinputs, nnc_true);
    }
    else if(layer->type == NNCLayerType_Activation_SoftMax){
        return NNCModelLayerOutputAlloc(NNCActivationSoftMaxLossCCELBackward(layer_output_forward->data, layer_output_previous->data), nnc_false);
    }
    else return nnc_null;
}

void NNCModelOptimizerPass(NNCIModelType model) {
    dprintf("x. - %s - optimizer - %s - %s\n", model->tag, model->optimizer->tag, NNCModelLayerElementTypeToString[model->optimizer->type]);

    if(model->optimizer->type == NNCLayerType_Optimizer_Adam){
        NNCOptimizerAdamPreUpdateParams(model->optimizer->layer);
        for(int layer_index = 0; layer_index < model->layer_len; layer_index ++){
            if(model->layers[layer_index]->type == NNCLayerType_Layer_Dense || model->layers[layer_index]->type == NNCLayerType_Layer_Dense_With_Regularization) NNCOptimizerAdamUpdateParams(model->optimizer->layer, model->layers[layer_index]->layer);
        }
        NNCOptimizerAdamPostUpdateParams(model->optimizer->layer);
    }
    else if(model->optimizer->type == NNCLayerType_Optimizer_AdaGrad){
        NNCOptimizerAdaGradPreUpdateParams(model->optimizer->layer);
        for(int layer_index = 0; layer_index < model->layer_len; layer_index ++){
            if(model->layers[layer_index]->type == NNCLayerType_Layer_Dense || model->layers[layer_index]->type == NNCLayerType_Layer_Dense_With_Regularization) NNCOptimizerAdaGradUpdateParams(model->optimizer->layer, model->layers[layer_index]->layer);
        }
        NNCOptimizerAdaGradPostUpdateParams(model->optimizer->layer);
    }
    else if(model->optimizer->type == NNCLayerType_Optimizer_RMSProp){
        NNCOptimizerRMSPropPreUpdateParams(model->optimizer->layer);
        for(int layer_index = 0; layer_index < model->layer_len; layer_index ++){
            if(model->layers[layer_index]->type == NNCLayerType_Layer_Dense || model->layers[layer_index]->type == NNCLayerType_Layer_Dense_With_Regularization) NNCOptimizerRMSPropUpdateParams(model->optimizer->layer, model->layers[layer_index]->layer);
        }
        NNCOptimizerRMSPropPostUpdateParams(model->optimizer->layer);
    }
    else if(model->optimizer->type == NNCLayerType_Optimizer_SGD){
        NNCOptimizerSGDPreUpdateParams(model->optimizer->layer);
        for(int layer_index = 0; layer_index < model->layer_len; layer_index ++){
            if(model->layers[layer_index]->type == NNCLayerType_Layer_Dense || model->layers[layer_index]->type == NNCLayerType_Layer_Dense_With_Regularization) NNCOptimizerSGDUpdateParams(model->optimizer->layer, model->layers[layer_index]->layer);
        }
        NNCOptimizerSGDPostUpdateParams(model->optimizer->layer);
    }
}

NNCIModelLayerType NNCModelLayerAlloc(void *layer, enum NNCModelLayerElementType type, char *tag) {
    NNCIModelLayerType modelLayer = malloc(sizeof(NNCModelLayerType));
    modelLayer->layer = layer;
    modelLayer->tag = tag;
    modelLayer->type = type;
    return modelLayer;
}

void NNCModelLayerDeAlloc(NNCIModelLayerType layer) {
    free(layer);
}

void NNCModelLayerDeAllocAll(NNCIModelLayerType layer) {

    if(layer->type == NNCLayerType_Optimizer_Adam) NNCOptimizerAdamDeAlloc(layer->layer);
    else if(layer->type == NNCLayerType_Optimizer_AdaGrad) NNCOptimizerAdaGradDeAlloc(layer->layer);
    else if(layer->type == NNCLayerType_Optimizer_RMSProp) NNCOptimizerRMSPropDeAlloc(layer->layer);
    else if(layer->type == NNCLayerType_Optimizer_SGD) NNCOptimizerSGDDeAlloc(layer->layer);
    else if(layer->type == NNCLayerType_Layer_Dense) NNCDenseLayerDeAlloc(layer->layer);
    else if(layer->type == NNCLayerType_Layer_Dense_With_Regularization) NNCDenseLayerDeAlloc(layer->layer);
    else if(layer->type == NNCLayerType_Layer_Dropout) NNCDropoutLayerDeAlloc(layer->layer);

    free(layer);
}

void NNCModelPrintLayers(NNCIModelType model) {
#if DEBUG == 1
    dprintf("Model %s layers : \n");
    for(int x = 0; x < model->layer_len; x ++) dprintf("  %d. %s - %s\n", x + 1, model->layers[x]->tag, NNCModelLayerElementTypeToString[model->layers[x]->type]);
    if(model->layer_len == 0) dprintf("Model has 0 layers");
    dprintf("--------------\n");
#endif
}

NNCIModelLayerOutputType NNCModelLayerOutputAlloc(NNCIMatrixType output, nnc_bool must_not_deallocate){
    NNCIModelLayerOutputType layer_output = malloc(sizeof(NNCModelLayerOutputType));
    layer_output->data = output;
    layer_output->must_not_deallocate = must_not_deallocate;
    return layer_output;
}

void NNCIModelLayerOutputDeAlloc(NNCIModelLayerOutputType output){
    if(output->must_not_deallocate == nnc_false && output->data != nnc_null) NNCMatrixDeAlloc(output->data);
    free(output);
}