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

NNCIMatrixType NNCModelTrain(NNCIModelType model, NNCITrainerType trainer, NNCIMatrixType input, NNCIMatrixType target){
    NNCIMatrixType* output_forward_lst = NNCModelLayerForwardPass(model, input);
    NNCIMatrixType* output_backward_lst = NNCModelLayerBackwardPass(model, target, output_forward_lst);
    NNCModelOptimizerPass(model);

    for(int i = 0; i < model->layer_len; i++) if(output_forward_lst[i] != nnc_null) free(output_forward_lst[i]);
    for(int i = 0; i < model->layer_len; i++) if(output_backward_lst[i] != nnc_null) free(output_backward_lst[i]);
}

NNCIMatrixType* NNCModelLayerForwardPass(NNCIModelType model, NNCIMatrixType input){
    NNCIMatrixType* _output_forward_lst = malloc(sizeof(NNCIMatrixType) * model->layer_len);
    NNCIMatrixType _output_forward = input;

    for(int layer_index = 0; layer_index < model->layer_len; layer_index ++){
        dprintf("%d. - %s - ", layer_index + 1, model->tag);
        _output_forward = NNCModelLayerForwardStep(model->layers[layer_index], _output_forward);
        _output_forward_lst[layer_index] = _output_forward;
    }
    return _output_forward_lst;
}

NNCIMatrixType NNCModelLayerForwardStep(NNCIModelLayerType layer, NNCIMatrixType input){
    dprintf("forward - %s - %s\n", layer->tag, NNCModelLayerElementTypeToString[layer->type]);

    if(layer->type == NNCLayerType_Layer_Dense){
        return NNCDenseLayerForward(input, layer->layer);
    }
    else if(layer->type == NNCLayerType_Activation_ReLU){
        return NNCActivationReLUForward(input);
    }
    else if(layer->type == NNCLayerType_Layer_Dropout){
        return NNCDropoutLayerForward(input, layer->layer);
    }
    else if(layer->type == NNCLayerType_Activation_SoftMax){
        return NNCActivationSoftMaxForward(input);
    }
    else return nnc_null;
}

NNCIMatrixType* NNCModelLayerBackwardPass(NNCIModelType model, NNCIMatrixType target, NNCIMatrixType* output_forward_lst){
    NNCIMatrixType* _output_backward_lst = malloc(sizeof(NNCIMatrixType) * model->layer_len);
    NNCIMatrixType _output_backward = target;

    for(int layer_index = model->layer_len - 1; layer_index >= 0; layer_index --){
        dprintf("%d. - %s - ", layer_index + 1, model->tag);
        _output_backward = NNCModelLayerBackwardStep(model->layers[layer_index], _output_backward, output_forward_lst[layer_index]);
        _output_backward_lst[layer_index] = _output_backward;
    }
    return _output_backward_lst;
}

NNCIMatrixType NNCModelLayerBackwardStep(NNCIModelLayerType layer, NNCIMatrixType dvalues, NNCIMatrixType layer_output_forward){
    dprintf("backward - %s - %s\n", layer->tag, NNCModelLayerElementTypeToString[layer->type]);

    if(layer->type == NNCLayerType_Layer_Dense){
        NNCDenseLayerBackward(dvalues, layer->layer);
        return ((NNCIDenseLayerType)(layer->layer))->dinputs;
    }
    else if(layer->type == NNCLayerType_Activation_ReLU){
        return NNCActivationReLUBackward(layer_output_forward, dvalues);
    }
    else if(layer->type == NNCLayerType_Layer_Dropout){
        NNCDropoutLayerBackward(dvalues, layer->layer);
        return ((NNCIDropoutLayerType)(layer->layer))->dinputs;
    }
    else if(layer->type == NNCLayerType_Activation_SoftMax){
        return NNCActivationSoftMaxLossCCELBackward(layer_output_forward, dvalues);
    }
    else return nnc_null;
}

void NNCModelOptimizerPass(NNCIModelType model) {
    dprintf("x. - %s - optimizer - %s - %s\n", model->tag, model->optimizer->tag, NNCModelLayerElementTypeToString[model->optimizer->type]);

    if(model->optimizer->type == NNCLayerType_Optimizer_Adam){
        NNCOptimizerAdamPreUpdateParams(model->optimizer->layer);
        for(int layer_index = 0; layer_index < model->layer_len; layer_index ++){
            if(model->layers[layer_index]->type == NNCLayerType_Layer_Dense) NNCOptimizerAdamUpdateParams(model->optimizer->layer, model->layers[layer_index]->layer);
        }
        NNCOptimizerAdamPostUpdateParams(model->optimizer->layer);
    }
    else if(model->optimizer->type == NNCLayerType_Optimizer_AdaGrad){
        NNCOptimizerAdaGradPreUpdateParams(model->optimizer->layer);
        for(int layer_index = 0; layer_index < model->layer_len; layer_index ++){
            if(model->layers[layer_index]->type == NNCLayerType_Layer_Dense) NNCOptimizerAdaGradUpdateParams(model->optimizer->layer, model->layers[layer_index]->layer);
        }
        NNCOptimizerAdaGradPostUpdateParams(model->optimizer->layer);
    }
    else if(model->optimizer->type == NNCLayerType_Optimizer_RMSProp){
        NNCOptimizerRMSPropPreUpdateParams(model->optimizer->layer);
        for(int layer_index = 0; layer_index < model->layer_len; layer_index ++){
            if(model->layers[layer_index]->type == NNCLayerType_Layer_Dense) NNCOptimizerRMSPropUpdateParams(model->optimizer->layer, model->layers[layer_index]->layer);
        }
        NNCOptimizerRMSPropPostUpdateParams(model->optimizer->layer);
    }
    else if(model->optimizer->type == NNCLayerType_Optimizer_SGD){
        NNCOptimizerSGDPreUpdateParams(model->optimizer->layer);
        for(int layer_index = 0; layer_index < model->layer_len; layer_index ++){
            if(model->layers[layer_index]->type == NNCLayerType_Layer_Dense) NNCOptimizerSGDUpdateParams(model->optimizer->layer, model->layers[layer_index]->layer);
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

void NNCModelLayerDeAlloc(NNCIModelLayerType element) {
    free(element);
}

void NNCModelPrintLayers(NNCIModelType model) {
#if DEBUG == 1
    dprintf("Model %s layers : \n");
    for(int x = 0; x < model->layer_len; x ++) dprintf("  %d. %s - %s\n", x + 1, model->layers[x]->tag, NNCModelLayerElementTypeToString[model->layers[x]->type]);
    if(model->layer_len == 0) dprintf("Model has 0 layers");
    dprintf("--------------\n");
#endif
}

NNCIModelStatistics NNCModelCalculateStatistics(NNCIModelType model, NNCITrainerType trainer, NNCIMatrixType forward_pass_result, NNCIMatrixType target, nnc_uint target_len){
    NNCIModelStatistics statistics = malloc(sizeof(NNCModelStatistics));

    NNCIMatrixType ccel1_forward = NNCLossCCELForward(forward_pass_result, target);
    statistics->mean = NNCMatrixMean(ccel1_forward);

    nnc_vector argmax_prediction = NNCMatrixArgMax(forward_pass_result);
    nnc_vector argmax_target = NNCMatrixToVector(target, 1);
    statistics->accuracy = NNCVectorAccuracy(argmax_target, argmax_prediction, target_len);

    nnc_mtype regularization_loss = 0;
    for(int x = 0; x < model->layer_len; x ++){
        if(model->layers[x]->type == NNCLayerType_Layer_Dense) regularization_loss += NNCDenseLayerCalculateRegularizationLoss(model->layers[x]->layer);
    }
    statistics->regularization_loss = regularization_loss;

    statistics->sample_len = target_len;
    statistics->current_epoch = trainer->current_epoch;
    statistics->total_epoch = trainer->max_epoch;

    return statistics;
}


void NNCModelPrintTrainingStatistics(NNCIModelType model, NNCIMatrixType* forward_pass_result_lst, ){
//#if DEBUG == 1
//    NNCIMatrixType ccel1_forward = NNCLossCCELForward(softmax1_forward, target);
//    nnc_mtype mean = NNCMatrixMean(ccel1_forward);
//    nnc_vector argmax_prediction = NNCMatrixArgMax(softmax1_forward);
//    nnc_vector argmax_target = NNCMatrixToVector(target, 1);
//    nnc_mtype regularization_loss = NNCDenseLayerCalculateRegularizationLoss(dense1) + NNCDenseLayerCalculateRegularizationLoss(dense2);

//            NNCVectorPrint(argmax_prediction, sample_len);
//            NNCVectorPrint(argmax_target, sample_len);

    dprintf("epoch : %d ", epoch);
    dprintf(" lrate : %.9g ", optimizerAdam->current_learning_rate);
    dprintf("acc : %.9g", NNCVectorAccuracy(argmax_target, argmax_prediction, sample_len));
    dprintf(" reg loss : %.6g", regularization_loss);
    dprintf(" mean loss : %.9g \n", mean);

//    if(epoch == epoch_len) {
//        NNCVectorAccuracy(argmax_target, argmax_prediction, sample_len);
//        dprintf("Target :    ");
//        NNCVectorPrint(argmax_target, sample_len);
//        dprintf("Prediction :");
//        NNCVectorPrint(argmax_prediction, sample_len);
//    }

    free(argmax_target);
    free(argmax_prediction);
    NNCMatrixDeAlloc(ccel1_forward);
//#endif
}

NNCIMatrixType NNCModelTest(NNCIModelType model, NNCIMatrixType input, NNCIMatrixType target);
NNCIMatrixType NNCModelPredict(NNCIModelType model, NNCIMatrixType input);
