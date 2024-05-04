#include "nnc_statistics.h"
#include "nnc_layer.h"
#include "nnc_vector.h"
#include "nnc_loss_function.h"

NNCIModelStatistics NNCStatisticsCalculate(NNCIModelType model, nnc_uint current_epoch, nnc_uint max_epoch, NNCIMatrixType forward_pass_result, NNCIMatrixType target){
    NNCIModelStatistics statistics = malloc(sizeof(NNCModelStatistics));

    NNCIMatrixType ccel_forward = NNCLossCCELForward(forward_pass_result, target);

    nnc_vector argmax_prediction = NNCMatrixArgMax(forward_pass_result);
    nnc_vector argmax_target = NNCMatrixToVector(target, 1);

    nnc_mtype regularization_loss = 0;
    for(int x = 0; x < model->layer_len; x ++){
        if(model->layers[x]->type == NNCLayerType_Layer_Dense || model->layers[x]->type == NNCLayerType_Layer_Dense_With_Regularization) regularization_loss += NNCDenseLayerCalculateRegularizationLoss(model->layers[x]->layer);
    }

    statistics->accuracy = NNCVectorAccuracy(argmax_target, argmax_prediction, target->y);
    statistics->mean_loss = NNCMatrixMean(ccel_forward);
    statistics->regularization_loss = regularization_loss;
    statistics->sample_len = target->y;
    statistics->current_epoch = current_epoch;
    statistics->total_epoch = max_epoch;

    free(argmax_target);
    free(argmax_prediction);
    NNCMatrixDeAlloc(ccel_forward);

    return statistics;
}

void NNCStatisticsPrint(NNCIModelStatistics statistics){
    dprintf("# Epoch : %d / %d ", statistics->current_epoch, statistics->total_epoch);
    dprintf(" Acc : %.5g%% ", statistics->accuracy * 100);
    dprintf(" Reg loss : %.5g ", statistics->regularization_loss);
    dprintf(" Mean loss : %.5g \n", statistics->mean_loss);
}
