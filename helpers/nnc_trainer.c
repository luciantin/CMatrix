#include "nnc_trainer.h"
#include "nnc_statistics.h"
#include "nnc_optimizer.h"
#include "nnc_loss_function.h"
#include "nnc_activation_layer.h"
#include "nnc_vector.h"


NNCITrainerType NNCTrainerAlloc(char* tag, enum NNCTrainerTypeStrategy strategy, nnc_uint max_epoch){
    NNCITrainerType trainer = malloc(sizeof(NNCTrainerType));
    trainer->tag = tag;
    trainer->max_epoch = max_epoch;
    trainer->current_epoch = 1;
    trainer->strategy = strategy;
    return trainer;
}
void NNCTrainerDeAlloc(NNCITrainerType trainer){
    free(trainer);
}

NNCIModelStatistics NNCTrainerTrain(NNCITrainerType trainer, NNCIModelType model, NNCIMatrixType input, NNCIMatrixType target){
    NNCIModelStatistics statistics = nnc_null;

    NNCIDenseLayerType dense1 = model->layers[0]->layer;
    NNCIDenseLayerType dense2 = model->layers[2]->layer;


    for(trainer->current_epoch; trainer->current_epoch <= trainer->max_epoch; trainer->current_epoch++){

        #if NNC_CALCULATE_EXECUTION_TIME == 1
                time_type epoch_start = dgetTime();
        #endif

        NNCIModelLayerOutputType* output_forward_lst = NNCModelLayerForwardPass(model, input);

        if(statistics != nnc_null) free(statistics);
        statistics = NNCStatisticsCalculate(model, trainer->current_epoch, trainer->max_epoch, output_forward_lst[model->layer_len-1]->data, target);

        NNCIModelLayerOutputType* output_backward_lst = NNCModelLayerBackwardPass(model, target, output_forward_lst);
        NNCModelOptimizerPass(model);

        for(int i = 0; i < model->layer_len; i++) if(output_forward_lst[i] != nnc_null) NNCIModelLayerOutputDeAlloc(output_forward_lst[i]);
        for(int i = 0; i < model->layer_len; i++) if(output_backward_lst[i] != nnc_null) NNCIModelLayerOutputDeAlloc(output_backward_lst[i]);

        free(output_forward_lst);
        free(output_backward_lst);
//        if(trainer->current_epoch == trainer->max_epoch){
//            for(int i = 0; i < model->layer_len; i++) if(output_forward_lst[i] != nnc_null) NNCIModelLayerOutputDeAllocForced(output_forward_lst[i]);
//            for(int i = 0; i < model->layer_len; i++) if(output_backward_lst[i] != nnc_null) NNCIModelLayerOutputDeAllocForced(output_backward_lst[i]);
//        }

        NNCStatisticsPrint(statistics);

        #if NNC_CALCULATE_EXECUTION_TIME == 1
                time_type epoch_end = dgetTime();
                time_type time_diff = dgetTimeDiff(epoch_start, epoch_end);
                dprintf("# Epoch duration : %.3Lg seconds, finished in ~%.4Lg minutes / ~%.4Lg hours\n", time_diff, (time_diff * (trainer->max_epoch - trainer->current_epoch))/60, (time_diff * (trainer->max_epoch - trainer->current_epoch))/60/60);
        #endif
    }

    return statistics;
}

void NNCTrainerLinkModel(NNCITrainerType trainer);

NNCIModelStatistics NNCTrainerTest(NNCITrainerType trainer, NNCIModelType model, NNCIMatrixType input, NNCIMatrixType target){
    NNCIModelStatistics statistics = nnc_null;

    NNCIDenseLayerType dense1 = model->layers[0]->layer;
    NNCIDenseLayerType dense2 = model->layers[2]->layer;

    dprintf("------------TEST------------");

    #if NNC_CALCULATE_EXECUTION_TIME == 1
        time_type epoch_start = dgetTime();
    #endif

    NNCIModelLayerOutputType* output_forward_lst = NNCModelLayerForwardPass(model, input);

    statistics = NNCStatisticsCalculate(model, trainer->current_epoch, trainer->max_epoch, output_forward_lst[model->layer_len-1]->data, target);

    for(int i = 0; i < model->layer_len; i++) if(output_forward_lst[i] != nnc_null) NNCIModelLayerOutputDeAlloc(output_forward_lst[i]);
    NNCStatisticsPrint(statistics);

    free(output_forward_lst);

    #if NNC_CALCULATE_EXECUTION_TIME == 1
        time_type epoch_end = dgetTime();
        time_type time_diff = dgetTimeDiff(epoch_start, epoch_end);
        dprintf("# Test duration : %.3Lg seconds\n", time_diff);
    #endif

    return statistics;
}

NNCIMatrixType NNCTrainerPredict(NNCITrainerType trainer, NNCIModelType model, NNCIMatrixType input);
