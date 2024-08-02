#ifndef CMATRIX_DEVELOPMENT_H
#define CMATRIX_DEVELOPMENT_H

#include <malloc.h>
#include "helpers/nnc_matrix.h"
#include "helpers/nnc_layer.h"
#include "helpers/nnc_optimizer.h"
#include "helpers/nnc_activation_layer.h"
#include "helpers/nnc_loss_function.h"
#include "helpers/nnc_vector.h"
#include "helpers/nnc_config.h"
#include "helpers/nnc_model.h"
#include "helpers/nnc_trainer.h"
#include "helpers/nnc_statistics.h"
#include "helpers/nnc_serializer.h"

void RunDevelopment(){

    nnc_init_rand(111111);

    #if DEBUG
    setbuf(stdout, NULL); // Clion dprintf lag fix
    #endif

    dprintf("Start\n");
    dputs("--------------");

    NNCIMatrixType inputs_training = NNCImportMatrixFromFile("../training/datasets/dataset_numbers_1000_train.matrix");
    NNCIMatrixType target_training = NNCImportMatrixFromFile("../training/datasets/dataset_numbers_1000_truth_train.matrix");

    NNCIMatrixType inputs_test = NNCImportMatrixFromFile("../training/datasets/dataset_numbers_1000_test.matrix");
    NNCIMatrixType target_test = NNCImportMatrixFromFile("../training/datasets/dataset_numbers_1000_truth_test.matrix");

    NNCIMatrixType input = inputs_training;
    NNCIMatrixType target = target_training;

    int sample_len = input->y;
    int input_len = input->x;
    int output_len = 10;
    int epoch_len = 10;

    nnc_mtype momentum = 0;
    nnc_mtype learning_rate = 0.05;
    nnc_mtype decay = 1e-5;
    nnc_mtype epislon = 1e-7;
    nnc_mtype beta_1 = 0.9;
    nnc_mtype beta_2 = 0.999;
    nnc_mtype dropout_rate = 0.01;

    NNCIDenseLayerType dense1 = NNCDenseLayerAlloc(input_len, 16);
    NNCDenseLayerSetRegularizationParameters(dense1, 0, 5e-4, 0, 5e-4);

    NNCIDropoutLayerType dropout1 = NNCDropoutLayerAlloc(dropout_rate);

    NNCIDenseLayerType dense2 = NNCDenseLayerAlloc(16, 16);
    NNCDenseLayerSetRegularizationParameters(dense2, 0, 5e-4, 0, 5e-4);

    NNCIDenseLayerType dense3 = NNCDenseLayerAlloc(16, output_len);
    NNCDenseLayerSetRegularizationParameters(dense3, 0, 5e-4, 0, 5e-4);

    NNCIOptimizerAdamType optimizerAdam = NNCOptimizerAdamAlloc(learning_rate, decay, epislon, beta_1, beta_2);
    NNCIOptimizerAdaGradType optimizerAdaGrad = NNCOptimizerAdaGradAlloc(learning_rate, decay);

    NNCIModelType model = NNCModelAlloc("DoubleDense32");

    NNCIModelLayerType layerDense1 = NNCModelLayerAlloc(dense1, NNCLayerType_Layer_Dense_With_Regularization, "dense1");
    NNCIModelLayerType layerActivationReLu = NNCModelLayerAlloc(nnc_null, NNCLayerType_Activation_ReLU, "relu1");
    NNCIModelLayerType layerActivationSoftmax1 = NNCModelLayerAlloc(nnc_null, NNCLayerType_Activation_SoftMax, "softmax1");
    NNCIModelLayerType layerDropout1 = NNCModelLayerAlloc(dropout1, NNCLayerType_Layer_Dropout, "dropout1");
    NNCIModelLayerType layerDense2 = NNCModelLayerAlloc(dense2, NNCLayerType_Layer_Dense_With_Regularization, "dense2");
    NNCIModelLayerType layerDense3 = NNCModelLayerAlloc(dense3, NNCLayerType_Layer_Dense_With_Regularization, "dense3");
    NNCIModelLayerType layerActivationSoftMax = NNCModelLayerAlloc(nnc_null, NNCLayerType_Activation_SoftMax, "softmax1");
    NNCIModelLayerType layerOptimizerAdam = NNCModelLayerAlloc(optimizerAdam, NNCLayerType_Optimizer_Adam, "optimizerAdam");
    NNCIModelLayerType layerOptimizerAdaGrad = NNCModelLayerAlloc(optimizerAdaGrad, NNCLayerType_Optimizer_AdaGrad, "optimizerAdaGrad");
    NNCITrainerType trainer = NNCTrainerAlloc("trainer", NNCTrainerTypeStrategy_Iterative, epoch_len);

    NNCModelLayerAdd(model, layerDense1);
    NNCModelLayerAdd(model, layerActivationReLu);
//    NNCModelLayerAdd(model, layerDropout1);
    NNCModelLayerAdd(model, layerDense2);
//    NNCModelLayerAdd(model, layerActivationSoftmax1);
    NNCModelLayerAdd(model, layerDense3);
    NNCModelLayerAdd(model, layerActivationSoftMax);

    NNCModelSetOptimizer(model, layerOptimizerAdaGrad);

    NNCModelPrintLayers(model);

    NNCIModelStatistics statistics_train = NNCTrainerTrain(trainer, model, input, target);
    NNCStatisticsPrint(statistics_train);

    NNCIModelStatistics statistics_test = NNCTrainerTest(trainer, model, inputs_test, target_test);
    NNCStatisticsPrint(statistics_test);

//    dprintf("%s", NNCIMatrixSerialize(dense2->weights)->matrix);

    NNCISerializedModelType demodel = NNCSerialize(NNCSerializer_TrainedModel, model, trainer);

    char* file_name = malloc(sizeof(char) * 200);
    sprintf(file_name, "%s_%d_%f.model", model->tag, statistics_train->total_epoch, statistics_test->accuracy);
    NNCSerializedModelSaveToFile(demodel, file_name);

    dprintf("Done");
}

#endif //CMATRIX_DEVELOPMENT_H
