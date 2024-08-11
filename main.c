#include "development.h"
#include "helpers/nnc_list.h"

int main() {

    RunDevelopment();

    NNCISerializedModelType model = NNCSerializedModelLoadFromFile("DoubleDense32_10_0.149000.model");
    NNCSerializedModelPrint(model);

    NNCIDeSerializedModelType desmodel = NNCSerializedModelDeSerialize(model);
    NNCModelPrintLayers(desmodel->model);

    NNCIMatrixType inputs_test = NNCImportMatrixFromFile("../training/datasets/dataset_numbers_1000_test.matrix");
    NNCIMatrixType target_test = NNCImportMatrixFromFile("../training/datasets/dataset_numbers_1000_truth_test.matrix");

    NNCITrainerType trainer = NNCTrainerAlloc("trainer", NNCTrainerTypeStrategy_Iterative, 2);
    NNCIModelStatistics statistics_test = NNCTrainerTest(trainer, desmodel->model, inputs_test, target_test);
    NNCStatisticsPrint(statistics_test);

//    NNCSerializedModelDeAlloc(model);

    return 0;
}
