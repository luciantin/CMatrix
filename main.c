#include "development.h"
#include "helpers/nnc_list.h"

int main() {

    RunDevelopment();

    NNCISerializedModelType model = NNCSerializedModelLoadFromFile("DoubleDense32_10_0.149000.model");
    NNCSerializedModelPrint(model);
    NNCSerializedModelDeSerialize(model);
    NNCSerializedModelDeAlloc(model);

    return 0;
}
