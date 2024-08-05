#include "development.h"
#include "helpers/nnc_list.h"

int main() {

//    dprintf("\n%d\n", NNCStrComp("1", "1test"));

    RunDevelopment();

    NNCISerializedModelType model = NNCSerializedModelLoadFromFile("DoubleDense32_10_0.149000.model");
    NNCSerializedModelPrint(model);
    NNCSerializedModelDeAlloc(model);

    return 0;
}
