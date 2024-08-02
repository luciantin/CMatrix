#include "development.h"
#include "helpers/nnc_list.h"

int main() {


    NNCSerializedModelLoadFromFile("DoubleDense32_10_0.232000.model");

    RunDevelopment();


    return 0;
}
