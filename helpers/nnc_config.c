#include <stdlib.h>
#include <stdio.h>
#include "nnc_config.h"

// izmedu -1 i 1
nnc_mtype NNCGetRandomMType() {
    return (nnc_mtype)rand()/RAND_MAX * 2.0 -1.0;//rand() % (NNC_MAX_RAND + 1 - NNC_MIN_RAND) + NNC_MIN_RAND;
}

nnc_mtype NNCGetRandomUnsignedMType() {
    return (nnc_mtype)rand()/RAND_MAX * 1.0;//rand() % (NNC_MAX_RAND + 1 - NNC_MIN_RAND) + NNC_MIN_RAND;
}
