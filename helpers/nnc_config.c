#include "nnc_config.h"

// izmedu -1 i 1
nnc_mtype NNCGetRandomMType() {
    return (nnc_mtype)nnc_get_rand_int / nnc_rand_max * 2.0 -1.0;//rand() % (NNC_MAX_RAND + 1 - NNC_MIN_RAND) + NNC_MIN_RAND;
}

nnc_mtype NNCGetRandomUnsignedMType() {
    return (nnc_mtype)nnc_get_rand_int / nnc_rand_max * 1.0;//rand() % (NNC_MAX_RAND + 1 - NNC_MIN_RAND) + NNC_MIN_RAND;
}
