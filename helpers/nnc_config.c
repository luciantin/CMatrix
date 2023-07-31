#include <stdlib.h>
#include <stdio.h>
#include "nnc_config.h"

nnc_mtype NNCGetRandomMType() {
    return (nnc_mtype)rand()/RAND_MAX*2.0-1.0;//rand() % (NNC_MAX_RAND + 1 - NNC_MIN_RAND) + NNC_MIN_RAND;
}
