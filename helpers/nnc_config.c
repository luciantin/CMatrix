#include "nnc_config.h"

// izmedu -1 i 1
nnc_mtype NNCGetRandomMType() {
    return (nnc_mtype)nnc_get_rand_int / nnc_rand_max * 2.0 -1.0;//rand() % (NNC_MAX_RAND + 1 - NNC_MIN_RAND) + NNC_MIN_RAND;
}

nnc_mtype NNCGetRandomUnsignedMType() {
    return (nnc_mtype)nnc_get_rand_int / nnc_rand_max * 1.0;//rand() % (NNC_MAX_RAND + 1 - NNC_MIN_RAND) + NNC_MIN_RAND;
}

nnc_bool NNCStrComp(const char* str_a, const char* str_b) {
    nnc_uint index = 0;
    while(1) {
        if(str_a[index] == '\0' && str_b[index] == '\0') return nnc_true;
        else if(str_a[index] != str_b[index]) return nnc_false;
        else if(str_a[index] == '\0' || str_b[index] == '\0') return nnc_false;
        index ++;
    }
}
