#ifndef CMATRIX_NNC_TRAINER_H
#define CMATRIX_NNC_TRAINER_H


#include "nnc_config.h"

typedef struct NNCTrainerType
{
    char*                   tag;
    nnc_uint                max_epoch;
    nnc_uint                current_epoch;
}
NNCTrainerType;

#define NNCITrainerType NNCTrainerType*

#endif //CMATRIX_NNC_TRAINER_H
