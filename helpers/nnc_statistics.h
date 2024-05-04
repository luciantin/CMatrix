#ifndef CMATRIX_NNC_STATISTICS_H
#define CMATRIX_NNC_STATISTICS_H

#include "nnc_model.h"

typedef struct NNCModelStatistics
{
    nnc_uint sample_len;
    nnc_uint current_epoch;
    nnc_uint total_epoch;
    nnc_mtype regularization_loss;
    nnc_mtype mean_loss;
    nnc_mtype accuracy;
}
NNCModelStatistics;

#define NNCIModelStatistics NNCModelStatistics*


NNCIModelStatistics NNCStatisticsCalculate(NNCIModelType model, nnc_uint current_epoch, nnc_uint max_epoch, NNCIMatrixType forward_pass_result, NNCIMatrixType target);
void NNCStatisticsPrint(NNCIModelStatistics statistics);

#endif //CMATRIX_NNC_STATISTICS_H
