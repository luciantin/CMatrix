#ifndef CMATRIX_NNC_CONFIG_H
#define CMATRIX_NNC_CONFIG_H

#define NNC_MIN_RAND -1
#define NNC_MAX_RAND 1

#define nnc_mtype       double
#define nnc_vector      nnc_mtype*
#define nnc_uint        unsigned int
#define nnc_matrix      nnc_mtype**
#define nnc_endOfString '\0'

#define PRINT_TOP_LEFT_MATRIX_START       true
#define PRINT_BOTTOM_LEFT_MATRIX_START    false

nnc_mtype NNCGetRandomMType();

#endif //CMATRIX_NNC_CONFIG_H
