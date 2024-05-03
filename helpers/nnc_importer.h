#ifndef CMATRIX_NNC_IMPORTER_H
#define CMATRIX_NNC_IMPORTER_H

#include "nnc_matrix.h"

// file structure :
// lenX;lenY;val1;val2;...;valN;
NNCIMatrixType NNCImportMatrixFromFile(char* fileName);


#endif //CMATRIX_NNC_IMPORTER_H
