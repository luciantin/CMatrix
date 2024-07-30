#include "development.h"
#include "helpers/nnc_list.h"

int main() {

    RunDevelopment();


//    NNCIList lst = NNCListAllocChar('S');
//    NNCListAppendFloat(lst, 3.432);
//    NNCListAppendInt(lst, 80082);
//    NNCListAppendInt(lst, 2);
//    NNCListAppendFloat(lst, 0);

//    NNCIListCString str = NNCListToCString(lst, 1, ';', 1);
//    printf("\n%s\n", str->string);
//
//    NNCIMatrixType inputs_test = NNCImportMatrixFromFile("../training/datasets/dataset_numbers_100_test.matrix");
//    NNCIMatrixType target_test = NNCImportMatrixFromFile("../training/datasets/dataset_numbers_100_truth_test.matrix");
//
//    NNCIMatrixType matrix = inputs_test;//NNCMatrixAllocDiagonal(5, 5, 1);
//    NNCIList mlst = NNCMatrixTypeToList(matrix);
//
//    NNCIListCString mstr = NNCListToCString(mlst, 1, ';', 0);
//    printf("\n%s\n", mstr->string);
//
//    FILE *fp = fopen("sm.txt", "ab");
//    if (fp != NULL)
//    {
//        fputs(mstr->string, fp);
//        fclose(fp);
//    }

    return 0;
}
