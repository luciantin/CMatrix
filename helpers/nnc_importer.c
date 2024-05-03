#include <stdio.h>
#include <malloc.h>
#include "nnc_importer.h"

NNCIMatrixType NNCImportMatrixFromFile(char* fileName){
    size_t max_num_len = 1024;

    FILE *file = fopen(fileName, "r");
    if (file == NULL) return nnc_null; //could not open file

    // Get Size
//    size_t n = 0;
//    fseek(file, 0, SEEK_END);
//    long f_size = ftell(file);
//    fseek(file, 0, SEEK_SET);

    int c;

    char* currentConcatValue;
    currentConcatValue = malloc(1024);
    int currentConcatValueIndex = 0;

    int lenX = -1;
    int lenY = -1;
    int currentMatrixIndex = 0;
    int currentCharIndex = 0;
    int currentFieldIndex = 0;
    NNCIMatrixType matrix = nnc_null;

    while ((c = fgetc(file)) != EOF) {
        if((char)c == ';'){

            currentConcatValue[currentConcatValueIndex] = '\0';

            if(currentFieldIndex >= 2){
                nnc_mtype value = atof(currentConcatValue);
                if(matrix != nnc_null) matrix->matrix[(int)(currentMatrixIndex/lenX)][(int)(currentMatrixIndex%lenX)] = value;
                currentMatrixIndex += 1;
            }else if(currentFieldIndex == 0){
                lenX = atoi(currentConcatValue);
            }else if(currentFieldIndex == 1){
                lenY = atoi(currentConcatValue);
                if(lenX > 0 && lenY > 0) matrix = NNCMatrixAlloc(lenX, lenY);
            }else if(matrix == nnc_null || lenY == 0 || lenX == 0){
                break;
            }

            // 5*5, 21,  x = 4, y = (21 - 4*5)

            currentFieldIndex += 1;
            currentConcatValueIndex = 0;

        }else{
            currentConcatValue[currentConcatValueIndex] = (char)c;
            currentConcatValueIndex += 1;
        }
        currentCharIndex += 1;
    }

    free(currentConcatValue);
    fclose(file);
    return matrix;
}