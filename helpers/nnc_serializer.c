#include "nnc_serializer.h"
#include "nnc_optimizer.h"


NNCISerializedLayerType NNCSerializedLayerTypeAlloc(Node* layer, nnc_uint len, enum NNCModelLayerElementType type){
    NNCISerializedLayerType l = malloc(sizeof(NNCSerializedLayerType));
    l->layer = layer;
    l->type = type;
    l->len = len;
    return l;
}

void NNCSerializedLayerTypeDeAlloc(NNCISerializedLayerType layer){
    free(layer->layer);
    free(layer);
}

NNCIDeSerializedModelType NNCDeSerializedModelAlloc(enum NNCSerializerType type, NNCIModelType model, NNCITrainerType trainer){
    NNCIDeSerializedModelType dm = malloc(sizeof(NNCDeSerializedModelType));
    dm->type = type;
    dm->model = model;
    dm->trainer = trainer;
    return dm;
}

void NNCDeSerializedModelDeAlloc(NNCIDeSerializedModelType model){
    free(model);
}

NNCISerializedModelType NNCSerializedModelAlloc(enum NNCSerializerType type, Node* model, nnc_uint model_len, Node* trainer, nnc_uint trainer_len, NNCISerializedLayerType* layers, nnc_uint layers_len, NNCISerializedLayerType optimizer){
    NNCISerializedModelType sm = malloc(sizeof(NNCSerializedModelType));

    sm->type = type;
    sm->model = model;
    sm->model_len = model_len;

    sm->trainer = trainer;
    sm->trainer_len = trainer_len;

    sm->layers = layers;
    sm->layers_len = layers_len;

    sm->optimizer = optimizer;

    return sm;
}

void NNCSerializedModelDeAlloc(NNCISerializedModelType model){
    if(model->model_len > 0) free(model->model);
    for(int i = 0; i < model->layers_len; i++) NNCSerializedLayerTypeDeAlloc(model->layers[i]);
    if(model->optimizer != nnc_null) free(model->optimizer);
    if(model->trainer_len > 0) free(model->trainer);
}

void NNCSerializedModelAddLayer(NNCISerializedModelType model, NNCISerializedLayerType layer){
    NNCISerializedLayerType* tmp_layer = malloc(sizeof(NNCISerializedLayerType*) * (model->layers_len + 1));

    for(int i = 0; i < model->layers_len; i ++) tmp_layer[i] = model->layers[i];
    tmp_layer[model->layers_len] = layer;

    free(model->layers);
    model->layers = tmp_layer;
    model->layers_len = model->layers_len + 1;
}

void NNCSerializedModelSaveToFile(NNCISerializedModelType model, char* file_name){
#if NNC_ENABLE_FILE_RW == 1
    FILE *file = fopen(file_name, "w");
    if (file == NULL) return; //could not open file

    fprintf(file, "%s", NNCSerializerTypeToString[model->type]); // field index 0
    fprintf(file, ";");

    if(model->model_len > 0){
        fprintf(file, "%d", model->model_len); // field index 1
        fprintf(file, ";");
        fprintf(file, "%s", nodesToString(model->model)); // field index 2
        fprintf(file, ";");
    }
    else{
        fprintf(file, "%d", 0); // field index 1
        fprintf(file, ";");
        fprintf(file, " "); // field index 2
        fprintf(file, ";");
    }

    if(model->trainer_len > 0){
        fprintf(file, "%d", model->trainer_len); // field index 3
        fprintf(file, ";");
        fprintf(file, "%s", nodesToString(model->trainer)); // field index 4
        fprintf(file, ";");
    }
    else{
        fprintf(file, "%d", 0); // field index 3
        fprintf(file, ";");
        fprintf(file, " "); // field index 4
        fprintf(file, ";");
    }

    if(model->trainer_len > 0){
        fprintf(file, "%d", model->optimizer->len); // field index 5
        fprintf(file, ";");
        fprintf(file, "%s", NNCModelLayerElementTypeToString[model->optimizer->type]); // field index 6
        fprintf(file, ";");
        fprintf(file, "%s", nodesToString(model->optimizer->layer)); // field index 7
        fprintf(file, ";");
    }
    else{
        fprintf(file, "%d", 0); // field index 5
        fprintf(file, ";");
        fprintf(file, "%s", NNCModelLayerElementTypeToString[model->optimizer->type]); // field index 6
        fprintf(file, ";");
        fprintf(file, " "); // field index 7
        fprintf(file, ";");
    }

    for(int i = 0; i < model->layers_len; i++){
        fprintf(file, "%d", model->layers[i]->len); // field index 7 + 1
        fprintf(file, ";");
        fprintf(file, "%s", NNCModelLayerElementTypeToString[model->layers[i]->type]); // field index 7 + 2
        fprintf(file, ";");

        if(model->layers[i]->len > 0){
            fprintf(file, "%s", nodesToString(model->layers[i]->layer)); // field index 7 + 3
        }
        else{
            fprintf(file, " "); // field index 7 + 3
        }

        fprintf(file, ";");
    }

    fclose(file);
#endif
}

NNCISerializedModelType NNCSerializedModelLoadFromFile(char* file_name){

    FILE *file = fopen(file_name, "w");
    if (file == NULL) return nnc_null; //could not open file

    NNCISerializedModelType model = malloc(sizeof(NNCSerializedModelType));

    char* currentConcatValue = malloc(4096);
    char c;
    int currentCharIndex = 0;
    int currentConcatValueIndex = 0;

    while ((c = fgetc(file)) != EOF) {
        if((char)c == ';'){
            currentConcatValue[currentConcatValueIndex] = '\0';

        }
        else{
            currentConcatValue[currentConcatValueIndex] = (char)c;
            currentConcatValueIndex += 1;
        }
        currentCharIndex += 1;
    }

    free(currentConcatValue);
    return model;
}

NNCIDeSerializedModelType NNCSerializedModelDeSerialize(NNCISerializedModelType model);

NNCISerializedModelType   NNCDeSerializedModelSerialize(NNCIDeSerializedModelType model){
    NNCISerializedModelType smodel = malloc(sizeof(NNCSerializedModelType));

    smodel->type = model->type;



    return smodel;
}

NNCISerializedLayerType NNCSerializedLayerTypeSerializeLayer(NNCIModelLayerType layer){
    NNCISerializedLayerType slayer = malloc(sizeof(NNCSerializedLayerType));

    if(layer->type == NNCLayerType_Optimizer_Adam){


    }
    else if(layer->type == NNCLayerType_Optimizer_AdaGrad){
    }
    else if(layer->type == NNCLayerType_Optimizer_RMSProp){
    }
    else if(layer->type == NNCLayerType_Optimizer_SGD){

        NNCIOptimizerSGDType opt = (NNCIOptimizerSGDType)layer->layer;

        Node* learning_rate = floatToStrNode(opt->learning_rate);
        Node* current_learning_rate = floatToStrNode(opt->current_learning_rate);
        Node* decay = floatToStrNode(opt->decay);
        Node* iteration = floatToStrNode(opt->iteration);
        Node* momentum = floatToStrNode(opt->momentum);

        appendNodeAtEnd(learning_rate, newNode(':'));
        appendNodeAtEnd(learning_rate, current_learning_rate);
        appendNodeAtEnd(learning_rate, newNode(':'));
        appendNodeAtEnd(learning_rate, decay);
        appendNodeAtEnd(learning_rate, newNode(':'));
        appendNodeAtEnd(learning_rate, iteration);
        appendNodeAtEnd(learning_rate, newNode(':'));
        appendNodeAtEnd(learning_rate, momentum);
        appendNodeAtEnd(learning_rate, newNode(':'));

        slayer->type = layer->type;
        slayer->layer = learning_rate;

    }
    else if(layer->type == NNCLayerType_Layer_Dense){
    }
    else if(layer->type == NNCLayerType_Layer_Dense_With_Regularization){
    }
    else if(layer->type == NNCLayerType_Activation_ReLU){
    }
    else if(layer->type == NNCLayerType_Layer_Dropout){
    }
    else if(layer->type == NNCLayerType_Activation_SoftMax){
    }

    return NNCSerializedLayerTypeAlloc(slayer, len, layer->type);
}

NNCIModelLayerType NNCSerializedLayerTypeDeSerializeLayer(NNCISerializedLayerType layer){
    if( layer->type == NNCLayerType_Optimizer_Adam){
    }
    else if( layer->type == NNCLayerType_Optimizer_AdaGrad){
    }
    else if( layer->type == NNCLayerType_Optimizer_RMSProp){
    }
    else if( layer->type == NNCLayerType_Optimizer_SGD){
    }
    else if(layer->type == NNCLayerType_Layer_Dense){
    }
    else if(layer->type == NNCLayerType_Layer_Dense_With_Regularization){
    }
    else if(layer->type == NNCLayerType_Activation_ReLU){
    }
    else if(layer->type == NNCLayerType_Layer_Dropout){
    }
    else if(layer->type == NNCLayerType_Activation_SoftMax){
    }
}

NNCIMatrixType NNCImportMatrixFromFile(char* fileName){

#if NNC_ENABLE_FILE_RW == 1
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

#else
    return nnc_null;
#endif
}