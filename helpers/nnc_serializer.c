//#include <string.h>
#include <string.h>
#include "nnc_serializer.h"
#include "nnc_optimizer.h"


NNCISerializedLayerType NNCSerializedLayerTypeAlloc(NNCIList layer, nnc_uint len, enum NNCModelLayerElementType type){
    NNCISerializedLayerType l = malloc(sizeof(NNCSerializedLayerType));
    l->layer = layer;
    l->type = type;
    l->len = len;
    return l;
}

void NNCSerializedLayerTypeDeAlloc(NNCISerializedLayerType layer){
    NNCListDeAllocAll(layer->layer);
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

NNCISerializedModelType NNCSerializedModelAlloc(enum NNCSerializerType type, NNCIList tag, NNCISerializedLayerType trainer, NNCISerializedLayerType optimizer, NNCISerializedLayerType* layers, nnc_uint layers_len){
    NNCISerializedModelType sm = malloc(sizeof(NNCSerializedModelType));

    sm->type = type;
    sm->tag = tag;
    sm->trainer = trainer;
    sm->layers = layers;
    sm->layers_len = layers_len;
    sm->optimizer = optimizer;

    return sm;
}

void NNCSerializedModelDeAlloc(NNCISerializedModelType model){
    if(model->tag != nnc_null) NNCListDeAllocAll(model->tag);
    for(int i = 0; i < model->layers_len; i++) if(model->layers[i] != nnc_null) NNCSerializedLayerTypeDeAlloc(model->layers[i]);
    if(model->optimizer != nnc_null) NNCSerializedLayerTypeDeAlloc(model->optimizer);
    if(model->trainer != nnc_null) NNCSerializedLayerTypeDeAlloc(model->trainer);
    free(model->layers);
    free(model);
}

void NNCSerializedModelAddLayer(NNCISerializedModelType model, NNCISerializedLayerType layer){
    NNCISerializedLayerType* tmp_layer = malloc(sizeof(NNCSerializedLayerType*) * (model->layers_len + 1));

    for(int i = 0; i < model->layers_len; i ++) tmp_layer[i] = model->layers[i];
    tmp_layer[model->layers_len] = layer;

    free(model->layers);
    model->layers = tmp_layer;
    model->layers_len = model->layers_len + 1;
}

NNCIDeSerializedModelType NNCSerializedModelDeSerialize(NNCISerializedModelType model);

NNCISerializedModelType   NNCDeSerializedModelSerialize(enum NNCSerializerType type, NNCIDeSerializedModelType dsmodel){
    NNCISerializedModelType smodel = malloc(sizeof(NNCSerializedModelType));

    smodel->type = dsmodel->type;

    NNCIListCString ctg = NNCListCStringAllocFromCString(dsmodel->model->tag);
    smodel->tag = NNCCStringToList(ctg, nnc_false, NNC_VALUE_EMPTY_BYTE, false);

    free(ctg);

    if(smodel->type == NNCSerializer_ModelWithTrainer) {
        smodel->trainer = nnc_null; //NNCSerializedLayerTypeSerializeLayer(type, dsmodel->trainer);
    }
    else {
        smodel->trainer = nnc_null;
    }

    if(smodel->type == NNCSerializer_Model || smodel->type == NNCSerializer_ModelWithTrainer) {
        smodel->optimizer = NNCSerializedLayerTypeSerializeLayer(type, dsmodel->model->optimizer);
    }
    else {
        smodel->optimizer = nnc_null;
    }

    smodel->layers = malloc(sizeof(NNCIModelLayerType) * dsmodel->model->layer_len);
    smodel->layers_len = dsmodel->model->layer_len;

    for(int i = 0; i < dsmodel->model->layer_len; i++) {
        smodel->layers[i] = NNCSerializedLayerTypeSerializeLayer(type, dsmodel->model->layers[i]);
    }

    return smodel;
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

            currentConcatValue[currentConcatValueIndex] = nnc_end_of_string;

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

NNCISerializedModelType NNCSerialize(enum NNCSerializerType type, NNCIModelType model, NNCITrainerType trainer) {
    NNCIDeSerializedModelType demodel = NNCDeSerializedModelAlloc(type, model, trainer);
    NNCISerializedModelType smodel = NNCDeSerializedModelSerialize(type, demodel);
    NNCDeSerializedModelDeAlloc(demodel);
    return smodel;
}

NNCISerializedMatrixType NNCIMatrixSerialize(NNCIMatrixType matrix) {
    NNCISerializedMatrixType smatrix = malloc(sizeof(NNCSerializedMatrixType));

    smatrix->matrix = nnc_null;
    smatrix->len = 0;

    for(int x = 0; x < matrix->x; x++){
        for(int y = 0; y < matrix->y; y++){
            int tmp_len = snprintf(NULL, 0, "%f", matrix->matrix[x][y]);
            char* tmp_str = malloc(tmp_len);
            snprintf(tmp_str, tmp_len, "%f", matrix->matrix[x][y]);
            tmp_str[tmp_len-1] = ':';

            if(smatrix->matrix == nnc_null){
                smatrix->matrix = tmp_str;
                smatrix->len = tmp_len;
            }else { // TODO implement realloc
                char* tmp_cpy = malloc(smatrix->len + tmp_len);

                memcpy(tmp_cpy, smatrix->matrix, smatrix->len);
                memcpy(&tmp_cpy[smatrix->len], tmp_str, tmp_len);

                free(smatrix->matrix);
                free(tmp_str);

                smatrix->matrix = tmp_cpy;
                smatrix->len += tmp_len;
            }
        }
    }

    smatrix->matrix[smatrix->len-1] = nnc_end_of_string;
    return smatrix;
}

void NNCISerializedMatrixDeAlloc(NNCISerializedMatrixType smatrix) {
    free(smatrix->matrix);
    free(smatrix);
}


NNCISerializedLayerType NNCSerializedLayerTypeSerializeLayer(enum NNCSerializerType type, NNCIModelLayerType layer){
    NNCISerializedLayerType slayer = malloc(sizeof(NNCSerializedLayerType));

    if(layer->type == NNCLayerType_Optimizer_Adam){
        NNCIOptimizerAdamType opt = (NNCIOptimizerAdamType)layer->layer;

        slayer->type = layer->type;
        slayer->layer = NNCListAllocChar(NNC_VALUE_EMPTY_BYTE);

        NNCListAppendMType(slayer->layer,opt->learning_rate);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->current_learning_rate);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->decay);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->iteration);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->epsilon);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->beta_1);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->beta_2);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

    }
    else if(layer->type == NNCLayerType_Optimizer_AdaGrad){
        NNCIOptimizerAdaGradType opt = (NNCIOptimizerAdaGradType)layer->layer;

        slayer->type = layer->type;
        slayer->layer = NNCListAllocChar(NNC_VALUE_EMPTY_BYTE);

        NNCListAppendMType(slayer->layer,opt->learning_rate);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->current_learning_rate);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->decay);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->iteration);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->epsilon);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

    }
    else if(layer->type == NNCLayerType_Optimizer_RMSProp){
        NNCIOptimizerRMSPropType opt = (NNCIOptimizerRMSPropType)layer->layer;

        slayer->type = layer->type;
        slayer->layer = NNCListAllocChar(NNC_VALUE_EMPTY_BYTE);

        NNCListAppendMType(slayer->layer,opt->learning_rate);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->current_learning_rate);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->decay);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->iteration);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->epsilon);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->rho);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

    }
    else if(layer->type == NNCLayerType_Optimizer_SGD){
        NNCIOptimizerSGDType opt = (NNCIOptimizerSGDType)layer->layer;

        slayer->type = layer->type;
        slayer->layer = NNCListAllocChar(NNC_VALUE_EMPTY_BYTE);

        NNCListAppendMType(slayer->layer,opt->learning_rate);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->current_learning_rate);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->decay);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->iteration);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,opt->momentum);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

    }
    else if(layer->type == NNCLayerType_Layer_Dense || layer->type == NNCLayerType_Layer_Dense_With_Regularization){
        NNCIDenseLayerType lyr = (NNCIDenseLayerType)layer->layer;

        slayer->type = layer->type;
        slayer->layer = NNCListAllocChar(NNC_VALUE_EMPTY_BYTE);

        NNCListAppendInt(slayer->layer,(int)lyr->num_inputs);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendInt(slayer->layer,(int)lyr->num_neurons);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,lyr->l1r_weights);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,lyr->l1r_biases);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,lyr->l2r_weights);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
        NNCListAppendMType(slayer->layer,lyr->l2r_biases);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

        if(lyr->weights != nnc_null) NNCListAppend(slayer->layer, NNCMatrixTypeToList(lyr->weights, nnc_false, NNC_VALUE_END_BYTE));
        else NNCListAppendChar(slayer->layer, NNC_VALUE_NOT_DEFINED);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

        if(lyr->biases != nnc_null) NNCListAppend(slayer->layer, NNCMatrixTypeToList(lyr->biases, nnc_false, NNC_VALUE_END_BYTE));
        else NNCListAppendChar(slayer->layer, NNC_VALUE_NOT_DEFINED);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

        if(lyr->dweights != nnc_null && (type == NNCSerializer_ModelWithTrainer || type == NNCSerializer_Model)) NNCListAppend(slayer->layer, NNCMatrixTypeToList(lyr->dweights, nnc_false, NNC_VALUE_END_BYTE));
        else NNCListAppendChar(slayer->layer, NNC_VALUE_NOT_DEFINED);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

        if(lyr->dbiases != nnc_null && (type == NNCSerializer_ModelWithTrainer || type == NNCSerializer_Model)) NNCListAppend(slayer->layer, NNCMatrixTypeToList(lyr->dbiases, nnc_false, NNC_VALUE_END_BYTE));
        else NNCListAppendChar(slayer->layer, NNC_VALUE_NOT_DEFINED);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

        if(lyr->cweights != nnc_null && (type == NNCSerializer_ModelWithTrainer || type == NNCSerializer_Model)) NNCListAppend(slayer->layer, NNCMatrixTypeToList(lyr->cweights, nnc_false, NNC_VALUE_END_BYTE));
        else NNCListAppendChar(slayer->layer, NNC_VALUE_NOT_DEFINED);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

        if(lyr->cbiases != nnc_null && (type == NNCSerializer_ModelWithTrainer || type == NNCSerializer_Model)) NNCListAppend(slayer->layer, NNCMatrixTypeToList(lyr->cbiases, nnc_false, NNC_VALUE_END_BYTE));
        else NNCListAppendChar(slayer->layer, NNC_VALUE_NOT_DEFINED);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

        if(lyr->mweights != nnc_null && (type == NNCSerializer_ModelWithTrainer || type == NNCSerializer_Model)) NNCListAppend(slayer->layer, NNCMatrixTypeToList(lyr->mweights, nnc_false, NNC_VALUE_END_BYTE));
        else NNCListAppendChar(slayer->layer, NNC_VALUE_NOT_DEFINED);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

        if(lyr->mbiases != nnc_null && (type == NNCSerializer_ModelWithTrainer || type == NNCSerializer_Model)) NNCListAppend(slayer->layer, NNCMatrixTypeToList(lyr->mbiases, nnc_false, NNC_VALUE_END_BYTE));
        else NNCListAppendChar(slayer->layer, NNC_VALUE_NOT_DEFINED);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

        if(lyr->inputs != nnc_null && (type == NNCSerializer_ModelWithTrainer || type == NNCSerializer_Model)) NNCListAppend(slayer->layer, NNCMatrixTypeToList(lyr->inputs, nnc_false, NNC_VALUE_END_BYTE));
        else NNCListAppendChar(slayer->layer, NNC_VALUE_NOT_DEFINED);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

        if(lyr->dinputs != nnc_null && (type == NNCSerializer_ModelWithTrainer || type == NNCSerializer_Model)) NNCListAppend(slayer->layer, NNCMatrixTypeToList(lyr->dinputs, nnc_false, NNC_VALUE_END_BYTE));
        else NNCListAppendChar(slayer->layer, NNC_VALUE_NOT_DEFINED);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);
    }
    else if(layer->type == NNCLayerType_Activation_ReLU){
        slayer->type = layer->type;
        slayer->layer = nnc_null;
    }
    else if(layer->type == NNCLayerType_Layer_Dropout){
        NNCIDropoutLayerType lyr = (NNCIDropoutLayerType)layer->layer;

        slayer->type = layer->type;
        slayer->layer = NNCListAllocChar(NNC_VALUE_EMPTY_BYTE);

        NNCListAppendMType(slayer->layer,lyr->dropout_rate);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

        if(lyr->binary_mask != nnc_null) NNCListAppend(slayer->layer, NNCMatrixTypeToList(lyr->binary_mask, nnc_false, NNC_VALUE_END_BYTE));
        else NNCListAppendChar(slayer->layer, NNC_VALUE_NOT_DEFINED);
        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

        if(lyr->dinputs != nnc_null && (type == NNCSerializer_ModelWithTrainer || type == NNCSerializer_Model)) NNCListAppend(slayer->layer, NNCMatrixTypeToList(lyr->dinputs, nnc_false, NNC_VALUE_END_BYTE));
        else NNCListAppendChar(slayer->layer, NNC_VALUE_NOT_DEFINED);
//        //        NNCListAppendChar(slayer->layer, NNC_VALUE_END_BYTE);

    }
    else if(layer->type == NNCLayerType_Activation_SoftMax){
        slayer->type = layer->type;
        slayer->layer = nnc_null;
    }

    if(slayer->layer != nnc_null) slayer->len = NNCListLength(slayer->layer);

    return slayer;
}

NNCIList NNCSerializedModelMinify(NNCISerializedModelType model) {
    NNCIListCString lctype = NNCListCStringAllocFromCString(NNCSerializerTypeToString[model->type]);
    NNCIList lclist = NNCCStringToList(lctype, nnc_false, NNC_VALUE_NOT_DEFINED, nnc_false);
    free(lctype);
    NNCListAppend(lclist, NNCListAllocChar(NNC_VALUE_END_BYTE));
    NNCListAppend(lclist, NNCListAllocCopy(model->tag));
    NNCListAppend(lclist, NNCListAllocChar(NNC_VALUE_END_BYTE));

    NNCListAppend(lclist, NNCSerializedLayerTypeToList(model->trainer));
    NNCListAppend(lclist, NNCSerializedLayerTypeToList(model->optimizer));

    NNCListAppend(lclist, NNCListAllocInt((int)model->layers_len));

    for(int i = 0; i < model->layers_len; i++) NNCListAppend(lclist, NNCSerializedLayerTypeToList(model->layers[i]));

    return lclist;
}

NNCISerializedModelType NNCSerializedModelMaxify_Destructive(NNCIList list) {

    NNCISerializedModelType             smodel = nnc_null;

    enum NNCSerializerType              type        = NNCSerializer_NONE;
    NNCIList                            model_type  = NNCListAllocInt(1);
    NNCIList                            tag         = NNCListAllocInt(1);
    NNCISerializedLayerType             trainer     = nnc_null;
    NNCISerializedLayerType             optimizer   = nnc_null;
    NNCISerializedLayerType*            layers      = nnc_null;
    nnc_uint                            layers_len  = 0;

    int index = 0;
    NNCIList current_node = NNCListPop(&list);

    while(current_node != nnc_null){
        if(current_node->type == CHAR && *(char*)current_node->value == NNC_VALUE_END_BYTE) {
            NNCListDeAlloc(current_node);
            index += 1;
        }
        else if(index == 0 && current_node->type == CHAR) NNCListAppend(model_type, current_node);
        else if(index == 1 && current_node->type == CHAR) NNCListAppend(tag, current_node);
        else return nnc_null;//NNCListDeAlloc(current_node);

        if(index == 2) break;

        current_node = NNCListPop(&list);
    }

    current_node = NNCListPop(&model_type);
    NNCListDeAlloc(current_node);

    current_node = NNCListPop(&tag);
    NNCListDeAlloc(current_node);

    trainer = NNCSerializedLayerTypeFromList_Destructive(&list);
    optimizer = NNCSerializedLayerTypeFromList_Destructive(&list);

    current_node = NNCListPop(&list);
    if(current_node->type != INT && (nnc_uint)*(int*)current_node->value <= 0) return nnc_null;

    layers_len = (nnc_uint)*(int*)current_node->value;
    NNCListDeAlloc(current_node);

    layers = malloc(sizeof(NNCISerializedLayerType) * layers_len);
    for(int i = 0; i < layers_len; i++) layers[i] = NNCSerializedLayerTypeFromList_Destructive(&list);

    NNCListDeAllocAll(model_type);
    smodel = NNCSerializedModelAlloc(type, tag, trainer, optimizer, layers, layers_len);

    if(list != nnc_null) NNCListDeAllocAll(list);

    return smodel;
}

void NNCSerializedModelSaveToFile(NNCISerializedModelType model, char* file_name) {
    FILE *fp = fopen(file_name, "w");
    if (fp != NULL)
    {
        NNCIList cstr = NNCSerializedModelMinify(model);
        NNCIListCString ccstr = NNCListToCString(cstr, nnc_false, NNC_VALUE_EMPTY_BYTE, nnc_true);
        fputs(ccstr->string, fp);

        NNCListCStringDeAlloc(ccstr);
        NNCListDeAllocAll(cstr);

        fclose(fp);
    }
}

NNCISerializedModelType NNCSerializedModelLoadFromFile(char* file_name) {

    FILE *f = fopen(file_name, "r");
    fseek(f, 0, SEEK_END);
    long flen = ftell(f);
    fseek(f, 0, SEEK_SET);

    if(flen <= 0) return nnc_null;

    char *smodel = malloc(flen + 1);
    fread(smodel, flen, 1, f);
    fclose(f);

    smodel[flen] = nnc_end_of_string;

    NNCIListCString cclist = NNCListCStringAllocFromCString(smodel);
    NNCIList clist = NNCCStringToList(cclist, nnc_false, NNC_VALUE_END_BYTE, nnc_true);
    NNCListCStringDeAlloc(cclist);

    NNCISerializedModelType model = NNCSerializedModelMaxify_Destructive(clist);

    return model;
}

NNCIList NNCSerializedLayerTypeToList(NNCISerializedLayerType layer) {
    if(layer != nnc_null){
        NNCIList lclist = NNCListAllocChar(NNC_LAYER_START_BYTE);
        NNCIListCString cltype = NNCListCStringAllocFromCString(NNCModelLayerElementTypeToString[layer->type]);
        NNCListAppend(lclist, NNCCStringToList(cltype, nnc_false, NNC_VALUE_NOT_DEFINED, nnc_false));
        NNCListAppend(lclist, NNCListAllocChar(NNC_VALUE_END_BYTE));
        NNCListAppend(lclist, NNCListAllocInt((int)layer->len));
        NNCListAppend(lclist, NNCListAllocChar(NNC_VALUE_END_BYTE));
        if(layer->layer != nnc_null) NNCListAppend(lclist, NNCListAllocCopy(layer->layer));
        else NNCListAppendChar(lclist, NNC_VALUE_NOT_DEFINED);
        NNCListAppend(lclist, NNCListAllocChar(NNC_LAYER_END_BYTE));
        free(cltype);
        return lclist;
    }
    else {
        NNCIList lclist = NNCListAllocChar(NNC_LAYER_START_BYTE);
        NNCIListCString cltype = NNCListCStringAllocFromCString(NNCModelLayerElementTypeToString[NNCLayerType_NONE]);
        NNCListAppend(lclist, NNCCStringToList(cltype, nnc_false, NNC_VALUE_NOT_DEFINED, nnc_false));
        NNCListAppend(lclist, NNCListAllocChar(NNC_VALUE_END_BYTE));
        NNCListAppend(lclist, NNCListAllocInt(0));
        NNCListAppend(lclist, NNCListAllocChar(NNC_VALUE_END_BYTE));
        NNCListAppend(lclist, NNCListAllocChar(NNC_VALUE_EMPTY_BYTE));
        NNCListAppend(lclist, NNCListAllocChar(NNC_LAYER_END_BYTE));
        free(cltype);
        return lclist;
    }
}

NNCISerializedLayerType NNCSerializedLayerTypeFromList_Destructive(NNCIList* list) {
    if(list == nnc_null) return nnc_null;

    nnc_bool take = nnc_false;
    NNCISerializedLayerType slayer = NNCSerializedLayerTypeAlloc(NNCListAllocInt(0), 1, NNCLayerType_NONE);
    NNCIList current_node = NNCListPop(list);

    while(current_node != nnc_null){

        if(current_node->type == CHAR && *(char*)current_node->value == NNC_LAYER_END_BYTE) {
            NNCListDeAlloc(current_node);
            break;
        }
        else if(current_node->type == CHAR && *(char*)current_node->value == NNC_LAYER_START_BYTE) {
            NNCListDeAlloc(current_node);
            take = nnc_true;
        }
        else if(take == nnc_true) {
            NNCListAppend(slayer->layer, current_node);
            slayer->len += 1;
        }
        else return nnc_null;

        current_node = NNCListPop(list);
    }

    current_node = NNCListPop(&(slayer->layer));
    NNCListDeAlloc(current_node);

    return slayer;
}

void NNCSerializedModelPrint(NNCISerializedModelType model) {
//    NNCIListCString cstr1 = NNCListToCString(model->tag, nnc_false, NNC_VALUE_EMPTY_BYTE, nnc_false);
//    dprintf("\n%s\n", cstr1->string);
//    NNCListCStringDeAlloc(cstr1);
//
//    NNCIListCString cstr2 = NNCListToCString(model->trainer->layer, nnc_false, NNC_VALUE_EMPTY_BYTE, nnc_false);
//    dprintf("\n%s\n", cstr2->string);
//    NNCListCStringDeAlloc(cstr2);
//
//    NNCIListCString cstr3 = NNCListToCString(model->optimizer->layer, nnc_false, NNC_VALUE_EMPTY_BYTE, nnc_false);
//    dprintf("\n%s\n", cstr3->string);
//    NNCListCStringDeAlloc(cstr3);

//    for(int i = 0; i < model->layers_len; i++) {
//        NNCIListCString cstr4 = NNCListToCString(model->layers[i]->layer, nnc_false, NNC_VALUE_EMPTY_BYTE, nnc_false);
////        dprintf("\n%s\n", cstr4->string);
////        NNCListCStringDeAlloc(cstr4);
//    }


    NNCIListCString cstr4 = NNCListToCString(model->layers[0]->layer, nnc_false, NNC_VALUE_EMPTY_BYTE, nnc_false);
    dprintf("\n%s\n", cstr4->string);
    cstr4 = NNCListToCString(model->layers[1]->layer, nnc_false, NNC_VALUE_EMPTY_BYTE, nnc_false);
    dprintf("\n%s\n", cstr4->string);
    cstr4 = NNCListToCString(model->layers[2]->layer, nnc_false, NNC_VALUE_EMPTY_BYTE, nnc_false);
    dprintf("\n%s\n", cstr4->string);
    cstr4 = NNCListToCString(model->layers[3]->layer, nnc_false, NNC_VALUE_EMPTY_BYTE, nnc_false);
    dprintf("\n%s\n", cstr4->string);
    cstr4 = NNCListToCString(model->layers[4]->layer, nnc_false, NNC_VALUE_EMPTY_BYTE, nnc_false);
    dprintf("\n%s\n", cstr4->string);
    cstr4 = NNCListToCString(model->layers[5]->layer, nnc_false, NNC_VALUE_EMPTY_BYTE, nnc_false);
    dprintf("\n%s\n", cstr4->string);
}

