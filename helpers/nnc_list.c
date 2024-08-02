#include <string.h>
#include "nnc_list.h"
#include "nnc_config.h"

NNCIList NNCListAlloc(void *value) {
    NNCIList node = malloc(sizeof(NNCList));

    node->next = nnc_null;
    node->prev = nnc_null;
    node->type = VOID;
    node->value = value;

    return node;
}

NNCIList NNCListAllocChar(char value) {
    NNCIList node = malloc(sizeof(NNCList));

    node->next = nnc_null;
    node->prev = nnc_null;
    node->type = CHAR;
    node->value = malloc(sizeof(char));
    *(char *)node->value = value;

    return node;
}

NNCIList NNCListAllocFloat(float value) {
    NNCIList node = malloc(sizeof(NNCList));

    node->next = nnc_null;
    node->prev = nnc_null;
    node->type = FLOAT;
    node->value = malloc(sizeof(int));
    *(float *)node->value = value;

    return node;
}

NNCIList NNCListAllocInt(int value) {
    NNCIList node = malloc(sizeof(NNCList));

    node->next = nnc_null;
    node->prev = nnc_null;
    node->type = INT;
    node->value = malloc(sizeof(int));
    *(int *)node->value = value;

    return node;
}

NNCIList NNCListAllocMType(nnc_mtype value) {
    NNCIList node = malloc(sizeof(NNCList));

    node->next = nnc_null;
    node->prev = nnc_null;
    node->type = MTYPE;
    node->value = malloc(sizeof(nnc_mtype));
    *(nnc_mtype *)node->value = value;

    return node;
}

void NNCListDeAlloc(NNCIList node) {
    if(node != nnc_null){
        if(node->value != nnc_null) free(node->value);
        free(node);
    }
}

void NNCListDeAllocNextAll(NNCList *node) {
    if(node != nnc_null){
        NNCListDeAllocNextAll(node->next);
        NNCListDeAlloc(node);
    }
}

void NNCListDeAllocPrevAll(NNCList *node) {
    if(node != nnc_null){
        NNCListDeAllocPrevAll(node->next);
        NNCListDeAlloc(node);
    }
}

void NNCListDeAllocAll(NNCIList node) {
    if(node != nnc_null){
        NNCListDeAllocNextAll(node->next);
        NNCListDeAllocPrevAll(node->prev);
        NNCListDeAlloc(node);
    }
}

NNCIList NNCListGetLastNode(NNCIList node) {
#if NNCLIST_ITERATIVE == 1
    while(node->next != nnc_null) node = node->next;
    return node;
#endif
#if NNCLIST_RECURSIVE == 1 && NNCLIST_ITERATIVE == 0
    if(node->next == nnc_null) return node;
    else return NNCListGetLastNode(node->next);
#endif
}

NNCIList NNCListGetFirstNode(NNCIList node) {

#if NNCLIST_ITERATIVE == 1
    while(node->prev != nnc_null) node = node->prev;
    return node;
#endif
#if NNCLIST_RECURSIVE == 1 && NNCLIST_ITERATIVE == 0
    if(node->prev == nnc_null) return node;
    else return NNCListGetFirstNode(node->prev);
#endif
}

void NNCListInsertAfter(NNCIList head, NNCIList node) {
    NNCIList node_first = NNCListGetFirstNode(node);
    NNCIList node_last = NNCListGetLastNode(node);

    node_last->next = head->next;
    if(head->next != nnc_null) head->next->prev = node_last;

    node_first->prev = head;
    head->next = node_first;
}

void NNCListInsertAfterChar(NNCIList head, char value) {
    NNCListInsertAfter(head, NNCListAllocChar(value));
}

void NNCListInsertAfterFloat(NNCIList head, float value) {
    NNCListInsertAfter(head, NNCListAllocFloat(value));
}

void NNCListInsertAfterInt(NNCIList head, int value) {
    NNCListInsertAfter(head, NNCListAllocInt(value));
}

void NNCListInsertAfterMType(NNCIList head, nnc_mtype value) {
    NNCListInsertAfter(head, NNCListAllocMType(value));
}

void NNCListInsertBefore(NNCIList head, NNCIList node) {
    if(head->prev == nnc_null) head->prev = node;
    else {
        NNCIList head_prev = head->prev;
        head->prev = node;
        NNCListGetFirstNode(node)->prev = head_prev;
    }
}

void NNCListInsertBeforeChar(NNCIList head, char value) {
    NNCListInsertBefore(head, NNCListAllocChar(value));
}

void NNCListInsertBeforeFloat(NNCIList head, float value) {
    NNCListInsertBefore(head, NNCListAllocFloat(value));
}

void NNCListInsertBeforeInt(NNCIList head, int value) {
    NNCListInsertBefore(head, NNCListAllocInt(value));
}

void NNCListInsertBeforeMType(NNCIList head, nnc_mtype value) {
    NNCListInsertBefore(head, NNCListAllocMType(value));
}

void NNCListAppend(NNCIList head, NNCIList node) {
    if(head->next == nnc_null) {
        head->next = node;
        node->prev = head;
    }
    else {
        node->prev = NNCListGetLastNode(head);
        node->prev->next = node;
    }
}

void NNCListAppendChar(NNCIList head, char value) {
    NNCListAppend(head, NNCListAllocChar(value));
}

void NNCListAppendFloat(NNCIList head, float value) {
    NNCListAppend(head, NNCListAllocFloat(value));
}

void NNCListAppendInt(NNCIList head, int value) {
    NNCListAppend(head, NNCListAllocInt(value));
}

void NNCListAppendMType(NNCIList head, nnc_mtype value) {
    NNCListAppend(head, NNCListAllocMType(value));
}

int NNCListLengthToLastNode(NNCIList node) {
    if(node->next == nnc_null) return 1;
    else return 1 + NNCListLengthToLastNode(node->next);
}

int NNCListLengthToFirstNode(NNCIList node) {
    if(node->prev == nnc_null) return 1;
    else return 1 + NNCListLengthToFirstNode(node->prev);
}

int NNCListLength(NNCIList node) {
    if(node == nnc_null) return 0;
    else if(node->next == nnc_null && node->prev == nnc_null) return 1;
    else return NNCListLengthToLastNode(node) + NNCListLengthToFirstNode(node) - 1;
}

NNCIListCString NNCListToCString(NNCIList node, nnc_bool use_delimiter, char delimiter, nnc_bool minified) {
    NNCIListCString lcstr = malloc(sizeof(NNCListCString));

    lcstr->string = nnc_null;
    lcstr->len = 0;

    int prefix_len = 0;
    if(minified == 1) prefix_len = 1;

    if(node == nnc_null) return lcstr;
    if(node->value == nnc_null) return lcstr;

    NNCIList currentNode = NNCListGetFirstNode(node);

    while(1){
        if(currentNode == nnc_null) break;

        int tmp_len = 0;
        char* tmp_str = nnc_null;

        if(currentNode->type == INT){
            tmp_len = snprintf(NULL, 0, "%d", *((int*)currentNode->value));
            if(tmp_len == 1) tmp_len = 2;
            tmp_len += prefix_len;
            tmp_str = malloc(tmp_len);
            snprintf(&tmp_str[prefix_len], tmp_len, "%d", *((int*)currentNode->value));
            if(minified == 1) tmp_str[0]='I';
        } else if(currentNode->type == FLOAT){
            tmp_len = snprintf(NULL, 0, "%f", *((float*)currentNode->value));
            if(tmp_len == 1) tmp_len = 2;
            tmp_len += prefix_len;
            tmp_str = malloc(tmp_len);
            snprintf(&tmp_str[prefix_len], tmp_len, "%f", *((float*)currentNode->value));
            if(minified == 1) tmp_str[0]='F';
        } else if(currentNode->type == CHAR){
            tmp_len = 2;
            tmp_len += prefix_len;
            tmp_str = malloc(tmp_len);
            tmp_str[0 + prefix_len] = *((char*)currentNode->value);
            tmp_str[1 + prefix_len] = nnc_end_of_string;
            if(minified == 1) tmp_str[0]='C';
        } else if(currentNode->type == MTYPE){ // FIXME float != nnc_mtype
            tmp_len = snprintf(NULL, 0, "%f", *((nnc_mtype*)currentNode->value));
            if(tmp_len == 1) tmp_len = 2;
            tmp_len += prefix_len;
            tmp_str = malloc(tmp_len);
            snprintf(tmp_str + prefix_len, tmp_len, "%f", *((nnc_mtype*)currentNode->value));
            if(minified == 1) tmp_str[0]='M';
        }

        if(use_delimiter == 1) tmp_str[tmp_len-1] = delimiter;
        else tmp_len -= 1;


        if(lcstr->string == nnc_null){
            lcstr->string = tmp_str;
            lcstr->len = tmp_len;
        }else { // TODO implement realloc
            char* tmp_cpy = malloc(lcstr->len + tmp_len);

            memcpy(tmp_cpy, lcstr->string, lcstr->len);
            memcpy(&tmp_cpy[lcstr->len], tmp_str, tmp_len);

            free(lcstr->string);
            free(tmp_str);

            lcstr->string = tmp_cpy;
            lcstr->len += tmp_len;
        }

        currentNode = currentNode->next;
    }

    lcstr->string[lcstr->len] = nnc_end_of_string;
    return lcstr;
}

NNCIList NNCMatrixTypeToList(NNCIMatrixType matrix, nnc_bool use_delimiter, char delimiter) {
    NNCIList lmatrix = NNCListAllocChar(NNC_MODEL_SERIALIZED_MATRIX_START);
    if(use_delimiter == nnc_true) NNCListAppendChar(lmatrix, delimiter);

    NNCListAppend(lmatrix, NNCListAllocInt((int)matrix->x));
    if(use_delimiter == nnc_true) NNCListAppendChar(lmatrix, delimiter);
    NNCListAppend(lmatrix, NNCListAllocInt((int)matrix->y));
    if(use_delimiter == nnc_true) NNCListAppendChar(lmatrix, delimiter);

    for(int _y = 0; _y < matrix->y; _y ++){
        for(int _x = 0; _x < matrix->x; _x ++){
            NNCListAppend(lmatrix, NNCListAllocMType(matrix->matrix[_y][_x]));
            if(use_delimiter == nnc_true) NNCListAppendChar(lmatrix, delimiter);
        }
    }

    NNCListAppendChar(lmatrix, NNC_MODEL_SERIALIZED_MATRIX_END);

    return lmatrix;
}

NNCIListCString NNCListCStringAlloc() {
    NNCIListCString lcstring = malloc(sizeof(NNCListCString));

    lcstring->string = nnc_null;
    lcstring->len = 0;

    return lcstring;
}

NNCIListCString NNCListCStringAllocFromCString(char* cstring) {
    NNCIListCString lcstring = malloc(sizeof(NNCListCString));

    lcstring->string = cstring;
    lcstring->len = 0;

    while(true){
        lcstring->len += 1;
        if(cstring[lcstring->len] == nnc_end_of_string) break;
    }

    return lcstring;
}

void NNCListCStringDeAlloc(NNCIListCString cstring) {
    free(cstring->string);
    free(cstring);
}

// trenutno mora biti minified i bez delimitera
NNCIList NNCCStringToList(NNCIListCString str, nnc_bool use_delimiter, char delimiter, nnc_bool minified) {

    if(minified == nnc_false) {
        NNCIList clist = NNCListAllocChar(str->string[0]);
        for(int i = 1; i < str->len; i ++){
            NNCListAppend(clist, NNCListAllocChar(str->string[i]));
        }
        return clist;
    }

    NNCIList clist = NNCListAllocChar(str->string[0]);

    for(int i = 0; i < str->len; i ++){

        if(str->string[i] == 'I'){
            i += 1;
            int value = 0;
            nnc_bool is_negative = nnc_false;

            while(i < str->len && str->string[i] != 'I' && str->string[i] != 'C' && str->string[i] != 'F' && str->string[i] != 'M' && str->string[i] != '\0'){
                if(str->string[i] == '-'){
                    is_negative = nnc_true;
                    i += 1;
                } else{
                    value *= 10;
                    value += (int)(str->string[i] - '0');
                    i += 1;
                }
            }

            if(is_negative == nnc_true) value = value * -1;

            NNCListAppendInt(clist, value);
        }
        else if(str->string[i] == 'C'){
            NNCListAppendChar(clist, str->string[i+1]);
            i += 1;
        }
        else if(str->string[i] == 'F'){
            i += 1;
            int value = 0;
            float decimal = 0;
            int value_index = 0;
            nnc_bool is_negative = nnc_false;
            nnc_bool is_decimal = nnc_false;

            while(i < str->len && str->string[i] != 'I' && str->string[i] != 'C' && str->string[i] != 'F' && str->string[i] != 'M' && str->string[i] != '\0'){
                if(str->string[i] == '-'){
                    is_negative = nnc_true;
                    i += 1;
                } else if(str->string[i] == '.') {
                    is_decimal = nnc_true;
                    value_index = -1;
                    i += 1;
                } else{
                    if(is_decimal == nnc_false){
                        value *= 10;
                        value += (int)(str->string[i] - '0');
                    }
                    else{
                        float tmp = (float)(str->string[i] - '0');
                        for(int ti = 0; ti > value_index; ti--) tmp /= 10;
                        decimal += tmp;
                        value_index -= 1;
                    }
                    i += 1;
                }
            }

            decimal += (float)value;
            if(is_negative == nnc_true) decimal = decimal * -1;

            NNCListAppendFloat(clist, decimal);
        }
        else if(str->string[i] == 'M'){
            i += 1;
            int value = 0;
            float decimal = 0;
            int value_index = 0;
            nnc_bool is_negative = nnc_false;
            nnc_bool is_decimal = nnc_false;

            while(i < str->len && (str->string[i] != 'I' || str->string[i] != 'C' || str->string[i] != 'F' || str->string[i] != 'M' || str->string[i] != '\0')){
                if(str->string[i] == '-'){
                    is_negative = nnc_true;
                    i += 1;
                } else if(str->string[i] == '.') {
                    is_decimal = nnc_true;
                    value_index = -1;
                    i += 1;
                } else{
                    if(is_decimal == nnc_false){
                        value *= 10;
                        value += (int)(str->string[i] - '0');
                    }
                    else{
                        float tmp = (float)(str->string[i] - '0');
                        for(int ti = 0; ti > value_index; ti--) tmp /= 10;
                        decimal += tmp;
                        value_index -= 1;
                    }
                    i += 1;
                }
            }

            decimal += (float)value;
            if(is_negative == nnc_true) decimal = decimal * -1;

            NNCListAppendFloat(clist, decimal);
        }
    }


    return clist;
}

NNCIList NNCListAllocCopy(NNCIList list) {
    NNCIList head = NNCListAlloc(nnc_null);

    while(list != nnc_null){

        if(list->type == VOID) NNCListAppend(head, NNCListAlloc(list->value));
        if(list->type == CHAR) NNCListAppendChar(head, *((char*)list->value));
        if(list->type == INT) NNCListAppendInt(head, *((int*)list->value));
        if(list->type == FLOAT) NNCListAppendFloat(head, *((float *)list->value));
        if(list->type == MTYPE) NNCListAppendMType(head, *((nnc_mtype*)list->value));

        list = list->next;
    }

    NNCIList lst = head->next;
    lst->prev = nnc_null;
    head->next = nnc_null;
    NNCListDeAlloc(head);

    return lst;
}

NNCIList NNCListSplit(NNCIList head, int index) {
    NNCIList head_b = NNCListAtIndex(head, index);
    head_b->prev->next = nnc_null;
    head_b->prev = nnc_null;
    return head_b;
}

NNCIList NNCListAtIndex(NNCIList head, int index) {
    NNCIList head_b = nnc_null;

#if NNCLIST_ITERATIVE == 1
    for(int i = 0; i < index && head->next != nnc_null; i++) head = head->next;
    head_b = head;
#endif
#if NNCLIST_RECURSIVE == 1 && NNCLIST_ITERATIVE == 0
//    if(node->next == nnc_null) return node;
//    else return NNCListGetLastNode(node->next);
#endif

    return head_b;
}

NNCIList NNCListPop(NNCIList* head) {
    if((*head) == nnc_null) return nnc_null;
    if((*head)->next == nnc_null) {
        NNCIList poped = (*head);
        (*head) = nnc_null;
        return poped;
    }
    NNCIList poped = (*head);
    (*head) = (*head)->next;
    (*head)->prev = nnc_null;
    poped->next = nnc_null;
    return poped;
}
