#include <string.h>
#include "nnc_list.h"
#include "nnc_config.h"
#include "nnc_serializer.h"

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
    if(node->next == nnc_null) return node;
    else return NNCListGetLastNode(node->next);
}

NNCIList NNCListGetFirstNode(NNCIList node) {
    if(node->prev == nnc_null) return node;
    else return NNCListGetFirstNode(node->prev);
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
            tmp_str[1 + prefix_len] = '\0';
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

    lcstr->string[lcstr->len] = '\0';
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
        if(cstring[lcstring->len] == '\0') break;
    }

    return lcstring;
}

void NNCListCStringDeAlloc(NNCIListCString cstring) {
    free(cstring->string);
    free(cstring);
}

NNCIList NNCCStringToList(NNCIListCString str, nnc_bool use_delimiter, char delimiter, nnc_bool minified) {
    NNCIList clist = NNCListAllocChar(str->string[0]);
    for(int i = 1; i < str->len; i ++){
        NNCListAppend(clist, NNCListAllocChar(str->string[i]));
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



//NNC_ListNode *newNode(char x) {
//    Node *pnt = (Node *) malloc (sizeof(Node)); /* allocates physical memory for the node */
//    pnt -> n = x;                               /* inserts the information received as input in the field(s) in the list */
//    pnt -> next = nnc_null;                         /* initialize the pointer */
//    /* an insert function will take care of properly setting the next variable */
//    return pnt;
//}
//
//Node *insert(Node *top, char x) {
//    Node *tmp = newNode(x);						/* create a node with input information */
//    tmp -> next = top;							/* top corresponds to the first element of the list BEFORE this operation,
//												 * which provides to put "tmp" at the top of the list,
//												 * thus becoming the new "top" */
//    return tmp;									/* returns the new first element of the list */
//}
//
//Node *deleteList(Node *top) {
//    if (top != nnc_null) {				/* if not reached end of the list... */
//        deleteList(top -> next);	/* ...move on */
//        free(top);					/* delete the node */
//    }
//    else
//        return nnc_null;
//}
//
//int countNodes(Node *top) {
//    if (top == nnc_null)
//        return 0;
//    else
//        return 1 + countNodes(top -> next);
//}
//
//char* nodesToString(Node* node){
//    int len = countNodes(node);
//    char* str = malloc(sizeof(char) * len);
//
//    Node* current_node = node;
//    int index = 0;
//    while(current_node != nnc_null){
//        str[index] = current_node->n;
//        index += 1;
//        current_node = node->next;
//    }
//
//    return str;
//}
//
//void appendNodeAtEnd(Node* list, Node* end){
//    Node* current_node = list;
//    while(list->next != nnc_null){
//        current_node = list->next;
//    }
//    current_node->next = end;
//}
//
//Node* cstrToNode(char* str){
//    if(str != nnc_null && str[0] != '\0'){
//        int index = 0;
//
//        char current_char = str[index];
//        Node* head = newNode(current_char);
//        index += 1;
//
//        Node* current_node = head;
//        current_char = str[index];
//
//        while(current_char != '\0'){
//            current_node->next = newNode(current_char);
//            current_char = str[index];
//            index += 1;
//        }
//
//        return head;
//    } else{
//        return nnc_null;
//    }
//}
//
//Node* floatToStrNode(float val){
//    int tmp_len = snprintf(NULL, 0, "%f", val);
//    char* tmp_str = malloc(tmp_len + 1);
//    snprintf(tmp_str, tmp_len + 1, "%f", val);
//    Node* nval = cstrToNode(tmp_str);
//    free(tmp_str);
//    return nval;
//}
//
//Node* intToStrNode(int val){
//    int tmp_len = snprintf(NULL, 0, "%d", val);
//    char* tmp_str = malloc(tmp_len + 1);
//    snprintf(tmp_str, tmp_len + 1, "%d", val);
//    Node* nval = cstrToNode(tmp_str);
//    free(tmp_str);
//    return nval;
//}
