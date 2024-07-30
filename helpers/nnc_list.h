#ifndef CMATRIX_NNC_LIST_H
#define CMATRIX_NNC_LIST_H

#include "nnc_config.h"
#include "nnc_matrix.h"

#define NNCIList NNCList*
//#define NNCIListNode NNCListNode*
#define NNCIListCString NNCListCString*

// todo
#define NNCLIST_DOUBLY_LINKED 1
//#define NNCLIST_FAST_APPEND 1

enum NNCListValueType {
    VOID,
    MTYPE,
    CHAR,
    FLOAT,
    INT
};

typedef struct NNCListCString {
    char*   string;
    int     len;
}NNCListCString;

NNCIListCString NNCListCStringAlloc();
NNCIListCString NNCListCStringAllocFromCString(char* cstring);
void NNCListCStringDeAlloc(NNCIListCString cstring);

//typedef struct NNCList {
//    struct NNCIListNode first;
//    struct NNCIListNode last;
//    int len;
//}NNCList;
//
//typedef struct NNCListNode {
//    void*                   value;
//    enum NNCListValueType   type;
//    struct NNCIList         next;
//#if NNCLIST_DOUBLY_LINKED == 1
//    struct NNCIList         prev;
//#endif
//}NNCListNode;

typedef struct NNCList {
    void*                   value;
    enum NNCListValueType   type;
    struct NNCIList         next;
#if NNCLIST_DOUBLY_LINKED == 1
    struct NNCIList         prev;
#endif
}NNCList;

NNCIList NNCListAlloc(void* value);
NNCIList NNCListAllocChar(char value);
NNCIList NNCListAllocFloat(float value);
NNCIList NNCListAllocInt(int value);
NNCIList NNCListAllocMType(nnc_mtype value);
NNCIList NNCListAllocCopy(NNCIList list);

void NNCListDeAlloc(NNCIList node);
void NNCListDeAllocNextAll(NNCIList node);
void NNCListDeAllocPrevAll(NNCIList node);
void NNCListDeAllocAll(NNCIList node);

NNCIList NNCListGetLastNode(NNCIList node);
NNCIList NNCListGetFirstNode(NNCIList node);

void NNCListInsertAfter(NNCIList head, NNCIList node);
void NNCListInsertAfterChar(NNCIList head, char value);
void NNCListInsertAfterFloat(NNCIList head, float value);
void NNCListInsertAfterInt(NNCIList head, int value);
void NNCListInsertAfterMType(NNCIList head, nnc_mtype value);

void NNCListInsertBefore(NNCIList head, NNCIList node);
void NNCListInsertBeforeChar(NNCIList head, char value);
void NNCListInsertBeforeFloat(NNCIList head, float value);
void NNCListInsertBeforeInt(NNCIList head, int value);
void NNCListInsertBeforeMType(NNCIList head, nnc_mtype value);

void NNCListAppend(NNCIList head, NNCIList node);
void NNCListAppendChar(NNCIList head, char value);
void NNCListAppendFloat(NNCIList head, float value);
void NNCListAppendInt(NNCIList head, int value);
void NNCListAppendMType(NNCIList head, nnc_mtype value);

int NNCListLengthToLastNode(NNCIList node);
int NNCListLengthToFirstNode(NNCIList node);
int NNCListLength(NNCIList node);




NNCIList NNCMatrixTypeToList(NNCIMatrixType matrix);
NNCIListCString NNCListToCString(NNCIList node, nnc_bool use_delimiter, char delimiter, nnc_bool minified);
NNCIList NNCCStringToList(NNCIListCString str, nnc_bool use_delimiter, char delimiter, nnc_bool minified);


//char* nodesToString(Node*);
//void appendNodeAtEnd(Node*, Node*);
//Node* cstrToNode(char*);
//Node* floatToStrNode(float);
//Node* intToStrNode(int val);

#endif //CMATRIX_NNC_LIST_H
