#include "nnc_list.h"
#include "nnc_config.h"

Node *newNode(char x) {
    Node *pnt = (Node *) malloc (sizeof(Node)); /* allocates physical memory for the node */
    pnt -> n = x;                               /* inserts the information received as input in the field(s) in the list */
    pnt -> next = nnc_null;                         /* initialize the pointer */
    /* an insert function will take care of properly setting the next variable */
    return pnt;
}

Node *insert(Node *top, char x) {
    Node *tmp = newNode(x);						/* create a node with input information */
    tmp -> next = top;							/* top corresponds to the first element of the list BEFORE this operation,
												 * which provides to put "tmp" at the top of the list,
												 * thus becoming the new "top" */
    return tmp;									/* returns the new first element of the list */
}

Node *deleteList(Node *top) {
    if (top != nnc_null) {				/* if not reached end of the list... */
        deleteList(top -> next);	/* ...move on */
        free(top);					/* delete the node */
    }
    else
        return nnc_null;
}

int countNodes(Node *top) {
    if (top == nnc_null)
        return 0;
    else
        return 1 + countNodes(top -> next);
}

char* nodesToString(Node* node){
    int len = countNodes(node);
    char* str = malloc(sizeof(char) * len);

    Node* current_node = node;
    int index = 0;
    while(current_node != nnc_null){
        str[index] = current_node->n;
        index += 1;
        current_node = node->next;
    }

    return str;
}

void appendNodeAtEnd(Node* list, Node* end){
    Node* current_node = list;
    while(list->next != nnc_null){
        current_node = list->next;
    }
    current_node->next = end;
}

Node* cstrToNode(char* str){
    if(str != nnc_null && str[0] != '\0'){
        int index = 0;

        char current_char = str[index];
        Node* head = newNode(current_char);
        index += 1;

        Node* current_node = head;
        current_char = str[index];

        while(current_char != '\0'){
            current_node->next = newNode(current_char);
            current_char = str[index];
            index += 1;
        }

        return head;
    } else{
        return nnc_null;
    }
}

Node* floatToStrNode(float val){
    int tmp_len = snprintf(NULL, 0, "%f", val);
    char* tmp_str = malloc(tmp_len + 1);
    snprintf(tmp_str, tmp_len + 1, "%f", val);
    Node* nval = cstrToNode(tmp_str);
    free(tmp_str);
    return nval;
}

Node* intToStrNode(int val){
    int tmp_len = snprintf(NULL, 0, "%d", val);
    char* tmp_str = malloc(tmp_len + 1);
    snprintf(tmp_str, tmp_len + 1, "%d", val);
    Node* nval = cstrToNode(tmp_str);
    free(tmp_str);
    return nval;
}
