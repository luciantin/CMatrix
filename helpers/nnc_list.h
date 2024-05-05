#ifndef CMATRIX_NNC_LIST_H
#define CMATRIX_NNC_LIST_H

// posudeno od https://github.com/Leyxargon/c-linked-list/tree/master

typedef struct node {
    char n;
    struct node *next;
}Node;

Node* newNode(char );

Node* insert(Node *, char ); /* inserts a new item at the end of the list */

Node* deleteList(Node *); /* deletes a list */

int countNodes(Node *); /* returns the number of nodes in the list */

char* nodesToString(Node*);

void appendNodeAtEnd(Node*, Node*);

Node* cstrToNode(char*);

Node* floatToStrNode(float);

Node* intToStrNode(int val);

#endif //CMATRIX_NNC_LIST_H
