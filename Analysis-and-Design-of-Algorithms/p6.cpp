#include <iostream>
#include <stdio.h>
#include <conio.h>
using namespace std;

const int maxsize = 100;
const int null = 0;

struct huff_node {
	char symbol;
	int value;
	huff_node *parent;
	char childtype;
};

typedef huff_node *ptr;
ptr node[maxsize];

void create(int k) {
	ptr t = new huff_node;

	cout << "introduce el elemento numero " << k+1 << ": ";
	cin >> t->symbol;

	cout << "introduce su valor: ";
	cin >> t->value;

	t->parent = null;
	node[k] = t;
}

void print(int k) {
	ptr t = node[k];
	char code[10];
	int size = 0;
	cout << t->symbol << " - ";

	for(; t->parent != null; ++size) {
		if (t->childtype == 'L')
			code[size] = '0';
		else
			code[size] = '1';

		t = t->parent;
	}

	for (int j = size-1; j >= 0; --j)
		cout << code[j];

	cout << endl;
}

void twosmall(ptr &p, ptr &q, int numnodes) {
	int min1 = 9999;
	int min2 = 9999;

	p = null;
	q = null;

	for (int i = 0; i < numnodes; ++i) {
		if (node[i]->parent == null) {
			if (node[i]->value < min1) {
				min2 = min1;
				min1 = node[i]->value;
				q = p;
				p = node[i];
			}
			else if (node[i]->value < min2) {
				min2 = node[i]->value;
				q = node[i];
			}
		}
	}
}

int main(int argc, char const *argv[]) {
	int numsymbols;
	ptr p, q, r;
	cout << "Introduce el numero de elementos: ";
	cin >> numsymbols;

	for (int i = 0; i < numsymbols; ++i)
		create(i);

	for (int j = numsymbols; j < 2*numsymbols - 1; ++j) {
		r = new huff_node;
		node[j] = r;
		r->parent = null;

		twosmall(p, q, j);  
		p->parent = r;
		q->parent = r;

		p->childtype = 'L';
		q->childtype = 'R';
		r->symbol = ' ';
		r->value = p->value + q->value;
	}

	cout << endl << endl;
	cout <<"elemento *-------* codigo: " << endl;

	for (int k = 0; k < numsymbols; ++k)
		print(k);

	getch();

	return 0;
}