#include <stdio.h>
#include <stdlib.h>

void printArray(int arre[], int n){
	for(int i = 0; i < n; ++i) printf("%d ", arre[i]);
	printf("\n");
}

void insertionSort(int arre[], int n) {
	int aux = 0;
	for (int i = 0, j = 0; j < n; ++j) {
		i = j-1;
		aux = arre[j];
		for (; i >= 0 && aux < arre[i]; --i) arre[i+1] = arre[i];
		arre[i+1] = aux;
	}
}

int main(int argc, char const *argv[]) {
	int n = 10000;
	int a[n];

	for (int i = 0; i < n; ++i) a[i] = rand()%99 + 1;

	printf("Teniendo el arreglo de %d:\n", n);
	printArray(a, n);

	insertionSort(a, n);
	printf("Ordenando por insercion:\n");
	printArray(a, n);

	return 0;
}
