#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int *b) {
	int t = *a;
	*a = *b;
	*b = t;
}

int partition(int array[], int low, int high) {
	int pivot = array[high];
	int i = (low - 1);

	for (int j = low; j < high; j++) {
		if (array[j] <= pivot) {
			i++;
			swap(&array[i], &array[j]);
		}
	}
	swap(&array[i + 1], &array[high]);
	return (i + 1);
}

void quickSort(int array[], int low, int high) {
	if (low < high) {
		int pi = partition(array, low, high);
		quickSort(array, low, pi - 1);
		quickSort(array, pi + 1, high);
	}
}

void printArray(int array[], int size) {
	for (int i = 0; i < size; ++i) printf("%d  ", array[i]);
	printf("\n");
}

int main(int argc, char const *argv[]) {
	int n = 0;

	printf("Ingrese el tamaño del arreglo:\n\t");
	scanf("%d", &n);

	int data[n];

	for(int i = 0; i < n; ++i) {
		data[i] = 1 + rand() % 99;
	}

	printf("Arreglo Desordenado:\n\t");
	printArray(data, n);

	quickSort(data, 0, n - 1);

	printf("Arreglo Ordenado:\n\t");
	printArray(data, n);

	return 0;
}
