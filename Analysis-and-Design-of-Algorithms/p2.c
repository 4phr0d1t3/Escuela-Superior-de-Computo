#include <stdio.h>
#include <stdlib.h>

void printArray(int arr[], int n) {
	for(int i = 0; i < n; i++) printf("%d ", arr[i]);
	printf("\n");
}

void merge(int arr[], int p, int q, int r) {
	int n1 = q - p + 1;
	int n2 = r - q;
	int L[n1], R[n2];

	int i = 0, j = 0;

	for(i = 0; i < n1; ++i) L[i] = arr[p + i];
	for(j = 0; j < n2; ++j) R[j] = arr[q + 1 + j];

	i = 0; j = 0;
	int k = p;

	for(; i < n1 && j < n2; ++k){
		if (L[i] <= R[j]) {
			arr[k] = L[i];
			i++;
		}
		else {
			arr[k] = R[j];
			j++;
		}
	}

	for(; i < n1; ++i, ++k) arr[k] = L[i];
	for(; j < n2; ++j, ++k) arr[k] = R[j];
}

void mergeSort(int arr[], int l, int r) {
	if(l < r) {
		int m = l + (r - l) / 2;
		mergeSort(arr, l, m);
		mergeSort(arr, m + 1, r);
		merge(arr, l, m, r);
	}
}

int main(int argc, char const *argv[]) {
	int n = 10000;
	int a[n];

	for (int i = 0; i < n; ++i) a[i] = rand()%99 + 1;

	printf("Teniendo el arreglo de %d:\n", n);
	printArray(a, n);

	mergeSort(a, 0, n - 1);
	printf("Arreglo ordenado: \n");
	printArray(a, n);
	
	return 0;
}
