#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

int max2(int a, int b) {
	return (a > b) ? a : b;
}

int max(int a, int b, int c) {
	return max2(max2(a, b), c);
}

int maxSumcrus(int arr[], int l, int m, int h) {
	int sum = 0;
	int suma_izq = INT_MIN;
	for (int i = m; i >= l; --i) {
		sum += arr[i];
		if (sum > suma_izq) suma_izq = sum;
	}
	sum = 0;
	int suma_der = INT_MIN;
	for (int i = m + 1; i <= h; ++i) {
		sum += arr[i];
		if (sum > suma_der) suma_der = sum;
	}
	return max(suma_izq + suma_der, suma_izq, suma_der);
}

int sumMaxsubarr(int arr[], int l, int h) {
	if (l == h) return arr[l];
	int m = (l + h) / 2;

	return max(sumMaxsubarr(arr, l, m), sumMaxsubarr(arr, m + 1, h), maxSumcrus(arr, l, m, h));
}

void printArray(int array[], int size) {
	for (int i = 0; i < size; ++i) printf("%d  ", array[i]);
	printf("\n");
}

int main(int argc, char const *argv[]) {
	int n = 0;

	printf("Ingrese el tamaño del arreglo:\n\t");
	scanf("%d", &n);

	int arr[n];
	for(int i = 0; i < n; ++i) arr[i] = rand() % 99 - 50;

	printf("El arreglo es:\n\t");
	printArray(arr, n);

	int max_sum = sumMaxsubarr(arr, 0, n - 1);
	printf("La suma contigua maxima del arreglo es: %d\n", max_sum);

	return 0;
}
