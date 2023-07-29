#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[]) {

	char x[2][4] = {{0,0,1,1},{0,1,0,1}};
	printf("Patron:\n");
	printf("\tx_1\tx_2\n");
	for(char i = 0; i < 4; ++i) {
		for(char j = 0; j < 2; ++j)
			printf("\t%d", x[j][i]);
		printf("\n");
	}
	printf("\n");

	char expected[4] = {0,0,0,1};
	printf("Valores deseados:\n");
	for(char i = 0; i < 4; ++i)
		printf("\t%d\n", expected[i]);

	printf("\n");

	float w[3] = {1,0,-0.5};
	printf("Pesos:\n");
	for(char i = 0; i < 3; ++i)
		printf("\t%.1f\n", w[i]);
	printf("\n");

	float sum[4] = {0,0,0,0};
	printf("Suma:\n");
	for(char i = 0; i < 4; ++i) {
		sum[i] = x[0][i]*w[0] + x[1][i]*w[1] + w[2];
		printf("\t%.1f\n", sum[i]);
	}
	printf("\n");
	
	float activationFunction[4] = {0,0,0,0};
	printf("Funcion de activacion:\n");
	for(char i = 0; i < 4; ++i) {
		if(sum[i] >=0)
			activationFunction[i] = 1;
		printf("\t%.1f\n", activationFunction[i]);
	}
	printf("\n");

	float err[4] = {0,0,0,0};
	printf("Error:\n");
	for(char i = 0; i < 4; ++i) {
		err[i] = expected[i] - activationFunction[i];
		printf("\t%.1f\n", err[i]);
	}
	printf("\n");

	char numOfErr = 0;
	for(char i = 0; i < 4; ++i)
		if(err[i] != 0)
			++numOfErr;
	printf("Error(es) = %d\n", numOfErr);

	return 0;
}
