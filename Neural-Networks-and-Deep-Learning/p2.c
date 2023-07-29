#include <stdio.h>
#include <stdlib.h>
#include <math.h>

char x[3][4] = {{0,0,1,1},{0,1,0,1},{1,1,1,1}};
// char y[4] = {0,0,0,1}; // AND
char y[4] = {0,1,1,1}; // OR
// char y[4] = {0,1,1,0}; // XOR
float w[3] = {1.0,0.0,1.0};
const float learningRate = 0.1;
float err = 0.0;

void printWeights(int epoch) {
	printf("Weights epoch %d\n", epoch);
	for (char i = 0; i < 3; ++i) {
		printf("\t%.5f", w[i]);
		if (i == 2)
			printf(" <--- Bias\n");
		else
			printf(" <--- w_%d\n", i+1);
	}
	printf("\n");
}

void initialPrompt() {
	printf("Patron:\n");
	printf("\tx_1\tx_2\texpected\n");
	for (char i = 0; i < 4; ++i) {
		for (char j = 0; j < 2; ++j)
			printf("\t%d", x[j][i]);
		printf("\t%d\n", y[i]);
	}
	printf("\n");
	printWeights(0);
}

float sigmoidActivationFunction(float x) {
	return 1/(1 + exp(-x));
}

float perceptron(char row) {
	float sum = 0.0;
	for (char i = 0; i < 3; ++i)
		sum += w[i] * x[i][row];
	return sigmoidActivationFunction(sum);
}

void weightUpdate(char row) {
	for (char i = 0; i < 3; ++i)
		w[i] += learningRate * err * x[i][row];
}

float meanSquareError(float *y_pred) {
	float mse = 0.0;
	for (char i = 0; i < 4; ++i)
		mse += pow(y[i] - y_pred[i], 2);
	return mse/2;
}

void finalPrompt(float *y_pred, int lastEpoch) {
	printf("MSE surpassed at %d!\n\n", lastEpoch);

	printf("Final linear equation\n");
	printf("(%.5f)*x_1 + (%.5f)*x_2 + (%.5f) = y\n\n", w[0], w[1], w[2]);
	printf("\tPredicted y\n");
	for (char i = 0; i < 4; ++i)
		printf("\t%.5f\n", y_pred[i]);
}

int main(int argc, char const *argv[]) {

	initialPrompt();

	float y_pred[4] = {0.0,0.0,0.0,0.0};
	int lastEpoch = 0;

	for (int epoch = 0; ; ++epoch) {
		// printWeights(epoch+1);
		for (char i = 0; i < 4; ++i) {
			y_pred[i] = perceptron(i);
			err = y[i] - y_pred[i];
			weightUpdate(i);
		}
		if (meanSquareError(y_pred) < 0.001) {
			lastEpoch = epoch;
			break;
		}
	}

	finalPrompt(y_pred, lastEpoch);

	return 0;
}
