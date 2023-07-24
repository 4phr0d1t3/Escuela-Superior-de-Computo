#include <stdio.h>
#include <stdlib.h>

char q[3][2] = {"q0", "q1", "q2"};
char sigmaM[] = "ab";
char s[] = "q0", fS[] = "q2";
char deltaTab[4][3][3] = {{"si", "a", "b"}, {"q0", "q1", "q0"}, {"q1", "q1", "q2"}, {"q2", "Ø", "Ø"}};

int strSize;
char str[100];

char checkInAlpha(char movement) {
	for(char i = 0; i < 2; ++i)
		if(movement == sigmaM[i])
			return 1;
	return 0;
}

char findCol(char movement) {
	if(checkInAlpha(movement)) {
		for(char col = 1; col < 3; ++col)
			if(deltaTab[0][col][0] == movement)
				return col;
		return -1;
	}
	else
		return -2;
}

char findRow(char state[]) {
	char flg = 0;
	for(char row = 1; row < 4; ++row) {
		for(char i = 0; state[i]; ++i)
			if(deltaTab[row][0][i] == state[i])
				flg++;
			else
				flg--;
		if(flg == 2)
			return row;
	}
	return -1;
}

char check(char state[]) {
	for(char i = 0; state[i]; ++i)
		if(state[i] != fS[i])
			return 0;
	return 1;
}

void delta(char state[], int location) {
	printf(" -> %s / δ(%s,%c)", state, state, str[location]);

	char movement;
	movement = findCol(str[location]);

	if(movement != -2) {
		char whereAmI;
		whereAmI = findRow(state);
		if(movement >= 0 || whereAmI >= 0) {
			for(char i = 0; deltaTab[whereAmI][movement][i]; ++i)
				state[i] = deltaTab[whereAmI][movement][i];

			if(location < strSize-1)
				delta(state, location + 1);
			else {
				int flg = check(state);
				if(flg)
					printf(" -> %s\n\n\t--- Es valida ---\n\n", state);
				else
					printf(" -> %s\n\n\t--- No es valida ---\n\n", state);
			}
		}
		else
			printf(" -> Ø\n\n\t--- No es valida ---\n\n");
	}
	else
		printf(" -> Ø\n\n\t--- No es valida ---\n\n");
}

int main(int argc, char const *argv[]) {
	printf("El alfabeto es el siguiente: %s\n", sigmaM);

	printf("Ingrese la cadena: ");
	fflush(stdin);
	scanf("%s", str);
	for(strSize = 0; str[strSize]; ++strSize);

	printf("\nCamino:\n");
	delta(s, 0);

	return 0;
}
