#include <stdio.h>
#include <stdlib.h>

char str[100], *path;
char size, x, num = 0;

char *states, *sigmaM, s, *fS, *deltaTab;

char* getFromFile(const char *nameOfFile) {
	FILE *textFile;
	char *text;
	long numbytes;

	textFile = fopen(nameOfFile, "r");
	if(textFile == NULL)
		return 0;

	fseek(textFile, 0L, SEEK_END);
	numbytes = ftell(textFile);
	fseek(textFile, 0L, SEEK_SET);

	text = (char*) calloc(numbytes, sizeof(char));

	fread(text, sizeof(char), numbytes, textFile);
	fclose(textFile);

	return text;
}

void printAFN() {
	printf("\nEstados =");
	for(char i = 0; states[i]; ++i) printf(" %c", states[i]);
	
	printf("\nAlfabeto =");
	for(char i = 0; sigmaM[i]; ++i) printf(" %c",sigmaM[i]);

	printf("\nEstado inicial = %c", s);

	printf("\nEstado final =");
	for(char i = 0; fS[i]; ++i) printf(" %c",fS[i]);

	printf("\nDelta:\t");
	for(char i = 0; deltaTab[i]; ++i)
		if((i+1)%3 == 0)
			printf(" %c\n\t", deltaTab[i]);
		else
			printf(" %c", deltaTab[i]);
}

char* getInfo(char *text, char *i) {
	char auxSize;
	for (auxSize = *i; text[auxSize] != 10; ++auxSize);

	auxSize = (auxSize - *i)/2;
	char *obj = (char*) malloc((auxSize+1) * sizeof(char));
	obj[auxSize] = '\0';

	for(char aux = 0; text[*i] != 10; *i+=2, ++aux)
		obj[aux] = text[*i];

	return obj;
}

char getAFN(const char *nameOfFile) {
	char *text = getFromFile(nameOfFile);
	if(!text)
		return 1;

	char auxSize, aux, i = 0;

	states = getInfo(text, &i); // states
	++i;
	sigmaM = getInfo(text, &i); // alphabet
	s = text[++i]; // initial state
	i+=3;
	fS = getInfo(text, &i); // final states

	// delta table
	for (aux = ++i, x = 1; text[aux]; ++aux) if(text[aux] == 10) ++x;
	deltaTab = (char *) malloc((x*3+1)*sizeof(char));

	char j;
	for(j = 0, aux = i; text[aux]; ++aux)
		if(text[aux] != 10 && text[aux] != 13 && text[aux] != ',') {
			deltaTab[j] = text[aux];
			++j;
		}
	deltaTab[j] = '\0';

	printAFN();

	return 0;
}

char isFinalState(char state) {
	for(char i = 0; fS[i]; ++i)
		if(state == fS[i])
			return 1;
	return 0;
}

void solve(char state, char deltaPot) {
	if(str[deltaPot]) {
		for(char aux = 0; aux < x; ++aux)
			if(state == deltaTab[aux*3] && str[deltaPot] ==  deltaTab[aux*3+1]) {
				path[deltaPot] = deltaTab[aux*3+2];
				solve(deltaTab[aux*3+2], deltaPot + 1);
			}
	}
	else if(deltaPot == size && isFinalState(state)) {
		printf("Camino Valido:\n\t--> %c", s);
		for(char i = 0; i<size; ++i)
			printf(" -%c-> %c", str[i], path[i]);
		printf("\n");
		++num;
	}
}

int main(int argc, char const *argv[]) {
	printf("Usando el AFN '%s' con:", argv[argc-1]);
	char err = getAFN(argv[argc-1]);
	if(err)
		printf("\t(Err) No se encontro el archivo\n");
	else {
		printf("\nIngrese la cadena a verificar: ");
		fflush(stdin);
		scanf("%s", str);

		for(size = 0; str[size]; ++size);
		char auxPath[size];
		path = auxPath;

		solve(s, 0);

		printf("Caminos validos = %d\n", num);
	}
	return 0;
}
