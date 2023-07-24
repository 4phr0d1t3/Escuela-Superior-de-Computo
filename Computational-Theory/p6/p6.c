#include <stdio.h>
#include <stdlib.h>

char states, *sigmaM, s, fS, *deltaTab;
char rows, numSigmaM;

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

void printAInf() {
	printf("\nEstados = 1 - %d", states);
	
	printf("\nAlfabeto =");
	for(char i = 0; sigmaM[i]; ++i) printf(" %c",sigmaM[i]);

	printf("\nEstado inicial = %d", s);

	printf("\nEstado final = %d", fS);

	printf("\nDelta:\t");
	char flg = 1;
	for(char n = 0; deltaTab[n]; ++n)
		if(deltaTab[n] == ',')
			printf("\n\t");
		else if(deltaTab[n] == 32) {
			printf("-\t"); // ε
			flg = 2;
		}
		else if(flg != 0) {
			printf("%d\t", deltaTab[n]);
			--flg;
		}
		else {
			printf("%c\t", deltaTab[n]);
			flg = 2;
		}
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

char getAInf(const char *nameOfFile) {
	char *text = getFromFile(nameOfFile);
	if(!text) return 1;

	char auxSize, i = 3;
	int aux;

	// states
	if(text[1]!= 13) {
		states = (text[0]-48)*10 + (text[1]-48);
		++i;
	}
	else
		states = text[0]-48;

	numSigmaM = i;
	sigmaM = getInfo(text, &i); // alphabet
	numSigmaM = ((numSigmaM*-1)+i)/2;

	s = text[++i]-48; // initial state
	i+=3;

	// final state
	if(text[i+1]!= 13) {
		fS = (text[i]-48)*10 + (text[i+1]-48);
		++i;
	}
	else
		fS = text[i]-48;

	// delta table
	for (aux = i+=3, rows = 1; text[aux]; ++aux) if(text[aux] == 13) ++rows;
	printf("\n");
	deltaTab = (char *) malloc((rows*3+1)*sizeof(char));

	char j;
	for(j = 0, aux = i; text[aux]; ++aux)
		if(text[aux] != 13 && text[aux] != ',') {
			if(text[aux] == 10)
				deltaTab[j] = ',';
			else if(text[aux] > 47 && text[aux] < 58) {
				if(text[aux+1] > 47 && text[aux+1] < 58) {
					deltaTab[j] = (text[aux] - 48)*10 + text[aux+1] - 48;
					++aux;
				}
				else
					deltaTab[j] = text[aux] - 48;
			}
			else
				deltaTab[j] = text[aux];
			++j;
		}
	deltaTab[j] = '\0';

	printAInf();

	return 0;
}

char n;

char isIn(char state, int *set) {
	for(char m = 0; set[m]; ++m) 
		if(set[m] == state)
			return 1;
	return 0;
}

void epsilonLock(char actState, int *set, char num) {
	for(char i = 0; i<rows; ++i)
		if(deltaTab[i*4] == actState && deltaTab[i*4+1] == 32) {
			set[(num*(states+1))+deltaTab[i*4+2]] = 1;
			epsilonLock(deltaTab[i*4+2], set, num);
		}
}

void move(int *set, char num, int *whereTo, char numWhere, char delta) {
	for(char j = 0; j<rows; ++j)
		if(set[num*(states+1)+deltaTab[j*4]] && deltaTab[j*4+1] == delta)
			whereTo[numWhere*(states+1)+deltaTab[j*4+2]] = 1;
}

char numLetters = 0;
char letter = 65;

void newLetter(int *set, int *movement) {
	char newLetters[3] = {0, 0, 0};
	char repeated[3] = {0, 0, 0};
	char numRep = 0;


	for(char i = 0; i<numSigmaM; ++i)
		movement[i*(states+1)] = 1;

	for(char i = 0; i<numSigmaM; ++i)
		for(char j = 0; set[j*(states+1)]; ++j)
			for(char k = 0; k<states+1; ++k)
				if(movement[i*(states+1)+k] != set[j*(states+1)+k]) {
					break;
				}
				else if(k == states) {
					repeated[i] = (j*(states+1))/(states+1)+1;
					++numRep;
				}

	for(char i = 0; i<numSigmaM; ++i)
		for(char k = 1; k<(states+1); ++k)
			if(movement[i*(states+1)+k])
				break;
			else if(k == states-1) {
				repeated[i] = -1;
				++numRep;
			}

	for(char i = 0; i<numSigmaM; ++i)
		if(repeated[i] == 0) {
			set[(numLetters-numRep)] = 1;
			for(char j = 0; j<states+1; ++j)
				set[j+(numLetters*(states+1))] = movement[j+i*(states+1)];
			++numLetters;
		}

	printf(" | ");
	for(char i = 0; i<numSigmaM; ++i)
		if(repeated[i] == -1)
			printf(" Ø");
		else if(repeated[i] == 0)
			printf(" %c", ++letter);
		else
			printf(" %c", repeated[i]+64);

	for(char i = 0; i<numSigmaM; ++i)
		for(char k = 0; k<states+1; ++k)
			movement[i*(states+1)+k] = 0;
}

void go2(int *set, char num) {
	int whereTo[(states+1)*numSigmaM];

	for(int i = 0; i<(states+1)*numSigmaM; ++i)
		whereTo[i] = 0;
	
	for(int i = 0; sigmaM[i]; ++i)
		move(set, num, whereTo, i, sigmaM[i]);

	for(int i = 0; i<numSigmaM; ++i)
		for (int j = 0; j < states+1; ++j)
			if(whereTo[i*(states+1)+j])
				epsilonLock(j, whereTo, i);

	newLetter(set, whereTo);
}

void repeat(int * set) {
	printf("\nDelta del AFD:\n\tδ | ");
 	for(char n = 0; sigmaM[n]; ++n)
 		printf(" %c", sigmaM[n]);
	char i = 0;
	for(; set[i*(states+1)]; ++i) {
		printf("\n\t%c", 65+i);
		go2(set, i);
	}

	printf("\nEstado inicial: A\n");
	printf("Estados finales: ");
	for (char j = 0; set[j*(states+1)]; ++j) {
		if(set[j*(states)+fS+1] == 1) {
			printf("%d, %d = ", j*(states)+fS+1, set[j*(states)+fS+1]);
			printf(" %c", j+65);
		}
		printf(";");
	}
	printf("\n");
}

void toAFD() {
	int sets[(states+1)*states];

	for(int i = 0; i<(states+1)*states; ++i)
		sets[i] = 0;

	sets[0] = 1;
	sets[s] = 1;
	epsilonLock(s, sets, 0);
	++numLetters;

	repeat(sets);
}

int main(int argc, char const *argv[]) {
	char err = getAInf(argv[argc-1]);
	if(err)
		printf("\t(Err) No se encontro el archivo\n");
	else {
		printf("\n\nPasando a AFD...");
		toAFD();
	}
	return 0;
}
