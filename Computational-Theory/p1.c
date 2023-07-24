#include <stdio.h>
#include <stdlib.h>

int sizeOfStr(char *str) {
	int size = 0;
	while (str[size]) size++;
	return size;
}

void ajustar(char *str, char *size) {
	str[*size] = '\0';
	realloc(str, sizeof(char)*(*size));
}

void concatenacion(char *str1, char *size1, char *str2, char *size2) {
	char aux = *size1;
	*size1 = *size1 + *size2;
	ajustar(str1, size1);

	for(char i = aux; *size2 > i-aux; ++i) str1[i] = str2[i-aux];
}

void promptConcatenacion(char *str, char *size){
	printf("Ingrese su segunda cadena: ");
	char *str2 = (char *) malloc(sizeof(char)*100);
	scanf("%s", str2);
	fflush(stdin);

	char size2 = sizeOfStr(str2);
	ajustar(str2, &size2);

	concatenacion(str, size, str2, &size2);

	free(str2);
}

void prefijo(char *str, char *size, char *elim) {
	if (*elim >= 0 && *elim <= *size) {
		*size = *elim;
		ajustar(str, size);
	}
	else printf("No se puede hacer mayor al tamaño de la cadena\n");
}

void promptPrefijo(char *str, char *size) {
	printf("Ingrese el tamaño del prefijo:\n");
	char elim;
	scanf("%hhd", &elim);
	fflush(stdin);

	prefijo(str, size, &elim);
}

void sufijo(char *str, char *size , char *elim) {
	if (*elim >= 0 && *elim <= *size) {
		for(int i = *size - *elim; str[i]; ++i) str[i - *size + *elim] = str[i];
		ajustar(str, elim);
		*size = *elim;
	}
	else printf("No se puede hacer mayor al tamaño de la cadena\n");
}

void promptSufijo(char *str, char *size) {
	printf("Ingrese el tamaño del sufijo:\n");
	char elim;
	scanf("%hhd", &elim);
	fflush(stdin);

	sufijo(str, size, &elim);
}

void subcadena(char *str, char *size, char *izq, char *der) {
	char restanteL = *size - *der;
	prefijo(str, size, &restanteL);

	char restanteR = *size - *izq;
	sufijo(str, size, &restanteR);
}

void promptSubcadena(char *str, char *size) {
	char izq, der;
	printf("Ingrese cuantos quiere elimindar de la izquierda: ");
	scanf("%hhd", &izq);
	fflush(stdin);

	printf("y de la derecha: ");
	scanf("%hhd", &der);
	fflush(stdin);

	subcadena(str, size, &izq, &der);
}

void subsecuencia(char *str, char *size, char *elim) {
	// for(char i = 0; elim[i]; ++i) printf("%d\n", elim[i]);
}

void promptSubsecuencia(char *str, char *size) {
	// char n;
	// printf("Ingrese el numero de letras que desea quitar: ");
	// scanf("%hhd", &n);

	// if(n < 0 || n > (*size)) printf("No puede ser negativo, ni mayor a la cadena\n");
	// else {
	// 	char elim[n];
	// 	printf("Ingrese la posicion de las letras que quiere quitar:\n");

	// 	char i;
	// 	for(i = 0; i < n; ++i) scanf("%hhd", &elim[i]);
	// 	ajustar(elim, &i);

	// 	subsecuencia(str, size, elim);
	// 	free(elim);
	// }
}

void inversion(char *str, char *size) {
	char aux;
	for(int i = 0; i<(*size)/2; ++i) {
		aux = str[i];
		str[i] = str[*size-i-1];
		str[*size-i-1] = aux;
	}
}

void potencia(char *str, char *size, char *pow) {
	char flg = 0;
	if(*pow < 0) {
		(*pow) *= -1;
		flg = 1;
	}
	char *aux = (char *) malloc(sizeof(char) * (*size));
	char i = 0;

	for(; str[i]; ++(i)) aux[i] = str[i];
	for(; (*pow) > 1; --(*pow)) concatenacion(str, size, aux, &i);
	if(flg == 1) inversion(str, size);

	free(aux);
}

void promptPotencia(char *str, char *size) {
	printf("A que potencia lo quiere elevar? ");
	char pow;
	scanf("%hhd", &pow);
	fflush(stdin);

	potencia(str, size, &pow);
}

int main(int argc, char const *argv[]) {
	char *str = (char *) malloc(sizeof(char)*100);
	printf("Ingrese su cadena: ");
	scanf("%[^\n]s", str);
	fflush(stdin);

	char *size;
	*size = sizeOfStr(str);
	ajustar(str, size);

	short repetir, opcion;

	do {
		printf("Que desea hacer con su cadena (%s)?\n", str);
		printf("\t1. Concatenacion\n");
		printf("\t2. Prefijo\n");
		printf("\t3. Sufijo\n");
		printf("\t4. Subcadena\n");
		printf("\t5. Subsecuencia\n");
		printf("\t6. Inversion\n");
		printf("\t7. Potencia\n");
		scanf("%hd", &opcion);
		fflush(stdin);

		if(opcion==1) promptConcatenacion(str, size);
		else if (opcion==2) promptPrefijo(str, size);
		else if (opcion==3)	promptSufijo(str, size);
		else if (opcion==4) promptSubcadena(str, size);
		else if (opcion==5) promptSubsecuencia(str, size);
		else if (opcion==6) inversion(str, size);
		else if (opcion==7) promptPotencia(str, size);
		else printf("No existe esa opcion\n");

		printf("Cadena actual: %s\n", str);
		printf("Desea seguir? (1. Si / Otro. No)\n");
		scanf("%hd", &repetir);
		fflush(stdin);
	} while (repetir == 1);

	free(str);
	free(size);
	return 0;
}
