#include "ptt.h"

int menu();

int main() {
	int opc = 0;
	lista p, p1;
	soli_leng(&p);
	soli_leng(&p1);
	do {
		opc = menu();
		if(opc == 1)
			unir_leng(p, p1);
		else if(opc == 2)
			concatenar(p, p1);
		else if(opc == 3)
			pot(p, p1);
		else if(opc == 4)
		break;
		else
			printf("Opcion no valida\n");
	} while (opc != 4);
	dest_lis(&p);
	dest_lis(&p1);
	return 0;
}

int menu() {
	int opc;
	printf("Operaciones con lenguajes\n");
	printf("\t1. Union\n");
	printf("\t2. Concatenacion\n");
	printf("\t3. Potencia\n");
	printf("\t4. Salir\n");
	fflush(stdin);
	scanf("%d", &opc);

	return opc;
}
