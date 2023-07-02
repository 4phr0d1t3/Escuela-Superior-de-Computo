#include <stdio.h>
#define MAX 50

int S[MAX]; 
int c[MAX]; 
int f[MAX]; 

void ingresarActividades(int n) {
	for(int i=0; i<n; i++) {
		do {
			printf("Actividad %d\n\n", i+1);
			printf("\tinicio: ");
			scanf("%d", &c[i]);
			printf("\tfinal: ");
			scanf("%d", &f[i]);
			printf("\n");
		} while(c[i] > f[i]);
	}
}

void mostrarDatos(int n) {
	printf("Actividades ingresadas.\n\n");
	printf("\t\t Ai     :  ");

	for(int i = 0; i < n; ++i) printf("%d ", i+1);
	printf("\n\t\t-------------------------\n");
	printf("\t\t Inicio :  ");

	for(int i = 0; i < n; ++i) printf("%d ", c[i]);
	printf("\n\t\t Fin    :  ");

	for(int i = 0; i < n; ++i) printf("%d ", f[i]);
}

void ordenar(int n) {
	int aux1, aux2, aux3, band = 1;
	for(int i = n-1; i > 0 && band == 1; --i) {
		band = 0;
		for(int j = 0; j < i; ++j)
			if( f[j] > f[j+1]) {
				aux1 = f[j];
				f[j] = f[j+1];
				f[j+1] = aux1;

				aux2 = c[j];
				c[j] = c[j+1];
				c[j+1] = aux2;

				band=1;
			}
	}
	mostrarDatos(n);
}

void entregaSolucion() {
	printf("\n\nSolucion voraz:\t");
	int i = 0;
	while(1) {
		printf("A%d ", S[i] + 1);
		i++;
		if(S[i]==0)	break;
	}

	printf("\n");
}

void algoritmoSeleccion(int n) {
	ordenar(n);
	int z, k = 1;

	S[0] = 0;
	z = 0;

	for( int i=1; i<n; i++ )
		if(c[i] >= f[z]) {
			S[k] = i;
			z = i;
			k++;
		}
	entregaSolucion();
}

int main(int argc, char const *argv[]) {
	int n = 0;
	printf("Numero de actividades: ");
	scanf("%d", &n);
	printf("\n");

	ingresarActividades(n);
	algoritmoSeleccion(n);

	return 0;
}
