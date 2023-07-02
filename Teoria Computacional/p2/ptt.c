#include "ptt.h"

int soli_leng(lista *p) {
	int i, n;
	char aux[1], *s;
	apnodo rep;

	printf("Numero de cadenas del lenguaje: ");
	fflush(stdin);
	scanf("%d", &n);
	fgets(aux, max, stdin);

	crea_lis(p);
	if (n == 0) return 0;
	for (i = 0; i < n; i++) {
		s = malloc(max * sizeof(char));
		solchar(s);
		insf_lis(p, s);
	}
	repetido(p);
}

int unir_leng(lista p, lista p1) {
	lista u, paux, p1aux;
	char *s;
	int m, n;
	m = long_lis(&p);
	n = long_lis(&p1);

	if (m == 0 && n == 0) {
		printf("La union de los lenguajes es el vacio\n");
		return 0;
	}

	copiar(p, &paux);
	copiar(p1, &p1aux);

	crea_lis(&u);
	printf("\nEl 1er lenguaje es el siguiente:\n");
	imprimir(p);
	printf("\nEl 2do lenguaje es el siguiente:\n");
	imprimir(p1);
	printf("\n");

	while (paux) {
		insf_lis(&u, paux->dato);
		paux = paux->sig;
	}
	while (p1aux) {
		insf_lis(&u, p1aux->dato);
		p1aux = p1aux->sig;
	}
	printf("\nLa union de los lenguajes es la siguiente:\n");
	repetido(&u);
	imprimir(u);
	printf("\n");
	dest_lis(&paux);
	dest_lis(&p1aux);
}

int concatenar(lista p, lista p1) {
	lista u, paux, p1aux;
	int n, m;
	copiar(p, &paux);
	copiar(p1, &p1aux);
	printf("\nEl 1er lenguaje es:\n");
	imprimir(paux);

	printf("\nEl 2do lenguaje es:\n");
	imprimir(p1aux);
	m = long_lis(&paux);
	n = long_lis(&p1aux);
	if (m == 0 || n == 0) {
		printf("La concatenacion de los lenguajes es el vacio\n");
		dest_lis(&paux);
		dest_lis(&p1aux);
		return 0;
	}
	printf("\nLa concatenacion de los lenguajes es:\n");
	u = conc(paux, p1aux);

	repetido(&u);
	imprimir(u);
	printf("\n");
	dest_lis(&paux);
	dest_lis(&p1aux);
}

int pot(lista p, lista p1) {
	lista u, paux,aux;
	int n, opc, i,m,o;
	m = long_lis(&p);
	o = long_lis(&p1);
	
	printf("\nEl 1er lenguaje es:\n");
	imprimir(p);
	
	printf("\nEl 2do lenguaje es:\n");
	imprimir(p1);
	
	printf("\nCual de los dos lenguajes quieres elevar? \n\t1.el primero\n\t2.el segundo\n");
	fflush(stdin);
	scanf("%d", &opc);
	
	printf("A que potencia?: ");
	fflush(stdin);
	scanf("%d", &n);
	
	if(n == 0) {
		printf("El resultado es una cadena vacia\n");
		return 0;
	}
	crea_lis(&u);

	if (opc == 1 && m != 0) copiar(p, &paux);
	else if (opc == 2 && o != 0) copiar(p1, &paux);
	else {
		printf("Esa no es una opcion valida de numero de lenguaje o es el conjunto vacio\n");
		return 0;
	}

	if(n < 0) invertir(&paux);
	u = paux;
	for (i = 1; i < fabs(n); i++) po(&u, paux);

	if (opc == 1 || opc == 2) {
		repetido(&u);
		imprimir(u);
		printf("\n");
		dest_lis(&paux);
		dest_lis(&u);
	}
}

int imprimir(lista p) {
	if (p == NULL) {
		printf("No hay, no existe\n");
		return 0;
	}
	lista aux;
	aux = p;
	while (aux->sig) {
		printf("%s", aux->dato);
		aux = aux->sig;
	}
	printf("%s", aux->dato);
}

int solchar(char *s) {
	int n;
	printf("Escribe la cadena: ");
	fflush(stdin);

	fgets(s, max, stdin);
	n = size(s);
	s = realloc(s, n * sizeof(char));
	return n;
}

int size(char *c) {
	int n = 0;
	while (c[n]) n++;
	return (n);
}

lista conc(lista p, lista p1) {
	lista u, paux, p1aux, tmp, tmp1;
	char *s;
	int n, m, i, j;
	copiar(p, &tmp);
	copiar(p1, &tmp1);

	paux = tmp;
	p1aux = tmp1;
	crea_lis(&u);
	while (paux) {
		n = size(paux->dato);
		p1aux = tmp1;
		while (p1aux) {
			m = size(p1aux->dato);
			s = malloc((m + n) * sizeof(char *));
			for (i = 0; i < (n - 1); i++) s[i] = paux->dato[i];

			for (j = 0; j < m; j++) s[i + j] = p1aux->dato[j];

			s[i + j + 1] = '\0';
			insf_lis(&u, s);
			p1aux = p1aux->sig;
		}
		paux = paux->sig;
	}
	dest_lis(&tmp);
	dest_lis(&tmp1);
	return u;
}

lista *repetido(lista *p) {
	lista *aux;
	aux = p;
	while (aux) aux = rep(p);
}

lista *rep(lista *p) {
	lista aux, aux2;
	int n, i, j;
	aux = *p;
	aux2 = aux->sig;
	n = long_lis(p);
	for (i = 0; i < (n - 1); i++) {
		aux2 = aux->sig;
		for (j = (i + 1); j < n; j++) {
			if (compara((aux->dato), (aux2->dato)))
				if (i == 0) {
					elip_lis(p);
					return (p);
				}
				else {
					elim_lis(p, i);
					return (p);
				}
			aux2 = aux2->sig;
		}
		aux = aux->sig;
	}
	return NULL;
}

void copiar(lista p, lista *cop) {
	lista tmp;
	tmp = p;
	crea_lis(cop);
	while (tmp) {
		insf_lis(cop, tmp->dato);
		tmp = tmp->sig;
	}
}

int compara(char *s, char *s2) {
	int n = size(s), m = size(s2), i;
	if (n != m) return 0;
	for (i = 0; i < n; i++)
		if (s[i] != s2[i])
			return 0;
	return 1;
}

void po(lista *p, lista p1) {
	lista u, paux, p1aux, tmp, tmp1;
	char *s;
	int n, m, i, j;
	copiar(*p, &tmp);
	copiar(p1, &tmp1);

	paux = tmp;
	p1aux = tmp1;
	crea_lis(&u);
	while (paux) {
		n = size(paux->dato);
		p1aux = tmp1;
		while (p1aux) {
			m = size(p1aux->dato);
			s = malloc((m + n) * sizeof(char *));
			for (i = 0; i < (n - 1); i++) s[i] = paux->dato[i];

			for (j = 0; j < m; j++) s[i + j] = p1aux->dato[j];

			s[i + j + 1] = '\0';
			insf_lis(&u, s);

			p1aux = p1aux->sig;
		}
		paux = paux->sig;
	}
	*p = u;
	dest_lis(&tmp);
	dest_lis(&tmp1);
}

void invertir(lista *p) {
	lista tmp, aux, ax;
	int i, tam;
	copiar(*p,&tmp);
	elip_lis(p);
	crea_lis(p);
	tam = long_lis(&tmp);
	ax = tmp;  
	while (ax) {
		invert(ax->dato,size(ax->dato));
		insf_lis(p,ax->dato);
		ax = ax->sig;
	}
}

void invert(char *s, int tam){
	char *tmp;
	int i;
	tmp = malloc(tam*sizeof(char));

	for (i = 0; i < tam ; i++) tmp[i] = s[i];
	for (i = 0; i < tam; i++) s[i] = tmp[tam - 2 - i];

	s[tam-1] = '\n';
	free(tmp);
}