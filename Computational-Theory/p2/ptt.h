#ifndef PTT_H
#define PTT_H

#include "lista.h"

#define max 30

int imprimir(lista p);
int soli_leng(lista *p);
int solchar(char *s);
int size(char *c);
int unir_leng(lista p, lista p1);
int concatenar(lista p, lista p1);
int pot(lista p, lista p1);
void copiar(lista p, lista *cop);
lista conc(lista p, lista p1);
lista *repetido(lista *p);
lista *rep(lista *p);
int compara(char *s, char *s2);
void po(lista *p, lista p1);
void invertir(lista *p);
void invert(char *s, int tam);

#endif