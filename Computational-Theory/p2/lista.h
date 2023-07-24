#ifndef LISTA_H
#define LISTA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define OVERFLO     30   /* Si no hay memoria*/
#define UNDERFLO    31   /* En casos de lista vacio*/
#define POS_INV      32   /* Posicion invalida */

#define DATO      char*
#define FORMATO   " %p"
#define TAMDAT    10

#define OK          0   /* No hay errores */
#define AP_INV      1   /* Recibe un apuntador nulo */
#define NO_MEM      2   /* malloc regresa NULL*/

#define AP_VAL(p)   ((p)!=NULL)

typedef struct _nodo{
	DATO dato;
	struct _nodo* sig;
	struct _nodo* ant;
} nodo;
typedef nodo* apnodo;

#define TIE_DAT(l)  ((l)!=NULL)   /*Si tiene datos*/
#define VAC_LIS(l)  ((l)==NULL)   /* Si es un lista vac�o*/

typedef apnodo lista;

int crea_nod(apnodo* p, DATO d);    /*Crear un nodo*/
int elim_nod(apnodo* p);            /*Eliminar un nodo*/

int crea_lis(lista*);      /*Crea un lista vac�o*/
int dest_lis(lista*);      /*Elimina por completo un lista*/

int inse_lis(lista*,int,DATO);
int insp_lis(lista*,DATO); /*Inserta el primer nodo*/
int insf_lis(lista*,DATO); /*Inserta el ultimo*/
int long_lis(lista*);      /*Calcula la longitud de la lista*/
lista ante_lis(lista, int);

int elim_lis(lista*,int);  /*Saca un elemento del lista*/
int eliu_lis(lista*);      /*Elimina el �ltimo dato*/
int elip_lis(lista*);      /*Elimna el primer dato*/

int elip_list(lista* l);
int dest_list(lista* l);
#endif