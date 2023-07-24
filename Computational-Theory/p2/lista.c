#include "lista.h"

int crea_lis(lista* l) {
	if(!AP_VAL(l)) return AP_INV;
	*l = NULL;
	return OK;
}

int dest_lis(lista* l) {
	if(!AP_VAL(l)) return AP_INV;
	while(!VAC_LIS(*l)) elip_lis(l);
	return OK;
}

int dest_list(lista* l) {
	if(!AP_VAL(l)) return AP_INV;
	while(!VAC_LIS(*l)) elip_list(l);
	return OK;
}

int long_lis(lista* l) {
	int cont;
	lista aux;
	if(!AP_VAL(l)) return AP_INV;

	cont = 0;
	aux = *l;
	while(aux) {
		cont++;
		aux = aux->sig;
	}
	return cont;
}

lista ulti_lis(const lista l) {
	lista aux,ant;
	aux = l;
	ant = aux;
	while(aux) {
		ant = aux;
		aux = aux->sig;
	}
	return ant;
}

lista ante_lis(lista l, int p) {
	lista aux,ant;
	int pos;
	aux = l; ant = aux;
	pos = 0;
	while (aux && pos<p) {
		ant = aux;
		aux = aux->sig;
		pos++;
	}
	return ant;
}

int insp_lis(lista* l,DATO d) {
	apnodo nuevo;
	if(!AP_VAL(l)) return AP_INV;
	if(!(crea_nod(&nuevo,d)==OK)) return OVERFLO;
	nuevo->sig = *l;
	nuevo->ant = NULL;
	if(*l) (*l)->ant = nuevo;
	*l = nuevo;
	return OK;
}

int insf_lis(lista* l,DATO d) {
	apnodo nuevo,aux;

	if(!AP_VAL(l)) return AP_INV;  
	if(VAC_LIS(*l)) return insp_lis(l,d);
	if(!(crea_nod(&nuevo,d)==OK)) return OVERFLO;

	aux = ulti_lis(*l);
	nuevo->ant = aux;
	aux->sig = nuevo;
	return OK;
}

int inse_lis(lista* l ,int pos ,DATO d) {
	apnodo nuevo;
	lista aux;

	if(!AP_VAL(l)) return AP_INV;
	if(!(pos <= long_lis(l) && pos >=0)) return POS_INV;
	if(!(crea_nod(&nuevo,d)==OK)) return OVERFLO;

	aux = ante_lis(*l,pos);
	nuevo->sig = aux->sig;
	nuevo->ant = aux;
	if(aux->sig) aux->sig->ant = nuevo;
	aux->sig = nuevo;

	return OK;
}

int elim_lis(lista*l,int pos) {
	lista aux,borrar;

	if(!AP_VAL(l)) return AP_INV;
	if(!TIE_DAT(*l)) return UNDERFLO;
	if(!(pos <= long_lis(l) && pos >0)) return POS_INV;
	aux = ante_lis(*l,pos);
	if(aux->sig) {
		borrar = aux->sig;
		aux->sig = borrar->sig;
		if(borrar->sig) borrar->sig->ant = aux;

		free(borrar->dato);
		free(borrar);
	}
	return OK; 
}

int eliu_lis(lista* l) {
	lista aux;
	
	if(!AP_VAL(l)) return AP_INV;
	if(!TIE_DAT(*l)) return UNDERFLO;
	aux = ulti_lis(*l);
	
	if(aux->ant) aux->ant->sig=NULL;
	else *l=NULL;

	free(aux);
	return OK;
}

int elip_lis(lista* l) {
	apnodo borrar;
	if(!AP_VAL(l)) return AP_INV;
	if(!TIE_DAT(*l)) return UNDERFLO;

	borrar = *l;
	*l=(*l)->sig;
	if(*l) (*l)->ant = NULL;

	free(borrar);
	return OK;
}

int elim_nod(apnodo* p) {
	if(!AP_VAL(p)) return AP_INV;
	free(*p);
	*p = NULL;
	return OK;
}

int crea_nod(apnodo* p, DATO d) {
	if(!AP_VAL(p)) return AP_INV;

	*p = (nodo*)malloc(sizeof(nodo));
	if(!AP_VAL(*p)) return NO_MEM;

	(*p)->dato = d;
	(*p)->sig = NULL;
	(*p)->ant = NULL;
	return OK;
}

int elip_list(lista* l) {
	apnodo borrar;

	if(!AP_VAL(l)) return AP_INV;
	if(!TIE_DAT(*l)) return UNDERFLO;
	borrar =*l;

	*l=(*l)->sig;
	if(*l) (*l)->ant = NULL;

	free(borrar->dato);
	free(borrar);
	return OK;
}
