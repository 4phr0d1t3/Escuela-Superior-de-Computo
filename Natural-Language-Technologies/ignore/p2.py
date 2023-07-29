#!/usr/bin/env python
# coding: utf-8

# Sebastian Ruiz Uvalle
# 
# 6BV1
# 
# Ing. en Inteligencia Artificial
# 
# 19 de abril de 2023
# 
# Este programa se encarga de la agrupación de muestras de 5 libros diferentes obtenidos de Project Gutenberg (www.gutenberg.org), siendo el dataset de 2166 observaciones.

# In[1]:


# Descarga de los titulos disponibles para descargar del Proyecto Gutenberg
import numpy as np
import nltk

nltk.download('gutenberg')
books_names=nltk.corpus.gutenberg.fileids()
books_names


# In[2]:


# Selección de los libros que vamos a usar para la practica

books_idx=[1,3,5,7,9]
selected_books=[]
for idx in books_idx:
  selected_books.append(books_names[idx])
print(selected_books)


# In[3]:


# Obtención del corpus de los libros seleccionados y su visualización

book_contents=[]
for book_name in selected_books:
    book_contents.append(nltk.corpus.gutenberg.raw(book_name))
book_contents[0][1:500]


# In[4]:


# Descarga de las bibliotecas necesarias para poder remover palabras irrelevantes, sustituir palabras
# flexionadas por un termino general y sustituir terminos por su raiz gramatical
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# In[5]:


import re

def get_wordnet_pos(word):
    # Mapeamos la etiqueta POS al primer caracter lemmatize() acepte
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def clean_text(text):
    # Definicion de patron de palabras para la mantencion de estas y pasado de texto a minusculas
    lemmatizer = WordNetLemmatizer()
    sub_pattern = r'[^A-Za-z]'
    split_pattern = r"\s+"
    stop_words = stopwords.words('english') + ['chapter','never','ever','couldnot','wouldnot','could','would','us',"i'm","you'd"]
    lower_book = text.lower()

    # Reemplazando todos los caracteres, excepto los que esten en los patrones definidos en sub_patten
    # a espacios, tokenizado de los documentos y lematizacion 
    filtered_book = re.sub(sub_pattern,' ',lower_book).lstrip().rstrip()
    filtered_book = word_tokenize(filtered_book)
    filtered_book = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in filtered_book if word not in stop_words]

    return filtered_book


# In[6]:


# Obtencion del corpus de los 5 libros ya limpiados en cleaned_boos_contents
cleaned_books_contents=[]
for book in book_contents :
    cleaned_books_contents.append(clean_text(book))
cleaned_books_contents[0][1:30]


# In[7]:


# Vista de cuantas palabras tiene cada documento
for i in range(len(cleaned_books_contents)):
    size = len(cleaned_books_contents[i])
    print(size)


# In[8]:


# Creacion de 500 muestras aproximadamente aleatorias de cada libro donde cada muestra contiene 50 palabras
def book_samples(book,n_samples) :
    import random
    samples=[]
    start=0
    while start +n_samples < len(book)-1:
        temp1=""
        for j in range(start,start+n_samples):
            temp1+= book[j] + " "
        samples.append(temp1)
        start+=n_samples
    random_samples_index=random.sample(range(0,len(samples)), k=min(500,len(samples)))
    partitions=[]
    for idx in random_samples_index :
        partitions.append(samples[idx])
    return partitions


# In[9]:


samples_of_books=[]
for cleaned_book in cleaned_books_contents :
    samples_of_books.append(book_samples(cleaned_book,50))
samples_of_books[0][0]


# In[10]:


import pandas as pd
data_frame = pd.DataFrame()
data_frame[0] = [item for sublist in samples_of_books for item in sublist]
target = [[selected_books[i]]*min(500,len(samples_of_books[i])) for i in range(len(selected_books)) ]
data_frame[1] = [item for sublist in target for item in sublist]
data_frame[1].unique()


# In[11]:


from sklearn.utils import shuffle

data = shuffle(data_frame[0])
data


# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer1Gram = TfidfVectorizer(ngram_range=(1,1))
vectorizer2Gram = TfidfVectorizer(ngram_range=(1,2))

X1g = vectorizer1Gram.fit_transform(data)
X2g = vectorizer2Gram.fit_transform(data)
print(X1g)
print(X2g)


# In[13]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []
for j in range(1,20):
    # print('Realizando kMeans con ' + str(j) + " kluster's")
    kmeans = KMeans(n_clusters=j, max_iter=300)
    kmeans.fit(X1g)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,20), wcss,'o-')
plt.title("Codo de Jambú")
plt.xlabel('N - Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()


# In[14]:


wcss = []
for j in range(1,20):
    # print('Realizando kMeans con ' + str(j) + " kluster's")
    kmeans = KMeans(n_clusters=j, max_iter=300)
    kmeans.fit(X2g)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,20), wcss,'o-')
plt.title("Codo de Jambú")
plt.xlabel('N - Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()


# In[16]:


from sklearn.decomposition import TruncatedSVD

kmeans = KMeans(n_clusters=5, max_iter=300)
kmeans.fit(X1g)
pred = kmeans.fit_predict(X1g)

svd = TruncatedSVD(n_components=2, random_state=24)
dataSVD = svd.fit_transform(X1g)
fitted = kmeans.fit_predict(dataSVD)
fitted_pd = pd.DataFrame(fitted)

clusters = kmeans.labels_
centers = kmeans.cluster_centers_

plt.scatter(dataSVD[:,0], dataSVD[:,1], c=clusters)
plt.scatter(centers[:,0], centers[:,1], marker='*', s=200, c='black')
plt.title('KMeans Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# In[17]:


kmeans = KMeans(n_clusters=5, max_iter=300)
kmeans.fit(X2g)
pred = kmeans.fit_predict(X2g)

svd = TruncatedSVD(n_components=2, random_state=24)
dataSVD = svd.fit_transform(X2g)
fitted = kmeans.fit_predict(dataSVD)
fitted_pd = pd.DataFrame(fitted)

clusters = kmeans.labels_
centers = kmeans.cluster_centers_

plt.scatter(dataSVD[:,0], dataSVD[:,1], c=clusters)
plt.scatter(centers[:,0], centers[:,1], marker='*', s=200, c='black')
plt.title('KMeans Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# In[18]:


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
train_data = tokenizer.texts_to_sequences(data)
train_data = tokenizer.sequences_to_matrix(train_data, mode='binary')

train_data

