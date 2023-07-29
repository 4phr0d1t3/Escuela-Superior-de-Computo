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
# Este programa se encarga de la categorización de muestras de 5 libros diferentes obtenidos de Project Gutenberg (www.gutenberg.org), siendo el dataset de 2166 observaciones.

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


# In[20]:


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


# In[21]:


# Obtencion del corpus de los 5 libros ya limpiados en cleaned_boos_contents
cleaned_books_contents=[]
for book in book_contents :
    cleaned_books_contents.append(clean_text(book))
cleaned_books_contents[0][1:30]


# In[22]:


# Vista de cuantas palabras tiene cada documento
for i in range(len(cleaned_books_contents)):
    size = len(cleaned_books_contents[i])
    print(size)


# In[23]:


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


# In[24]:


samples_of_books=[]
for cleaned_book in cleaned_books_contents :
    samples_of_books.append(book_samples(cleaned_book,50))
samples_of_books[0][0]


# In[29]:


import pandas as pd
data_frame =pd.DataFrame()
data_frame['Sample']=[item for sublist in samples_of_books for item in sublist]
target=[[selected_books[i]]*min(500,len(samples_of_books[i])) for i in range(len(selected_books)) ]
data_frame['Book Name']=[item for sublist in target for item in sublist]
data_frame['Book Name'].unique()


# In[30]:


from sklearn.utils import shuffle

data_frame = shuffle(data_frame)
data_frame


# In[31]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data_frame['Book Name'])
data_frame['Book Name'] = y


# In[32]:


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(data_frame,test_size=0.2,random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(data_frame['Sample'], data_frame['Book Name'], random_state = 0, test_size=0.2)

df_train


# In[35]:


from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_ngram(n_gram,X_train=X_train,X_test=X_test):
    vectorizer = TfidfVectorizer(ngram_range=(1,n_gram))
    x_train_vec = vectorizer.fit_transform(X_train)
    x_test_vec = vectorizer.transform(X_test)
    return x_train_vec,x_test_vec

X_train1g_cv, X_test1g_cv = tfidf_ngram(1,X_train=X_train,X_test=X_test)
X_train2g_cv, X_test2g_cv = tfidf_ngram(2,X_train=X_train,X_test=X_test)

print(X_train1g_cv)
print(X_test1g_cv)
print(X_train2g_cv)
print(X_test2g_cv)


# In[37]:


from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

text_embedding = {
    'TF_IDF 1_gram':(X_train1g_cv,X_test1g_cv),
    'TF_IDF 2_gram':(X_train2g_cv,X_test2g_cv)
}

models = [
          LinearRegression(),
          BernoulliNB(),
          GaussianNB(),
          KNeighborsClassifier()
          ]

results_dict={'Model Name':[],'Embedding type':[],'Testing Accuracy':[],'Cross Validation':[]}

for model in models:
    for embedding_vector in text_embedding.keys():
        train = text_embedding[embedding_vector][0].toarray()
        test = text_embedding[embedding_vector][1].toarray()
        model.fit(train, y_train)

        results_dict['Model Name'].append(type(model).__name__)
        results_dict['Embedding type'].append(embedding_vector)

        test_acc = model.score(test, y_test)
        results_dict['Testing Accuracy'].append(test_acc)

        score = cross_val_score(model,test,y_test, scoring='r2')
        results_dict['Cross Validation'].append(score.mean())

results_df=pd.DataFrame(results_dict)

results_df


# In[42]:


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
train_data = tokenizer.texts_to_sequences(X_train)
train_data = tokenizer.sequences_to_matrix(train_data, mode='binary')

tokenizer.fit_on_texts(X_test)
test_data = tokenizer.texts_to_sequences(X_test)
test_data = tokenizer.sequences_to_matrix(test_data, mode='binary')

train_data


# In[43]:


test_data

