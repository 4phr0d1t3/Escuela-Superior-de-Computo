#!/usr/bin/env python
# coding: utf-8

# Sebastian Ruiz Uvalle
# 
# 6BV1
# 
# Ing. en Inteligencia Artificial
# 
# 20 de junio de 2023
# 
# Este programa genera lo modelos de generacion de resumen de texto:
#  - Frecuencia de términos normalizada
#  - TextRank
#  - LSA
# 
# Usando el conjunto de datos de: https://huggingface.co/datasets/mlsum/viewer/es
# 
# Funciones encontradas en este programa:
#  - process_text(tupla): Creacion de una lista de texto donde el primer texto esta sin aplicar stemming ni lematization,
#  el segundo texto solo se le aplico stemming, al tercero solo lematization y al cuarto ambos, regresando la lista.
#  - tokenizacion(text): Devuelve las sentencias del texto ya tokenizadas.
#  - frecuencia_de_terminos_normalizada(text, num_sentences=3): Resume el texto usando el enfoque Frecuencia de términos normalizada.
#  - text_rank(text): Resume el texto usando el enfoque TextRank.
#  - lsa(text, num_sentences=3): Resume el texto usando el enfoque LSA.

# In[ ]:


pip install datasets


# In[1]:


from datasets import load_dataset
# Se carga el conjunto de datos "mlsum" en el idioma español
dataset = load_dataset('mlsum', 'es')


# In[4]:


import pandas as pd
# Se obtienen los datos de prueba del diccionario 'dataset'
test_data = dataset['test']
# Se crea un DataFrame 'df_test' utilizando los datos de prueba
df = pd.DataFrame(test_data)
# Se muestran las primeras filas del DataFrame 'df_test'
df.head()


# In[5]:


# Se eliminan las columnas 'url', 'date' y 'topic' del DataFrame df
df = df.drop('url', axis=1)
df = df.drop('date', axis=1)
df = df.drop('topic', axis=1)

# Se guarda el DataFrame df en un nuevo archivo CSV llamado 'data.csv'
df.to_csv('data.csv', index=False)


# In[6]:


import pandas as pd

# Se lee el archivo CSV y se carga en el DataFrame 'df'.
df = pd.read_csv('data.csv')

# Se muestra las primeras filas del DataFrame 'df'.
df.head()


# In[ ]:


get_ipython().system('python -m spacy download es_core_news_sm')


# In[7]:


def process_text(tupla):

    # Ninguna

    # Se convierte la tupla a minúsculas
    tupla = tupla.lower()
    # Se crea una lista con la tupla como único elemento
    lista = [tupla]

    # Stemming

    import nltk
    from nltk.stem import SnowballStemmer
    # Se crea un stemmer para el idioma español
    stemmer = SnowballStemmer('spanish')
    # Se tokeniza la tupla en una lista de tokens
    tokens = nltk.word_tokenize(tupla)
    # Se realiza la reducción de palabras a su raíz
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Se une los tokens procesados en un solo texto
    texto_procesado = ' '.join(stemmed_tokens)

    # Se agrega el texto lematizado a la lista 'lista'
    lista.append(texto_procesado)

    # Lematization

    import spacy
    # Se carga el modelo de procesamiento de lenguaje natural para español
    nlp = spacy.load('es_core_news_sm')
    # Se procesa la tupla utilizando el modelo de Spacy
    doc = nlp(tupla)
    # Se obtienen los lemas de cada token en el documento
    lemmas = [token.lemma_ for token in doc]
    # Se une nuevamente los lemas en un solo texto
    texto_lematizado = ' '.join(lemmas)

    lista.append(texto_lematizado)

    # Ambas

    # Se obtienen los lemas de cada token en la lista 'tokens'
    lemmas = [token.lemma_ for token in nlp(' '.join(tokens))]
    # Se realiza la reducción de palabras a su raíz
    stemmed_tokens = [stemmer.stem(lemma) for lemma in lemmas]
    # Se une nuevamente los tokens procesados en un solo texto
    texto_procesado = ' '.join(stemmed_tokens)

    lista.append(texto_procesado)

    return lista

# Puesta de los datos procesados en DataFrames diferentes para su posterior uso
df1 = pd.DataFrame(process_text(df['text'][0]), columns=['Texto'])
df2 = pd.DataFrame(process_text(df['text'][1]), columns=['Texto'])


# In[8]:


# Union de df1 y df2 en un solo DataFrame
textos =  pd.concat([df1, df2])

import pandas as pd

# Definir los valores de la columna 'Limpieza'
limpieza_valores = ['none', 'stemming', 'lemmatization', 'both'] * 2
textos['Transformacion'] = limpieza_valores

# Definir los valores de la columna 'idTexto'
id_texto_valores = [1, 1, 1, 1, 2, 2, 2, 2]
textos['idTexto'] = id_texto_valores
textos.reset_index(drop=True, inplace=True)

# Mostrado del contenido de 'textos'
textos


# In[9]:


# Guardado del DataFrame 'textos' en Drive para el facil seguimiento del codigo sin tener que guardar todo de nuevo
textos.to_csv(r'/content/drive/MyDrive/textos.csv', index=False)


# In[10]:


import pandas as pd

# Se lee el archivo CSV y se carga en un DataFrame llamado 'textos'
textos = pd.read_csv(r'/content/drive/MyDrive/textos.csv')

# Mostrado del contenido de 'textos'
textos


# In[11]:


# Agregacion de las columnas para guardado de los resumenes
textos["FTN"] = ""
textos["TextRank"] = ""
textos["LSA"] = ""


# In[16]:


import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from summa.summarizer import summarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def tokenizacion(text):
    # Tokenización de oraciones
    sentences = sent_tokenize(text, language='spanish')
    sentences = [[word for word in sentence.split() ] for sentence in sentences]

    return sentences

def frecuencia_de_terminos_normalizada(text, num_sentences=3):
    # Tokenizacion del texto
    sentences = tokenizacion(text)

    # Creación de vectores de frecuencia de términos
    vectorizer = CountVectorizer()
    term_freq_matrix = vectorizer.fit_transform([' '.join(sentence) for sentence in sentences])

    # Calcular puntajes de las oraciones
    sentence_scores = term_freq_matrix.sum(axis=1)
    sentence_scores = [score.item() for score in sentence_scores]

    # Selección de las oraciones más importantes
    top_sentences = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]

    # Generación del resumen
    summary = ' '.join([' '.join(sentences[i]) for i in top_sentences])

    return summary

def text_rank(text):
    # Tokenizacion del texto
    sentences = tokenizacion(text)

    # Unir las oraciones en un solo texto
    text = ' '.join([' '.join(sentence) for sentence in sentences])

    # Aplicar TextRank para generar el resumen
    summary = summarize(text, ratio=0.2)  # Ajusta el valor de 'ratio' según tus necesidades

    return summary

def lsa(text, num_sentences=3):
    # Tokenizacion del texto
    sentences = tokenizacion(text)

    # Unir las oraciones en un solo texto
    text = ' '.join([' '.join(sentence) for sentence in sentences])

    # Creación de la matriz término-documento con TF-IDF
    vectorizer = TfidfVectorizer()
    term_doc_matrix = vectorizer.fit_transform([text])

    # Aplicar descomposición de valores singulares (SVD) para reducir la dimensionalidad
    svd = TruncatedSVD(n_components=num_sentences, random_state=42)
    svd.fit(term_doc_matrix)
    sentence_scores = svd.components_.sum(axis=1)

    # Obtener las oraciones más importantes
    top_sentences = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]

    # Generación del resumen
    summary = ' '.join([sentences[i] for i in top_sentences])

    return summary


for i in range(8):
    # Asignacion del texto a resumir
    text = textos['Texto'][i]

    # Guardado de los resumenes en su respectiva columna y tupla
    textos['FTN'][i] = frecuencia_de_terminos_normalizada(text)
    textos['TextRank'][i] = text_rank(text)
    textos['LSA'][i] = text_rank(text)


# In[17]:


textos

