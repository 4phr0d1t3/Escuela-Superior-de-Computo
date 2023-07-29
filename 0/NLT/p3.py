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
# Este programa genera lo modelos de clasificacion:
#  - Regresión Logistica
#  - Maquinas de Soporte Vectorial
#  - Arboles de Decisión
# 
# Usando el conjunto de datos de: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
# 

# In[ ]:


import pandas as pd

# Lectura del csv descargado de https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
reviews = pd.read_csv(r'/content/drive/MyDrive/Reviews.csv')

# Mostrado de la cabeza para analisis de los datos
reviews.head()


# In[ ]:


# Eliminacion de las tuplas duplicadas con base en la columnta 'Text'
reviews_unique = reviews.drop_duplicates(subset=['Text'])


# In[ ]:


# Crea un nuevo DataFrame vacío llamado 'aux'.
aux = pd.DataFrame()

# Pasado a un DataFrame auxiliar para solo operar con las columnas que nos importan
aux['Summary'] = reviews_unique['Summary']
aux['Text'] = reviews_unique['Text']
aux['HelpfulnessNumerator'] = reviews_unique['HelpfulnessNumerator']
aux['HelpfulnessDenominator'] = reviews_unique['HelpfulnessDenominator']


# In[ ]:


# Esta función devuelve el valor asignado correspondiente al número de fila dado.
# Parámetros:
        # row_number: número de fila
        # assigned_value: diccionario con los valores asignados
# Retorna el valor asignado para la fila especificada.
def set_value(row_number, assigned_value):
    return assigned_value[row_number]

# Diccionario que mapea los valores de 'Score' a etiquetas de sentimiento.
sentimentDictionary = {
    1: 'Negative', 2: 'Negative',
    3: 'Neutral',
    4: 'Positive', 5: 'Positive'
}

# Se aplica la función 'set_value()' a la columna 'Score' del DataFrame 'reviews_unique',
# utilizando el diccionario 'sentimentDictionary' para asignar etiquetas de sentimiento a los valores de 'Score'.
# El resultado se guarda en una nueva columna llamada 'Sentiment' en el DataFrame 'aux'.
aux['Sentiment'] = reviews_unique['Score'].apply(set_value, args=(sentimentDictionary, ))

aux.head()


# In[ ]:


# Calcula el recuento de cada valor único en la columna 'Sentiment' del DataFrame 'aux'.
# Esto proporciona información sobre la distribución de los sentimientos en los datos.
aux['Sentiment'].value_counts()


# In[ ]:


# Crea un nuevo DataFrame vacío llamado 'neutral_rows'.
neutral_rows = pd.DataFrame()

# Filtra las filas del DataFrame 'aux' donde el valor de la columna 'Sentiment' es igual a 'Neutral'.
neutral_rows = aux[aux['Sentiment'] == 'Neutral']
# Agrega las filas filtradas al DataFrame 'aux' utilizando el método 'append()'.
# La opción 'ignore_index=True' asegura que se generen nuevos índices para las filas agregadas.
aux = aux.append(neutral_rows, ignore_index=True)


# In[ ]:


# Calcula el recuento de cada valor único en la columna 'Sentiment' del DataFrame 'aux'.
# Esto proporciona información sobre la distribución de los sentimientos en los datos.
aux['Sentiment'].value_counts()


# In[ ]:


# Filtra las filas del DataFrame 'aux' para seleccionar solo aquellas con el valor 'Positive' en la columna 'Sentiment'.
positive_df = aux[aux['Sentiment'] == 'Positive']

# El comentario de abajo fue para poder reanudar la codificacion desde el uso de este csv
#positive_df.to_csv('positive_sentiment.csv', index=False)


# In[ ]:


# Ordena el DataFrame 'positive_df' en función de las columnas 'HelpfulnessNumerator' y 'HelpfulnessDenominator'
# en orden descendente. Esto permite priorizar los registros con mayor valor de 'HelpfulnessNumerator'
# y 'HelpfulnessDenominator', lo que puede indicar una mayor utilidad de las reseñas .
sorted_reviews = positive_df.sort_values(by=['HelpfulnessNumerator', 'HelpfulnessDenominator'], ascending=False)

# Selecciona las primeras 60000 filas del DataFrame ordenado y sobrescribe el DataFrame 'positive_df' con ellas.
# Al hacer esto, se obtiene un subconjunto de las reseñas positivas más relevantes y útiles, que puede ser más manejable
# para análisis o visualización posteriores.
positive_df = sorted_reviews.head(60000)


# In[ ]:


# Define una lista llamada 'columns_of_interest' que contiene los nombres de las columnas que se desea seleccionar.
columns_of_interest = ['Summary','Text','Sentiment']

# Crea un nuevo DataFrame llamado 'df' para almacenar las filas que cumplen con cierta condición.
df = pd.DataFrame()

# Filtra las filas del DataFrame 'aux' donde el valor de la columna 'Sentiment' es igual a 'Negative',
# y selecciona solo las columnas de interés especificadas en la lista 'columns_of_interest'.
# La función 'copy()' se utiliza para realizar una copia de las filas seleccionadas en un nuevo DataFrame 'df'.
# Esto permite obtener un subconjunto de las filas con sentimiento negativo y solo las columnas de interés,
# que puede ser útil para análisis o visualización específicos.
df = aux.loc[aux['Sentiment'] == 'Negative', columns_of_interest].copy()


# In[ ]:


# Crea un nuevo DataFrame llamado 'neutral_df' para almacenar las filas que cumplen con cierta condición.
neutral_df = pd.DataFrame()

# Filtra las filas del DataFrame 'aux' donde el valor de la columna 'Sentiment' es igual a 'Neutral',
# y selecciona solo las columnas de interés especificadas en la lista 'columns_of_interest'.
# La función 'copy()' se utiliza para realizar una copia de las filas seleccionadas en un nuevo DataFrame 'neutral_df'.
# Esto permite obtener un subconjunto de las filas con sentimiento neutral y solo las columnas de interés,
# que puede ser útil para análisis o visualización específicos.
neutral_df = aux.loc[aux['Sentiment'] == 'Neutral', columns_of_interest].copy()


# In[ ]:


# Concatena verticalmente (une por filas) los DataFrames 'df' y 'neutral_df',
# y sobrescribe el resultado en el DataFrame 'df'.
df = pd.concat([df, neutral_df])


# In[ ]:


# Concatena verticalmente (une por filas) los DataFrames 'df' y 'positive_df' utilizando las columnas de interés especificadas en 'columns_of_interest',
# y sobrescribe el resultado en el DataFrame 'df'.
df = pd.concat([df, positive_df[columns_of_interest]])


# In[ ]:


# Calcula el recuento de cada valor único en la columna 'Sentiment' del DataFrame 'df'.
# Esto proporciona información sobre la distribución de los sentimientos en los datos filtrados.
df['Sentiment'].value_counts()


# In[ ]:


# Guarda el DataFrame 'df' en un archivo CSV llamado 'data.csv' sin incluir el índice en la salida.
df.to_csv('data.csv', index=False)


# In[ ]:


# Lee los datos desde un archivo CSV llamado 'data.csv' ubicado en la ruta "/content/drive/MyDrive/"
# y los carga en un nuevo DataFrame llamado 'data'.
data = pd.read_csv(r'/content/drive/MyDrive/data.csv')

# Muestra las primeras filas del DataFrame 'data'.
data.head()


# In[ ]:


# Combina las columnas 'Summary' y 'Text' como cadenas de texto en una nueva columna llamada 'SummaryText'.
data['SummaryText'] = data['Summary'].astype(str) + ' ' + data['Text'].astype(str)

# Convierte todas las cadenas de la columna 'SummaryText' a minúsculas.
data['SummaryText'] = data['SummaryText'].str.lower()

# Imprime los valores de la columna 'SummaryText'.
print(data['SummaryText'])


# In[ ]:


# Importa las bibliotecas necesarias de NLTK (Natural Language Toolkit).
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Descarga los recursos necesarios para el tokenizador de NLTK.
nltk.download('punkt')

# Inicializa el stemmer de Porter.
stemmer = PorterStemmer()

# Crea una lista vacía para almacenar los textos procesados mediante stemming.
stemmed_dataset = []

# Itera sobre cada texto en la columna 'SummaryText' del DataFrame 'data'.
for text in data['SummaryText']:
    # Tokeniza el texto en palabras y las convierte a minúsculas.
    tokens = word_tokenize(text.lower())
    # Realiza stemming en cada palabra del texto utilizando el stemmer de Porter.
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Une las palabras procesadas mediante stemming en un solo texto.
    stemmed_text = ' '.join(stemmed_tokens)
    # Agrega el texto procesado a la lista de 'stemmed_dataset'.
    stemmed_dataset.append(stemmed_text)

# Imprime un mensaje indicando que 'stemmed_dataset' está listo.
# print("stemmed_dataset ready")


# In[ ]:


# Crea un nuevo DataFrame 'df_stemming' utilizando los textos procesados por stemming.
df_stemming = pd.DataFrame(stemmed_dataset)

# Asigna la columna 'Sentiment' del DataFrame original 'data' a la columna 'Sentiment' del DataFrame 'df_stemming'.
df_stemming['Sentiment'] = data['Sentiment']

# Imprime los valores únicos de la columna 'Sentiment' en 'df_stemming'.
print(df_stemming["Sentiment"].unique())

# Renombra la columna '0' a 'Text' en 'df_stemming'.
df_stemming.rename(columns={0: 'Text'}, inplace=True)

# Muestra el DataFrame 'df_stemming'.
df_stemming


# In[ ]:


# Guarda el DataFrame 'df_stemming' en un archivo CSV llamado 'stemming.csv' sin incluir el índice en la salida.
df_stemming.to_csv('stemming.csv', index=False)


# In[ ]:


# Importa las bibliotecas necesarias de NLTK.
from nltk.stem import WordNetLemmatizer

# Descarga los recursos necesarios para la lematización.
nltk.download('wordnet')

# Inicializa el lematizador de WordNet.
lemmatizer = WordNetLemmatizer()

# Crea una lista vacía para almacenar los textos lematizados.
lemmatized_dataset = []

# Itera sobre cada texto en la columna 'SummaryText' del DataFrame 'data'.
for text in data['SummaryText']:
    # Tokeniza el texto en palabras y las convierte a minúsculas.
    tokens = word_tokenize(text.lower())
    # Realiza la lematización en cada palabra del texto utilizando el lematizador de WordNet.
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Une las palabras lematizadas en un solo texto.
    lemmatized_text = ' '.join(lemmatized_tokens)
    # Agrega el texto lematizado a la lista de 'lemmatized_dataset'.
    lemmatized_dataset.append(lemmatized_text)

# Imprime un mensaje indicando que 'lemmatized_dataset' está listo.
# print("lemmatized_dataset ready")


# In[ ]:


# Crea un nuevo DataFrame 'df_lematization' utilizando los textos lematizados.
df_lematization = pd.DataFrame(lemmatized_dataset)

# Asigna la columna 'Sentiment' del DataFrame original 'data' a la columna 'Sentiment' del DataFrame 'df_lematization'.
df_lematization['Sentiment'] = data['Sentiment']

# Imprime los valores únicos de la columna 'Sentiment' en 'df_lematization'.
print(df_lematization["Sentiment"].unique())

# Renombra la columna '0' a 'Text' en 'df_lematization'.
df_lematization.rename(columns={0: 'Text'}, inplace=True)

# Muestra el DataFrame 'df_lematization'.
df_lematization


# In[ ]:


# Guarda el DataFrame 'df_lematization' en un archivo CSV llamado 'lematization.csv' sin incluir el índice en la salida.
df_lematization.to_csv('lematization.csv', index=False)


# In[ ]:


# import nltk
# from nltk.stem import PorterStemmer, WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')

# Inicializa el stemmer de Porter y el lematizador de WordNet.
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Crea una lista vacía para almacenar los textos procesados tanto por stemming como por lematización.
both_dataset = []

# Itera sobre cada texto en la columna 'SummaryText' del DataFrame 'data'.
for text in data['SummaryText']:
    # Tokeniza el texto en palabras y las convierte a minúsculas.
    tokens = word_tokenize(text.lower())
    # Realiza el stemming en cada palabra del texto utilizando el stemmer de Porter.
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Realiza la lematización en cada palabra stemizada utilizando el lematizador de WordNet.
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    # Une las palabras lematizadas en un solo texto.
    processed_text = ' '.join(lemmatized_tokens)
    # Agrega el texto procesado a la lista 'both_dataset'.
    both_dataset.append(processed_text)

# Imprime un mensaje indicando que 'both_dataset' está listo.
print("both_dataset ready")


# In[ ]:


# Crea un nuevo DataFrame 'df_both' utilizando los textos procesados tanto por stemming como por lematización.
df_both = pd.DataFrame(both_dataset)

# Asigna la columna 'Sentiment' del DataFrame original 'data' a la columna 'Sentiment' del DataFrame 'df_both'.
df_both['Sentiment'] = data['Sentiment']

# Imprime los valores únicos de la columna 'Sentiment' en 'df_both'.
print(df_both["Sentiment"].unique())

# Renombra la columna '0' a 'Text' en 'df_both'.
df_both.rename(columns={0: 'Text'}, inplace=True)

# Muestra el DataFrame 'df_both'.
df_both


# In[ ]:


# Guarda el DataFrame 'df_both' en un archivo CSV llamado 'both.csv' sin incluir el índice en la salida.
df_both.to_csv('both.csv', index=False)


# In[ ]:


# Crea un nuevo DataFrame 'df_none' con las columnas 'Text' y 'Sentiment' del DataFrame original 'data'.
df_none = pd.DataFrame()
df_none['Text'] = data['SummaryText']
df_none['Sentiment'] = data['Sentiment']

# Imprime los valores únicos de la columna 'Sentiment' en 'df_both'.
print(df_both["Sentiment"].unique())

# Muestra el DataFrame 'df_both'.
df_both


# In[ ]:


# Guarda el DataFrame 'df_none' en un archivo CSV llamado 'none.csv' sin incluir el índice en la salida.
df_none.to_csv('none.csv', index=False)


# In[2]:


# Importa la biblioteca necesaria para montar Google Drive en Google Colab.
from google.colab import drive

# Monta Google Drive en el entorno de Colab.
drive.mount('/content/drive')


# In[1]:


# Importa la biblioteca pandas para trabajar con datos tabulares.
import pandas as pd

# Lee los archivos CSV correspondientes a los DataFrames df_none, df_stemming, df_lematization y df_both.
# df_none = pd.read_csv(r'none.csv')
# df_stemming = pd.read_csv(r'stemming.csv')
# df_lematization = pd.read_csv(r'lematization.csv')
# df_both = pd.read_csv(r'both.csv')

df_none = pd.read_csv(r'/content/drive/MyDrive/none.csv')
df_stemming = pd.read_csv(r'/content/drive/MyDrive/stemming.csv')
df_lematization = pd.read_csv(r'/content/drive/MyDrive/lematization.csv')
df_both = pd.read_csv(r'/content/drive/MyDrive/both.csv')

# Imprime las primeras filas de cada DataFrame para verificar la carga de datos.
print(df_none.head())
print(df_stemming.head())
print(df_lematization.head())
print(df_both.head())


# In[2]:


# Importa la clase LabelEncoder de scikit-learn para codificar las etiquetas de sentimiento.
from sklearn.preprocessing import LabelEncoder

# Crea una instancia de LabelEncoder.
label_encoder = LabelEncoder()

# Codifica las etiquetas de sentimiento en los DataFrames df_none, df_stemming, df_lematization y df_both.
df_none['Sentiment'] = label_encoder.fit_transform(df_none['Sentiment'])
df_stemming['Sentiment'] = label_encoder.fit_transform(df_stemming['Sentiment'])
df_lematization['Sentiment'] = label_encoder.fit_transform(df_lematization['Sentiment'])
df_both['Sentiment'] = label_encoder.fit_transform(df_both['Sentiment'])


# In[3]:


# Importa la función train_test_split de scikit-learn para dividir los conjuntos de datos.
from sklearn.model_selection import train_test_split

# Divide los conjunto de datos en conjuntos de entrenamiento y prueba.
X_none_train, X_none_test, y_none_train, y_none_test = train_test_split(df_none['Text'], df_none['Sentiment'], random_state = 0, test_size=0.2)
X_stemming_train, X_stemming_test, y_stemming_train, y_stemming_test = train_test_split(df_stemming['Text'], df_stemming['Sentiment'], random_state = 0, test_size=0.2)
X_lematization_train, X_lematization_test, y_lematization_train, y_lematization_test = train_test_split(df_lematization['Text'], df_lematization['Sentiment'], random_state = 0, test_size=0.2)
X_both_train, X_both_test, y_both_train, y_both_test = train_test_split(df_both['Text'], df_both['Sentiment'], random_state = 0, test_size=0.2)


# In[6]:


import tensorflow as tf

# Importa la clase TfidfVectorizer de scikit-learn para realizar la vectorización TF-IDF.
from sklearn.feature_extraction.text import TfidfVectorizer

# Definir la función para aplicar la vectorización TF-IDF a los conjuntos de datos de texto.
def tfidf_ngram(X_train, X_test):
	# Crea una instancia de TfidfVectorizer.
	vectorizer = TfidfVectorizer()
 # Aplica la vectorización TF-IDF a los conjuntos de entrenamiento y prueba.
	with tf.device('/gpu:0'): # Utiliza la GPU para acelerar el procesamiento si está disponible.
		x_train_vec = vectorizer.fit_transform(X_train)
		x_test_vec = vectorizer.transform(X_test)
	return x_train_vec,x_test_vec

# Aplicando la vectorizacion TF-IDF a los textos
X_tfidf_none_train, X_tfidf_none_test = tfidf_ngram(X_train=X_none_train,X_test=X_none_test)
X_tfidf_stemming_train, X_tfidf_stemming_test = tfidf_ngram(X_train=X_stemming_train,X_test=X_stemming_test)
X_tfidf_lematization_train, X_tfidf_lematization_test = tfidf_ngram(X_train=X_lematization_train,X_test=X_lematization_test)
X_tfidf_both_train, X_tfidf_both_test = tfidf_ngram(X_train=X_both_train,X_test=X_both_test)

print("Vectorizacion terminada")

# print(X_tfidf_none_train)
# print(X_tfidf_none_test)

# print(X_tfidf_stemming_train)
# print(X_tfidf_stemming_test)

# print(X_tfidf_lematization_train)
# print(X_tfidf_lematization_test)

# print(X_tfidf_both_train)
# print(X_tfidf_both_test)


# In[ ]:


# Importa las clases de los modelos y funciones relacionadas de scikit-learn.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Crea un diccionario que contiene los datos de texto vectorizados para cada tipo de procesamiento de texto.
text_embedding = {
	'none':(X_tfidf_none_train, X_tfidf_none_test, y_none_train, y_none_test),
	'stemming':(X_tfidf_stemming_train, X_tfidf_stemming_test, y_stemming_train, y_stemming_test),
	'lematization':(X_tfidf_lematization_train, X_tfidf_lematization_test, y_lematization_train, y_lematization_test),
	'both':(X_tfidf_both_train, X_tfidf_both_test, y_both_train, y_both_test)
}

# Crea una lista de modelos a entrenar y evaluar, cabe mencionar que se intento hacer el proceso iterable pero
#  se opto por hacerlo por separado ya que  tardaban mucho en entrenarse y evaluarlos
models = [
	LogisticRegression(),
	# SVC(),
	# DecisionTreeClassifier()
]

# Crea un diccionario para almacenar los resultados.
results_dict={
	'Model Name':[],
	'DataFrame':[],
	'Cross Validation':[]
}

# Para cada modelo y tipo de procesamiento de texto, realiza el entrenamiento y la evaluación.
for model in models:
	for embedding_vector in text_embedding.keys():
		X_train = text_embedding[embedding_vector][0]
		X_test = text_embedding[embedding_vector][1]
		y_train = text_embedding[embedding_vector][2]
		y_test = text_embedding[embedding_vector][3]

		with tf.device('/gpu:0'): # Utiliza la GPU para acelerar el entrenamiento si está disponible.
			model.fit(X_train, y_train)

		results_dict['Model Name'].append(type(model).__name__)
		results_dict['DataFrame'].append(embedding_vector)

		with tf.device('/gpu:0'): # Utiliza la GPU para acelerar la evaluación si está disponible.
			scores = cross_val_score(model, X_test, y_test, cv=5)
			results_dict['Cross Validation'].append(scores.mean())

# Crea un DataFrame con los resultados.
with tf.device('/gpu:0'): # Utiliza la GPU para acelerar la creación del DataFrame si está disponible.
	results_df=pd.DataFrame(results_dict)

results_df


# In[ ]:


# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import cross_val_score

# Siendo las mismas instrucciones para los comentarios del funcionamiento recurra a la anterior seccion

text_embedding = {
	'none':(X_tfidf_none_train, X_tfidf_none_test, y_none_train, y_none_test),
	'stemming':(X_tfidf_stemming_train, X_tfidf_stemming_test, y_stemming_train, y_stemming_test),
	'lematization':(X_tfidf_lematization_train, X_tfidf_lematization_test, y_lematization_train, y_lematization_test),
	'both':(X_tfidf_both_train, X_tfidf_both_test, y_both_train, y_both_test)
}

models = [
	# LogisticRegression(),
	SVC(kernel='linear', max_iter=1000),
	# DecisionTreeClassifier()
]

results_dict={
	'Model Name':[],
	'Embedding type':[],
	'Cross Validation':[]
}

for model in models:
	for embedding_vector in text_embedding.keys():
		X_train = text_embedding[embedding_vector][0]
		X_test = text_embedding[embedding_vector][1]
		y_train = text_embedding[embedding_vector][2]
		y_test = text_embedding[embedding_vector][3]

		with tf.device('/gpu:0'):
			model.fit(X_train, y_train)

		results_dict['Model Name'].append(type(model).__name__)
		results_dict['Embedding type'].append(embedding_vector)

		with tf.device('/gpu:0'):
			scores = cross_val_score(model, X_test, y_test, cv=5)
			results_dict['Cross Validation'].append(scores.mean())

with tf.device('/gpu:0'):
	results_df=pd.DataFrame(results_dict)

results_df


# In[ ]:


# Renombra la columna 'Embedding type' por 'DataFrame'
results_df.rename(columns = {'Embedding type':'DataFrame'}, inplace = True)
# Muestra los datos de 'results_df'
results_df


# In[ ]:


# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import cross_val_score

# Siendo las mismas instrucciones para los comentarios del funcionamiento recurra a la anterior seccion

text_embedding = {
	'none':(X_tfidf_none_train, X_tfidf_none_test, y_none_train, y_none_test),
	'stemming':(X_tfidf_stemming_train, X_tfidf_stemming_test, y_stemming_train, y_stemming_test),
	'lematization':(X_tfidf_lematization_train, X_tfidf_lematization_test, y_lematization_train, y_lematization_test),
	'both':(X_tfidf_both_train, X_tfidf_both_test, y_both_train, y_both_test)
}

models = [
	# LogisticRegression(),
	# SVC(),
	DecisionTreeClassifier()
]

results_dict={
	'Model Name':[],
	'Embedding type':[],
	'Cross Validation':[]
}

for model in models:
	for embedding_vector in text_embedding.keys():
		X_train = text_embedding[embedding_vector][0]
		X_test = text_embedding[embedding_vector][1]
		y_train = text_embedding[embedding_vector][2]
		y_test = text_embedding[embedding_vector][3]

		with tf.device('/gpu:0'):
			model.fit(X_train, y_train)

		results_dict['Model Name'].append(type(model).__name__)
		results_dict['Embedding type'].append(embedding_vector)

		with tf.device('/gpu:0'):
			scores = cross_val_score(model, X_test, y_test, cv=5)
			results_dict['Cross Validation'].append(scores.mean())
with tf.device('/gpu:0'):
	results_df=pd.DataFrame(results_dict)

results_df


# In[ ]:


# Renombra la columna 'Embedding type' por 'DataFrame'
results_df.rename(columns = {'Embedding type':'DataFrame'}, inplace = True)
# Muestra los datos de 'results_df'
results_df


# In[4]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# El siguiente código utiliza LabelEncoder para codificar variables categóricas
# y OneHotEncoder para aplicar codificación one-hot.
# para X_none_train
encoded = LabelEncoder().fit_transform(X_none_train)
print(encoded)

encoded = encoded.reshape(len(encoded), 1)
oneHot_none_train = OneHotEncoder(sparse=False).fit_transform(encoded)
print(oneHot_none_train)

# para X_none_test
encoded = LabelEncoder().fit_transform(X_none_test)
print(encoded)

encoded = encoded.reshape(len(encoded), 1)
oneHot_none_test = OneHotEncoder(sparse=False).fit_transform(encoded)
print(oneHot_none_test)


print("----------------------------------------------")
# X_stemming_train
encoded = LabelEncoder().fit_transform(X_stemming_train)
print(encoded)

encoded = encoded.reshape(len(encoded), 1)
oneHot_stemming_train = OneHotEncoder(sparse=False).fit_transform(encoded)
print(oneHot_stemming_train)

# para X_stemming_test
encoded = LabelEncoder().fit_transform(X_stemming_test)
print(encoded)

encoded = encoded.reshape(len(encoded), 1)
oneHot_stemming_test = OneHotEncoder(sparse=False).fit_transform(encoded)
print(oneHot_stemming_test)


print("----------------------------------------------")
# X_lematization_train
encoded = LabelEncoder().fit_transform(X_lematization_train)
print(encoded)

encoded = encoded.reshape(len(encoded), 1)
oneHot_lematization_train = OneHotEncoder(sparse=False).fit_transform(encoded)
print(oneHot_lematization_train)

# para X_lematization_test
encoded = LabelEncoder().fit_transform(X_lematization_test)
print(encoded)

encoded = encoded.reshape(len(encoded), 1)
oneHot_lematization_test = OneHotEncoder(sparse=False).fit_transform(encoded)
print(oneHot_lematization_test)


print("----------------------------------------------")
# X_both_train
encoded = LabelEncoder().fit_transform(X_both_train)
print(encoded)

encoded = encoded.reshape(len(encoded), 1)
oneHot_both_train = OneHotEncoder(sparse=False).fit_transform(encoded)
print(oneHot_both_train)

# para X_both_test
encoded = LabelEncoder().fit_transform(X_both_test)
print(encoded)

encoded = encoded.reshape(len(encoded), 1)
oneHot_both_test = OneHotEncoder(sparse=False).fit_transform(encoded)
print(oneHot_both_test)


# In[ ]:


# Esta seccion no se logro que funcionara por problemas de utilizacion de la memoria del servidor

# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# df_none = pd.read_csv(r'/content/drive/MyDrive/none.csv')
# df_stemming = pd.read_csv(r'/content/drive/MyDrive/stemming.csv')
# df_lematization = pd.read_csv(r'/content/drive/MyDrive/lematization.csv')
# df_both = pd.read_csv(r'/content/drive/MyDrive/both.csv')

model = Sequential()
model.add(Embedding(input_dim=oneHot_none_train.shape[1], output_dim=64, input_length=1))  # Embedding layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(oneHot_none_train, y_none_train, epochs=10, batch_size=32)
loss, accuracy = model.evaluate(oneHot_none_test, y_none_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

