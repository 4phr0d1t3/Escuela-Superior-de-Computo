from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, lower, expr, regexp_replace, udf
from pyspark.ml.feature import StringIndexer, Tokenizer, CountVectorizer,  HashingTF, IDF
from pyspark.sql import functions as F

import nltk
from nltk.stem import PorterStemmer

from wordcloud import WordCloud

import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("RegresionLineal").getOrCreate()

data = spark.read.csv("sms_spam.csv", header=True, inferSchema=True)

data.printSchema()

columnas_string = [col_name for col_name, data_type in data.dtypes if data_type == 'string']

data.select(columnas_string).show()


sms_corpus_pandas = data.toPandas()
print(sms_corpus_pandas)


sms_corpus_subset = data.limit(2).collect()

sms_corpus_subset_list = [row.asDict() for row in sms_corpus_subset]

for row in sms_corpus_subset_list:
	print(row)

spark = SparkSession.builder.appName("CreacionCorpus").getOrCreate()
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")

count_vectorizer = CountVectorizer(inputCol="tokens", outputCol="features")

pipeline = Pipeline(stages=[tokenizer, count_vectorizer])

sms_corpus = pipeline.fit(data).transform(data)

sms_corpus.show()

numero_documentos = sms_corpus.count()
print("Número de documentos:", numero_documentos)

documentos = data.select("text").limit(2).collect()

for i, documento in enumerate(documentos):
	print(f"Documento {i + 1}:")
	print("Metadata:", len(documento[0]))
	print("Content: chars:", len(documento[0]))
	print()

documento_str = str(documentos)
print(documento_str, "")

sms_corpus_clean = sms_corpus.withColumn("text", lower(sms_corpus["text"]))
sms_corpus_clean.show()

from pyspark.sql.functions import lower, col
stop_words = [".",",", "!", "‘", "'"]

sms_corpus_clean = sms_corpus.withColumn("text", lower(col("text")))

for word in stop_words:
	sms_corpus_clean = sms_corpus_clean.filter(~(col("text").contains(word)))

sms_corpus_clean.show()

from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace, col

spark = SparkSession.builder.appName("EjemploPySpark").getOrCreate()

def replace_punctuation(text_col):
	return regexp_replace(text_col, r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ')

sms_corpus_clean = sms_corpus.withColumn("text", lower(col("text")))

sms_corpus_clean = sms_corpus_clean.withColumn("text", replace_punctuation(col("text")))

sms_corpus_clean.show()



words = ["learn", "learned", "learning", "learns"]
nltk.download('punkt')
stemmer = PorterStemmer()
def stem_words(word_list):
	return [stemmer.stem(word) for word in word_list]
word_df = spark.createDataFrame([(words,)], ["words"])
stem_udf = udf(stem_words, StringType())
result_df = word_df.withColumn("stemmed_words", stem_udf("words"))

result_df.show()

import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()

def stem_text(text):
	words = text.split()
	stemmed_words = [stemmer.stem(word) for word in words]
	return ' '.join(stemmed_words)

stem_udf = udf(stem_text, StringType())

sms_corpus_clean = sms_corpus_clean.withColumn("text", stem_udf("text"))

sms_corpus_clean.show()


sms_corpus_clean = sms_corpus_clean.withColumn("text", regexp_replace("text", r'\s+', ' '))

sms_corpus_clean.show()


train_ratio = 0.7
test_ratio = 1.0 - train_ratio

sms_dtm_train, sms_dtm_test = sms_corpus_clean.randomSplit([train_ratio, test_ratio], seed=42)

print("Número de filas en el conjunto de entrenamiento:", sms_dtm_train.count())
print("Número de filas en el conjunto de prueba:", sms_dtm_test.count())


spam_data = data.filter(data['type'] == 'spam')

ham_data = data.filter(data['type'] == 'ham')

spam_data = spam_data.withColumn("text", F.col("text").cast("string"))
ham_data = ham_data.withColumn("text", F.col("text").cast("string"))

spam_text = " ".join(spam_data.select("text").rdd.flatMap(lambda x: x).collect())
ham_text = " ".join(ham_data.select("text").rdd.flatMap(lambda x: x).collect())

wordcloud_spam = WordCloud(max_words=40, width=300, height=150, scale=3.0).generate(spam_text)
wordcloud_ham = WordCloud(max_words=40, width=300, height=150, scale=3.0).generate(ham_text)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_spam, interpolation="bilinear")
plt.title("Word Cloud - Spam")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_ham, interpolation="bilinear")
plt.title("Word Cloud - Ham")
plt.axis("off")

plt.tight_layout()
plt.show()


tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures")
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="idf_features")
indexer = StringIndexer(inputCol="type", outputCol="label")
nb = NaiveBayes(labelCol="label", featuresCol="idf_features")

pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, indexer, nb])

model = pipeline.fit(sms_dtm_train)

predictions = model.transform(sms_dtm_test)

evaluator = MulticlassClassificationEvaluator(
	labelCol="label",
	predictionCol="prediction",
	metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print("Accuracy en el conjunto de prueba:", accuracy)
print(model)


data_with_predictions = data.join(predictions, on="type")

crosstab = data_with_predictions.crosstab("type", "prediction")

crosstab.show()
crosstab_percentage = crosstab.withColumn("ham_percentage", expr("ham / (ham + spam)"))
crosstab_percentage = crosstab_percentage.withColumn("spam_percentage", expr("spam / (ham + spam)"))
crosstab_percentage.show()
