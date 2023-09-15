import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator


spark = SparkSession.builder.appName('practica2').getOrCreate()

def obtain_features(df):
	input_cols = ['op','co','ex','ag','ne','wordcount']
	vect_assembler = VectorAssembler(
		inputCols= input_cols,
		outputCol= "features"
	)
	final_data = vect_assembler.transform(df)

	return final_data

def open_data():
	df = spark.read.csv(
		"./analisis.csv",
		header=True,
		inferSchema=True
	)
	# df.show()
	# df.printSchema()
	data = obtain_features(df)

	return data

def analyse_for_k(data):
	costs = []
	sse = []
	K = 10

	for k in range(2, K):
		kmeans = KMeans().setK(k).setSeed(1)
		model = kmeans.fit(data)
		cost = model.summary.trainingCost
		costs.append(cost)

		predictions = model.transform(data)
		evaluator = ClusteringEvaluator()
		silhouette = evaluator.evaluate(predictions)
		sse.append(silhouette)

	plt.figure(figsize=(10, 6))
	plt.plot(range(2, K), costs, marker='o', linestyle='-')
	plt.xlabel('Number of Clusters (k)')
	plt.ylabel('Cost (Inertia)')
	plt.title('Elbow Method for Optimal k')
	plt.grid(True)
	plt.show()

	plt.figure(figsize=(10, 6))
	plt.plot(range(2, K), sse, marker='o', linestyle='-', color='b')
	plt.xlabel('Number of Clusters (k)')
	plt.ylabel('SSE Score')
	plt.title('SSE Line')
	plt.grid(True)
	plt.show()


def plot(predictions, X_axis, y_axis):
	cluster_labels = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

	plt.figure(figsize=(10, 6))

	x_values = predictions.select(f.col(X_axis)).rdd.flatMap(lambda x: x).collect()
	y_values = predictions.select(f.col(y_axis)).rdd.flatMap(lambda x: x).collect()

	plt.scatter(x_values, y_values, c=cluster_labels, cmap='viridis')

	plt.title('K-Means Clustering')
	plt.xlabel(X_axis)
	plt.ylabel(y_axis)
	plt.legend()
	plt.show()

def cluster(data, k):
	kmeans = KMeans(featuresCol="features", k=k)
	model = kmeans.fit(data)
	centers = model.clusterCenters()
	predictions = model.transform(data)
	
	return predictions


data = open_data()
analyse_for_k(data)

k = 3
predictions = cluster(data, k)

X_axis = 'op'
y_axis = 'wordcount'
plot(predictions, X_axis, y_axis)

spark.stop()