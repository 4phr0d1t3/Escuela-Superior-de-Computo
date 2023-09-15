import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tkinter import Tk
from tkinter import Label
from tkinter import Text
from tkinter import Button
from tkinter import filedialog
from pyspark.sql import SparkSession

def obtain_features(df):
	input_cols = ['op','co','ex','ag','ne','wordcount']
	vect_assembler = VectorAssembler(
		inputCols= input_cols,
		outputCol= "features"
	)
	final_data = vect_assembler.transform(df)

	return final_data

def open_data(path, spark):
	df = spark.read.csv(
		path,
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

	canvas = FigureCanvasTkAgg(plt.gcf(), master=window)
	canvas.draw()
	canvas.get_tk_widget().pack()

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

	canvas = FigureCanvasTkAgg(plt.gcf(), master=window)
	canvas.draw()
	canvas.get_tk_widget().pack()

def cluster(data, k):
	kmeans = KMeans(featuresCol="features", k=k)
	model = kmeans.fit(data)
	centers = model.clusterCenters()
	predictions = model.transform(data)
	
	return predictions

def analisys(path):
	spark = SparkSession.builder.appName('practica2').getOrCreate()

	data = open_data(path, spark)
	analyse_for_k(data)

	spark.stop()

def classify(path):
	spark = SparkSession.builder.appName('practica2').getOrCreate()

	data = open_data(path, spark)
	k = 3
	predictions = cluster(data, k)

	X_axis = 'op'
	y_axis = 'wordcount'
	plot(predictions, X_axis, y_axis)

	spark.stop()

def browseFilesA():
	filename = filedialog.askopenfilename(
		initialdir = "./",
		title = "Select a File",
		filetypes = (
			("Text files","*.csv*"),
			("all files","*.*")
	))
	path = filename
	label_file_explorer.configure(text="File Opened: " + path)
	analisys(path)

def browseFilesC():
	filename = filedialog.askopenfilename(
		initialdir = "./",
		title = "Select a File",
		filetypes = (
			("Text files","*.csv*"),
			("all files","*.*")
	))
	path = filename
	label_file_explorer.configure(text="File Opened: " + path)
	classify(path)

window = Tk()

window.title('k means')

window.geometry("512x512")

label_file_explorer = Label(
	window,
	text = "File Explorer using Tkinter",
	width = 64,
	height = 4
)

label_id = Label(
	window,
	text = "Clasificador Kmeans",
	width = 64,
	height = 4
)

button_analisys = Button(
	window,
	text = "Para mostrar analisis de k",
	command = browseFilesA
)

button_classify = Button(
	window,
	text = "Clasificar con k = 3",
	command = browseFilesC
)

label_file_explorer.grid(column = 1, row = 1)

label_id.grid(column = 1, row = 2)

button_analisys.grid(column = 1, row = 3)
button_classify.grid(column = 1, row = 4)

window.mainloop()