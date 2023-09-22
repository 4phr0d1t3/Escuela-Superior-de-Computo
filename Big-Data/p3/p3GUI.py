# from tkinter import *
from tkinter import Tk
from tkinter import Label
from tkinter import Text
from tkinter import Button
from tkinter import filedialog
from pyspark.sql import SparkSession

def calculateDistances(id, df):
	targetRow = df.take(id + 1)[-1]
	distances = []
	for i in range(df.count()):
		sum = 0
		for feature in range(2, len(df.columns)):
			row = df.take(i + 1)[-1]
			sum += pow(targetRow[feature] - row[feature], 2)
		sum = pow(sum, 1/2)
		distances.append(sum)

		porcentile = i/df.count() * 100
		formatted_porcentile = "{:.2f}".format(porcentile)
		print('[', formatted_porcentile,'% ]')

	return distances

def knn(id, k, df):
	dist = calculateDistances(id, df)
	ids = sorted(range(len(dist)), key=lambda sub: dist[sub])[:(k+1)]
	m = 0
	print("Los mas cercanos son: ")
	print("\tdistancia\tid\tdiagnostico")
	for i in ids[1:k+1]:
		targetRow = df.take(i + 1)[-1]
		print(dist[i], '\t', i, '\t', targetRow['diagnosis'])
		if targetRow['diagnosis'] == 'M':
			m += 1
	print("Con", m, "M y", k-m, "B")
	print("Predicho: M") if m > k/2 else print("Predicho: B")

	row = df.take(id + 1)[-1]
	print("Real: ", row['diagnosis'])

def start(path):
	spark = SparkSession.builder.appName('practica1').getOrCreate()
	df = spark.read.csv(path, header=True, inferSchema=True)
	id = input_id.get(1.0, "end-1c")
	k = input_k.get(1.0, "end-1c")

	id = int(id)
	k = int(k)

	knn(id, k, df)

def browseFiles():
	filename = filedialog.askopenfilename(
		initialdir = "./",
		title = "Select a File",
		filetypes = (
			("Text files","*.csv*"),
			("all files","*.*")
	))
	path = filename
	label_file_explorer.configure(text="File Opened: " + path)
	start(path)

window = Tk()

window.title('File Explorer')

window.geometry("512x512")

label_file_explorer = Label(
	window,
	text = "File Explorer using Tkinter",
	width = 64,
	height = 4
)

label_id = Label(
	window,
	text = "Id del dato a clasificar",
	width = 64,
	height = 4
)

input_id = Text(
	window,
	height = 4,
	width = 20
)

label_k = Label(
	window,
	text = "k a usar",
	width = 64,
	height = 4
)

input_k = Text(
	window,
	height = 4,
	width = 20
)
  
button_explore = Button(
	window,
	text = "Browse Files",
	command = browseFiles
)

label_file_explorer.grid(column = 1, row = 1)

label_id.grid(column = 1, row = 2)
input_id.grid(column = 1, row = 3)

label_k.grid(column = 1, row = 4)
input_k.grid(column = 1, row = 5)

button_explore.grid(column = 1, row = 6)

window.mainloop()
