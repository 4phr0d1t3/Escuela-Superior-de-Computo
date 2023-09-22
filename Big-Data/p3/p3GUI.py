# from tkinter import *
from tkinter import Tk
from tkinter import Label
from tkinter import Text
from tkinter import Button
from tkinter import filedialog
from pyspark.sql import SparkSession

import numpy as np
import matplotlib.pyplot as plt

def hypothesis(x, m, b):
	return m * x + b

def start(path):
	spark = SparkSession.builder.appName('practica2').getOrCreate()
	df = spark.read.csv(path, header=True, inferSchema=True)

	x = np.array(df.select("age").rdd.map(lambda row: row[0]).collect())
	y = np.array(df.select("charges").rdd.map(lambda row: row[0]).collect())

	m = 0
	b = 0
	
	learning_rate = 0.001
	epochs = 1000
	n = len(x)
	for _ in range(epochs):
		d_m = -(1 / n) * np.sum(x * (y - hypothesis(x, m, b)))
		d_b = -(1 / n) * np.sum(y - hypothesis(x, m, b))
		m -= learning_rate * d_m
		b -= learning_rate * d_b

	plt.scatter(x, y, label='Data points')
	plt.plot(x, [hypothesis(xi, m, b) for xi in x], color='red', label='Regression line')
	plt.xlabel('Age')
	plt.ylabel('Charges')
	plt.legend()
	plt.show()

	label_m.configure(text=m)
	label_b.configure(text=b)

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

def predict():
	new_x = input_new_x.get(1.0, "end-1c")
	m = label_m.cget("text")
	b = label_b.cget("text")

	new_x = int(new_x)
	m = float(m)
	b = float(b)

	predicted_y = hypothesis(new_x, m, b)
	label_predicted.configure(text="Para la edad de " + str(new_x) + " se predice una Y: ")
	label_y.configure(text=str(predicted_y))

window = Tk()

window.title('File Explorer')

window.geometry("512x512")

label_file_explorer = Label(
	window,
	text = "File Explorer using Tkinter",
	width = 64,
	height = 4
)

input_id = Text(
	window,
	height = 4,
	width = 20
)

label_new_x = Label(
	window,
	text = "Nuevo dato a predecir",
	width = 64,
	height = 4
)

input_new_x = Text(
	window,
	height = 4,
	width = 20
)
  
button_explore = Button(
	window,
	text = "Apply Linear Regression",
	command = browseFiles
)

label_m = Label(
	window,
	text = "m",
	height = 1,
	width = 20
)

label_b = Label(
	window,
	text = "b",
	height = 1,
	width = 20
)

button_predict = Button(
	window,
	text = "Predict",
	command = predict
)

label_predicted = Label(
	window,
	text = "y",
	height = 2,
	width = 50
)

label_y = Label(
	window,
	text = ".",
	height = 2,
	width = 50
)

label_file_explorer.grid(column = 1, row = 1)
button_explore.grid(column = 1, row = 2)

label_m.grid(column = 1, row = 3)
label_b.grid(column = 1, row = 4)

label_new_x.grid(column = 1, row = 5)
input_new_x.grid(column = 1, row = 6)
button_predict.grid(column = 1, row = 7)

label_predicted.grid(column = 1, row = 8)
label_y.grid(column = 1, row = 9)

window.mainloop()
