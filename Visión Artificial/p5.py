import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.array([[0,0,0,0,1,1,1,1],
			  [0,0,1,1,0,0,1,1],
			  [0,1,0,1,0,1,0,1],
			  [1,1,1,1,1,1,1,1]])

y = np.array([0,1,1,1,0,0,0,1])
w = np.array([1.0,1.0,1.0,1.0])

learningRate = 0.3
err = 0.0

def askForWeights():
	for i in range(4):
		if i == 3:
			w[i] = float(input("Ingrese el bias: "))
		else:
			w[i] = float(input("Ingrese el peso w_" + str(i+1) + ": "))

def printWeights(epoch):
	print("Pesos de la epoca", epoch)
	for i in range(len(w)):
		if i == len(w)-1:
			print("Bias: {:.1f}".format(w[i]))
		else:
			print("w_", i+1, ": {:.1f}".format(w[i]))

def plotCube():
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x[0][:], x[1][:], x[2][:], c=y)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

def perceptron(row):
	sum = 0
	for i in range(4):
		sum += w[i] * x[i][row]
	if sum < 0:
		return 0
	else:
		return 1

def weightsUpdate(row):
	for i in range(4):
		w[i] += learningRate * err * x[i][row]

def finalPrompt(lastEpoch, maxNumEpochs):
	if lastEpoch == maxNumEpochs:
		print("Maximo de epocas alcanzado")
	else:
		print("Solucion encontrada en la epoca ", lastEpoch)

	printWeights(lastEpoch)
	x0 = np.linspace(0,1,10)
	x1 = np.linspace(0,1,10)
	x2 = np.linspace(0,1,10)

	X, Y = np.meshgrid(x0, x1)
	Z = (-w[0]*X - w[1]*Y - w[3]) / w[2]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x[0][:], x[1][:], x[2][:], c=y)
	ax.plot_surface(X, Y, Z, alpha=0.5)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()

y_pred = np.array([0,0,0,0,0,0,0,0])
maxNumEpochs = 10
lastEpoch = 0

askForWeights()
for epoch in range(maxNumEpochs):
	#printWeights(epoch)
	for i in range(8):
		y_pred[i] = perceptron(i)
		err = y[i] - y_pred[i]
		weightsUpdate(i)
	if np.array_equal(y_pred, y):
		lastEpoch = epoch
		break

finalPrompt(lastEpoch, maxNumEpochs)
