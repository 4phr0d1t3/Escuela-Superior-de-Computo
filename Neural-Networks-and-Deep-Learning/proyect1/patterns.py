import cv2
import numpy as np

def huInvariableMoments(imagen):
	grey = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
	moments = cv2.moments(grey)
	huMoments = cv2.HuMoments(moments)
	huMoments = -np.sign(huMoments) * np.log10(np.abs(huMoments))
	return huMoments.flatten()

patternsCircle = []
patternsSquare = []
patternsStar = []
patternsTriangle = []

for i in range(0, 3719, round(18.6)):
	pathCircles = f"./shapes/circle/{i}.png"
	pathSquares = f"./shapes/square/{i}.png"
	pathStars = f"./shapes/star/{i}.png"
	pathTriangles = f"./shapes/triangle/{i}.png"
	
	imgCircle = cv2.imread(pathCircles)
	imgSquare = cv2.imread(pathSquares)
	imgStar = cv2.imread(pathStars)
	imgTriangle = cv2.imread(pathTriangles)
	
	huMomentsCircle = huInvariableMoments(imgCircle)
	huMomentsSquare = huInvariableMoments(imgSquare)
	huMomentsStar = huInvariableMoments(imgStar)
	huMomentsTriangle = huInvariableMoments(imgTriangle)
	
	patternsCircle.append(huMomentsCircle)
	patternsSquare.append(huMomentsSquare)
	patternsStar.append(huMomentsStar)
	patternsTriangle.append(huMomentsTriangle)

patternsCircle = np.transpose(patternsCircle)
patternsSquare = np.transpose(patternsSquare)
patternsStar = np.transpose(patternsStar)
patternsTriangle = np.transpose(patternsTriangle)

import pandas as pd

dfC = pd.DataFrame()
dfSq = pd.DataFrame()
dfSt = pd.DataFrame()
dfT = pd.DataFrame()

for i, ndarray in enumerate(patternsCircle):
	column_name = f"Hu{i + 1}"
	dfC[column_name] = ndarray.flatten()
dfC["Shape"] = "Circle"

for i, array in enumerate(patternsSquare):
	column_name = f"Hu{i + 1}"
	dfSq[column_name] = array.flatten()
dfSq["Shape"] = "Square"

for i, array in enumerate(patternsStar):
	column_name = f"Hu{i + 1}"
	dfSt[column_name] = array.flatten()
dfSt["Shape"] = "Star"

for i, array in enumerate(patternsTriangle):
	column_name = f"Hu{i + 1}"
	dfT[column_name] = array.flatten()
dfT["Shape"] = "Triangle"

df = pd.DataFrame()
df = pd.concat([df, dfC])
df = pd.concat([df, dfSq])
df = pd.concat([df, dfSt])
df = pd.concat([df, dfT])

df.to_csv('data.csv', index=False)
