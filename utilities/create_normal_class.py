import os, sys
from os import listdir
import shutil
import pandas as pd

#Create normal class based on the Labels given in DCSASS dataset
DIR = 'DCSASS Dataset/'
classes = os.listdir(DIR)
OUT = 'DCSASS Dataset/Normal/'
if not os.path.isdir(OUT):
	os.mkdir(OUT)

for folder in classes:
	label = pd.read_csv("Labels/"+folder+".csv", header = None)
	FOLDER = DIR + folder + '/'
	for i in range(len(label)):
		if label[2][i] == 0:
			PATH = FOLDER + label[0][i] + '.mp4'
			dest = shutil.move(PATH, OUT, copy_function = shutil.copytree) 
			print(dest)
