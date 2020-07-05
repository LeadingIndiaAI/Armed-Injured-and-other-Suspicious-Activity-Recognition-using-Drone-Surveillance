import os, sys
from os import listdir
import shutil

DIR = 'DCSASS Dataset/'

#Use this to move all video clips from the subfolders into the main class folders for DCSASS dataset
for folder in os.listdir(DIR):
	folder_url = DIR + folder + '/'
	for subfolder in os.listdir(folder_url):
		subfolder_url = folder_url + subfolder + '/'
		for clip in os.listdir(subfolder_url):
			clip_url = subfolder_url + clip
			shutil.move(clip_url, folder_url, copy_function = shutil.copytree)
		#the empty folders will be deleted after clips are moved out of it
		os.rmdir(subfolder_url)