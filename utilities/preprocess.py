import os, sys
from os import listdir

#Extract the frames from the video so the program will be able to read the video data without needing to open each video
#This preprocessing saves time during training and also reducing RAM load.
main_folder = 'DCSASS Dataset/'
fps = 25

for folder in listdir(main_folder):
	folder_url = main_folder+folder
	for file in listdir(folder_url):
		file_url = folder_url+'/'+file
		if os.path.isfile(file_url):
			if not os.path.isdir(file_url[:-4]):
				os.mkdir(file_url[:-4])
				print("Extracting frames from ", file_url)
				os.system('ffmpeg -i "{}" -vf fps={} "{}/%05d.jpg"'.format(file_url, fps, file_url[:-4]))
				#this will remove the video file after taking the pics so be careful and keep another backup
				os.remove(file_url)