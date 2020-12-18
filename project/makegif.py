import cv2
import tkinter as tk
from tkinter import filedialog
import os
from tkinter.filedialog import askopenfilenames

filenames = askopenfilenames(title = "Open 'xls' or 'xlsx' file") 

image_folder = 'ps_d'
video_name = 'psd.avi'

images = filenames
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name,0, 5 ,(width,height))


for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()