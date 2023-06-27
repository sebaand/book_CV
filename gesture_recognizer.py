# import cv2
# import math

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# IMAGE_FILENAMES = 'Book_word_identification\Finger Pointing.PNG'

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
#   cv2.imshow('image',img)
#   cv2.waitKey(0)
  

# # Preview the images.
# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)

import mediapipe as mp
import numpy as np


# Load the input image from an image file.
mp_image = mp.Image.create_from_file('Book_word_identification\Finger Pointing.PNG')

# Load the input image from a numpy array.
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.int8(mp_image))

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the image mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE)
with GestureRecognizer.create_from_options(options) as recognizer:
  gesture_recognition_result = recognizer.recognize(mp_image) 
  # The detector is initialized. Use it here.