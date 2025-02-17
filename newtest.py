import cv2
import time
from tensorflow import keras
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os

import pyttsx3
text_speech = pyttsx3.init()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Further suppress TensorFlow logs

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
folder = "Data/C"
counter = 0
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y"]

last_prediction = None
stable_count = 0
threshold = 5  # Number of consecutive frames for stability
last_print_time = time.time()
delay = 1  # One second delay


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)  # Detect hands in the frame
    
    if hands:  # If a hand is detected
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Get bounding box of the hand
        
        # Create white background for the image crop
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # Crop the hand from the image
        
        # Handle aspect ratio to avoid stretching
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:  # If height is larger than width
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:  # If width is larger than height
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Get the prediction from the classifier
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        current_prediction = labels[index]

        # Only print if the prediction remains stable for a few frames and 1 second has passed
        if current_prediction == last_prediction:
            stable_count += 1
        else:
            stable_count = 0
        
        if stable_count == threshold and time.time() - last_print_time >= delay:
            print(current_prediction)
            last_print_time = time.time()
            text_speech.say(current_prediction)
            text_speech.runAndWait()
        
        last_prediction = current_prediction

        # Draw prediction result and bounding box on the output image
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)
        
        # Display the cropped and white image windows for debug purposes
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)  # Show the final output image
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()





