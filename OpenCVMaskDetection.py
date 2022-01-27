import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import requests


font = cv2.FONT_HERSHEY_SIMPLEX

# Load trained model and cascading
model = tf.keras.models.load_model('./Saved Model and Cascading/mask_detection.h5')
frontalface_haar_cascade = cv2.CascadeClassifier('./Saved Model and Cascading/haarcascade_frontalface_default.xml')
label_dict = {0 : 'Mask', 1: 'No Mask'}

# url to use phone camera
url = "http://192.168.5.103:8080/video"
camera = cv2.VideoCapture(0)              # Replace url = 0 if not using mobile camera

while True:
    success, frame = camera.read()
    if success:
        face = frontalface_haar_cascade.detectMultiScale(frame,
		        	                                    scaleFactor= 1.2,
		                                                minNeighbors=3)

        for (x,y,w,h) in face:
            img = frame[y:y+h, x:x+w]
            resize_image = cv2.resize(img, (224,224))
            normalize_img = resize_image/225
            img_pixels = normalize_img.reshape(224,224,3)
            img_pixels = np.expand_dims(img_pixels, axis=0)
  
            predictions = model.predict(img_pixels)
            mask_label = np.argmax(predictions)

            if mask_label == 0:
                cv2.rectangle(frame, (x,y), (x+w,y+h),(0,225,0),2)
                cv2.putText(frame,'Mask', (x, y), font, 0.5,(0,225,0), 1)

            if mask_label == 1:
                cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,225),2)
                cv2.putText(frame,'No Mask', (x, y), font, 0.5,(0,0,225), 1)
        
            # Display the resulting frame
            cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cv2.destroyAllWindows()

