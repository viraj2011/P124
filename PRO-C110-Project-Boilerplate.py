# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the modelp

import tensorflow as tf
model = tf.keras.models.load_model('keras_model.h5')

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		#resize the frame

		expanded_image = np.expand_dims(resized_image, axis=0)
		
		# expand the dimensions

		image = cv2.imread('your_image.jpg')

		# Define the new dimensions (e.g., width and height)

		new_width = 224
		new_height = 224

        # Resize the image
		resized_image = cv2.resize(image, (new_width, new_height))

		# normalize it before feeding to the model
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		
		normalized_image = (expanded_image / 255.0 - mean) / std
	
		# get predictions from the model
		prediction = model.predict(frame)		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
