# import the necessary packages
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import json
import time
import cv2
import os

# define the configuration dictionary
CONFIG = {
	# define the paths to the CNN
	"model_path": os.path.sep.join(["assets", "pokedex.model"]),
	# define the set of class labels (these were derived from the
	# label binarizer from the previous post)
	"labels": ['bulbasaur', 'charizard', 'charmander', 'gengar', 'jigglypuff', 'lapras',
		'lucario', 'mew', 'mewtwo', 'nidoran', 'oshawatt', 'pikachu', 'piplup',
		 'reshiram', 'snorlax', 'squirtle', 'zekrom'],

	# define the path to the JSON database of Pokemon info
	"db_path": os.path.sep.join(["assets", "pokemon_db.json"]),
	# define the number of seconds to display the Pokemon information
	# for after a classification
	"display_for": 24 * 10,
	# define the paths to the Pokedex background and mask images,
	# respectively
	"pdx_bg": os.path.sep.join(["assets", "NEW_BACKGROUND_1280_348_AI.png"]),
	"pdx_mask": os.path.sep.join(["assets", "NEW_MASK_1280_348_AI.png"]),

	# (x, y)-coordinates of where the video stream location lives
	"pdx_vid_x": 15, #25
	"pdx_vid_y": 10, #125
	# (x, y)-coordinates of where the Pokemon's name, height,
	# weight, and accuracy will be drawn
	"pdx_name_x": 650, #400
	"pdx_name_y": 90, #167
	"pdx_height_x": 650, #400
	"pdx_height_y": 270, #213
	"pdx_weight_x": 990, #485
	"pdx_weight_y": 270, #213
	"pdx_accuracy_x": 990,
	"pdx_accuracy_y": 90,
	# color of all text drawn on the Pokedex
	"pdx_color": (255, 255, 255)[::-1]
}

# initialize the current frame from the video stream, a boolean used
# to indicated if the screen was clicked, a frame counter, and the
# predicted class label
frame = None
clicked = False
counter = 0
predLabel = None
percentage = ""

def on_click(event, x, y, flags, param):
	# grab a reference to the global variables
	global frame, clicked, predLabel, percentage
	# check to see if the left mouse button was clicked, and if so,
	# perform the classification on the current frame
	if event == cv2.EVENT_LBUTTONDOWN:
		predLabel = classify(preprocess(frame))
		clicked = True

def preprocess(image):
	# preprocess the image
	image = cv2.resize(image, (96, 96))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# return the pre-processed image
	return image

def classify(image):
	# classify the input image
	global percentage
	proba = model.predict(image, batch_size=5)[0]
	idx = np.argmax(proba)
#	print(proba)
	percentage = "{:.2f}% ".format(proba[idx] * 100)
	# return the class label with the largest predicted probability
	return CONFIG["labels"][np.argmax(proba)]

# load the pokedex background image and grab its dimensions
print("[INFO] booting pokedex...")
pokedexBG = cv2.imread(CONFIG["pdx_bg"])
(bgH, bgW) = pokedexBG.shape[:2]
# load the pokedex mask (i.e., the part where the video will go and)
# binarize the image
pokedexMask = cv2.imread(CONFIG["pdx_mask"])
pokedexMask = cv2.cvtColor(pokedexMask, cv2.COLOR_BGR2GRAY)
pokedexMask = cv2.threshold(pokedexMask, 128, 255,
	cv2.THRESH_BINARY)[1]

# load the trained convolutional neural network and pokemon database
print("[INFO] loading pokedex model...")
model = load_model(CONFIG["model_path"])
db = json.loads(open(CONFIG["db_path"]).read())
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
# setup the mouse callback function
cv2.namedWindow("Pokedex")
cv2.setMouseCallback("Pokedex", on_click)

# loop over the frames from the video stream
while True:
	# if the window was clicked "freeze" the frame and increment
	# the total number of frames the stream has been frozen for
	if clicked and count < CONFIG["display_for"]:
		count += 1
	else:
		# grab the frame from the threaded video stream and resize
		# it to have a maximum width of 260 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=450)
		(fgH, fgW) = frame.shape[:2]
		# reset our frozen count, clicked flag, and predicted class
		# label
		count = 0
		clicked = False
		predLabel = None

	# create the pokedex image by first allocating an empty array
	# with the same dimensions of the background and then applying
	# array slicing to insert the frame
	pokedex = np.zeros((bgH, bgW, 3), dtype="uint8")
	pokedex[CONFIG["pdx_vid_y"]:CONFIG["pdx_vid_y"] + fgH,
		CONFIG["pdx_vid_x"]:CONFIG["pdx_vid_x"] + fgW] = frame
	# take the bitwise AND with the mask to create the rounded
	# corners on the frame + remove any content that falls outside
	# the viewport of the video display, then take the bitwise OR
	# to add the frame to add image
	pokedex = cv2.bitwise_or(pokedex, pokedexBG)

	# if the predicted class label is not None, then draw the Pokemon
	# stats on the Pokedex
	if predLabel is not None:
		# draw the name of the Pokemon
		pokedex = cv2.putText(pokedex, predLabel.capitalize(),
			(CONFIG["pdx_name_x"], CONFIG["pdx_name_y"]),
			cv2.FONT_HERSHEY_SIMPLEX, 1.25, CONFIG["pdx_color"], 2)
		# draw the Pokemon's height
		pokedex = cv2.putText(pokedex, db[predLabel]["height"],
			(CONFIG["pdx_height_x"], CONFIG["pdx_height_y"]),
			cv2.FONT_HERSHEY_SIMPLEX, .9, CONFIG["pdx_color"], 1)
		# draw the Pokemon's weight
		pokedex = cv2.putText(pokedex, db[predLabel]["weight"],
			(CONFIG["pdx_weight_x"], CONFIG["pdx_weight_y"]),
			cv2.FONT_HERSHEY_SIMPLEX, 0.9, CONFIG["pdx_color"], 1)
		# draw the Pokemon's accuracy
		pokedex = cv2.putText(pokedex, percentage, (CONFIG["pdx_accuracy_x"],
			CONFIG["pdx_accuracy_y"]), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
			CONFIG["pdx_color"], 2)

	# show the output frame
	cv2.imshow("Pokedex", pokedex)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
print("[INFO] shutting down pokedex...")
cv2.destroyAllWindows()
vs.stop()
