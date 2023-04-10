# modified script from the imutils package to utilize picamera2 rather than the recently outdated (September 2022) picamera package

# import the necessary packages
from picamera2 import Picamera2, Preview
from threading import Thread
import cv2
import time

class PiVideoStream:
#	global preview_config, capture_config
	def __init__(self, resolution=(1980, 1080), framerate=32, **kwargs):
		# initialize the camera
		self.camera = Picamera2()

		# set camera parameters
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		self.preview_config = self.camera.create_preview_configuration({"size": (1640, 1232), 'format': 'RGB888'})
		self.capture_config = self.camera.create_still_configuration()

		# set optional camera parameters (refer to PiCamera docs)
		for (arg, value) in kwargs.items():
			setattr(self.camera, arg, value)

		# initialize the stream
		self.camera.start()
		time.sleep(2)

		# initialize the frame and the variable used to indicate
		# if the thread should be stopped
		self.frame = None
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped

		# grab the frame from the stream and clear the stream in
		# preparation for the next frame
		while self.stopped != True:
			self.camera.set_controls({'ColourGains': (1.5, 1.5)})
			buffers, metadata = self.camera.switch_mode_and_capture_buffers(self.preview_config, ["main"])

			arr = self.camera.helpers.make_array(buffers[0], self.preview_config["main"])

			self.frame = arr
			time.sleep(1)


	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
