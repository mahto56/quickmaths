# import the necessary packages
from __future__ import print_function
from datetime import datetime
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
from imutils import contours
import cv2
import numpy as np
from PIL import ImageFont
import logging
from quickmaths.utils import bbutils,hist,predict,textutils
from quickmaths.utils.compare import *
from keras.models import load_model
import time;



class app():
	def __init__(self,logger,src):
		self.vs = WebcamVideoStream(src)
		self.fps = FPS().start()
		self.logger = logger
		self.point1=None
		self.point2=None;
		self.drag = False;
		self.rect = None;
		self.img=None
		self.thresh = None
		self.thresh2 = None
		self.roi = None
		self.select = False;
		self.old = None
		self.imdiffs = []
		self.static = False
		self.stacked = None
		self.msg = "Draw a region of interest using mouse"
		self.mode = "continous"
		self.res = ""
		self.angles=[]
		self.angle= 0
		# self.model = load_model('/home/kayshu/OpenCVProjects/quickmaths/models/model_0.1v4.h5')
		self.model = load_model('/home/kayshu/OpenCVProjects/quickmaths/models/model_0.1v7.h5')
		
		self.predictor = predict.Predictor(self.model)

		# self.hist = hist.Hist()
		logger.info('Hello world!')
		logger.debug('Womp womp!')
		cv2.namedWindow("Frame")
		cv2.setMouseCallback("Frame",self.handler,None)

	
	def run(self):
		self.vs.start()
		self.fps.start()
		draw = True
		font_30 = ImageFont.truetype('/home/kayshu/OpenCVProjects/quickmaths/fonts/raleway/Raleway-Light.ttf', 30)  
		font_20 = ImageFont.truetype('/home/kayshu/OpenCVProjects/quickmaths/fonts/raleway/Raleway-Italic.ttf', 30)  
	
		while True:
			frame = self.vs.read()
			orig = frame.copy()
			self.img = orig

			if self.drag and self.rect:
				(x1,y1),(x2,y2) = self.rect
				roi = orig[y1:y2,x1:x2].copy()				
				orig = cv2.blur(orig, (150, 150))
				orig[y1:y2,x1:x2] = roi

				cv2.rectangle(orig,(x1,y1),(x2,y2),(255,255,255),1)
				orig = self.draw_border(orig,(x1,y1),(x1,y2),(x2,y1),(x2,y2),30)

			elif self.select:
				(x1,y1),(x2,y2) = self.rect

				roi = orig[y1:y2,x1:x2].copy()
				orig = cv2.blur(orig, (150, 150))
				orig[y1:y2,x1:x2] = roi


				if roi.size == 0:
					self.select = False
					continue
					
				gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
				# edged = imutils.auto_canny(gray)
				
				#draw histogram
				# self.hist.draw(gray)



				kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,21))
				blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)

				_,thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
				thresh = cv2.GaussianBlur(thresh,(1,1),cv2.BORDER_DEFAULT)
			
				thresh = cv2.dilate(thresh,np.ones((3,3)))
				
				thresh = cv2.erode(thresh,np.ones((2,2)))
				
				# angle = bbutils.get_houghlines_angle(thresh)

				# self.angles.append(angle)
					
				# if len(self.angles) >10:
				# 	self.angle = int(sum(self.angles)/10)
				# 	self.angles = []

				# thresh = bbutils.rotate(thresh,self.angle)	
				# bbs = initial_boxes(thresh)
				final_bbs,symlist = bbutils.get_symlist(thresh.copy())
				self.res,self.stacked,updated_symlist  = self.predictor.predict(orig[y1:y2,x1:x2],symlist,orig,x1,y1)
				

				self.thresh2 = thresh.copy()
				final_bbs = np.array(final_bbs)
				# self.logger.info(f'Total contours: {final_bbs.shape[0]}')
				for img,name,x,y,xw,yh in updated_symlist:
					# (x,y),(xw,yh) = bb
					cv2.rectangle(orig,(x1+x,y1+y),(x1+xw,y1+yh),(0,255,0),2)
					# cv2.rectangle(roi,(x,y),(xw,yh),(0,255,0),1)
					cv2.rectangle(thresh,(x,y),(xw,yh),(255,255,255),1)
				self.thresh = thresh

				self.roi = roi
				cv2.imshow("roi",thresh)
				cv2.imshow("roi2",roi)
				
				cv2.rectangle(orig,(x1,y1),(x2,y2),(255,255,255),1)
				orig = self.draw_border(orig,(x1,y1),(x1,y2),(x2,y1),(x2,y2),30)

				if self.static:
					self.select = False

			else:
				# cv2.putText(orig,self.msg,(70,200),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
				orig = textutils.write(orig,self.msg,(70,200),font_30)

			orig = textutils.write(orig,f'Mode: {self.mode}',(20,10),font_20)
			orig = textutils.write(orig,f'Result: {self.res}',(20,420),font_20)
			
			# for i in range(200):
			# 	frame = imutils.resize(frame, width=400)
			cv2.imshow("Frame",orig)
			if self.stacked is not None:
				cv2.imshow('predict',self.stacked)
		
			# cv2.imshow("Frame",edged)
			# cv2.imshow('dst',dst)

			self.fps.update()

			key = cv2.waitKey(1) 

			if key == 27:
				break

			if key == ord('s'):
				print(self.static)
				self.static = not self.static
				self.mode = "static" if self.static else "continous"
				if not self.static and self.rect:
					self.select = True

			if key == ord('c'):
				cv2.imwrite(f'crop_{int(time.time())}.png',orig)
				cv2.imwrite(f'stacked_{int(time.time())}.png',self.stacked)
				cv2.imwrite(f'thresh_{int(time.time())}.png',self.thresh)
				cv2.imwrite(f'roi_{int(time.time())}.png',self.roi)
				cv2.imwrite(f'thresh2_{int(time.time())}.png',self.thresh2)
				
				self.logger.info("Image cropped!")
			

		
		self.fps.stop()
		self.vs.stop()
		cv2.destroyAllWindows()
		self.logger.info("elapsed time: {:.2f}".format(self.fps.elapsed()))
		self.logger.info("approx. FPS: {:.2f}".format(self.fps.fps()))


	# Mouse handler to draw rectangle on live video
	def handler(self,event,x,y,flags,param):

		if event == cv2.EVENT_LBUTTONDOWN and not self.drag:
			print("DOWN")
			self.select = False
			self.point1 = (x,y)
			self.drag = True

		if event == cv2.EVENT_MOUSEMOVE  and self.drag :
			print("MOVE")
			self.point2 = (x,y)
			self.rect = (self.point1,self.point2)

		if event == cv2.EVENT_LBUTTONUP and self.drag:
			print("UP")
			self.point2 = (x,y)
			
			self.drag = False
			if self.point1 != self.point2:
				self.select = True
				self.rect = (self.point1,self.point2)

	def draw_border(self,img, point1, point2, point3, point4, line_length):

		x1, y1 = point1
		x2, y2 = point2
		x3, y3 = point3
		x4, y4 = point4    

		# cv2.circle(img, (x1, y1), 5, (255, 0, 255), -1)    #-- top_left
		# cv2.circle(img, (x2, y2), 5, (255, 0, 255), -1)    #-- bottom-left
		# cv2.circle(img, (x3, y3), 5, (255, 0, 255), -1)    #-- top-right
		# cv2.circle(img, (x4, y4), 5, (255, 0, 255), -1)    #-- bottom-right

		cv2.rectangle(img, (x1, y1), (x1+5 , y1 + line_length), (255, 255, 255), -1)  #-- top-left
		cv2.rectangle(img, (x1, y1), (x1 + line_length , y1+5), (255, 255, 255), -1)

		cv2.rectangle(img, (x2, y2), (x2+5 , y2 - line_length), (255, 255, 255), -1)  #-- bottom-left
		cv2.rectangle(img, (x2, y2), (x2 + line_length , y2-5), (255, 255, 255), -1)

		cv2.rectangle(img, (x3, y3), (x3 - line_length, y3+5), (255, 255, 255), -1)  #-- top-right
		cv2.rectangle(img, (x3, y3), (x3-5, y3 + line_length), (255, 255, 255), -1)

		cv2.rectangle(img, (x4, y4), (x4-5 , y4 - line_length), (255, 255, 255), -1)  #-- bottom-right
		cv2.rectangle(img, (x4, y4), (x4 - line_length , y4-5), (255, 255, 255), -1)

		return img