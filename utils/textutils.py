from PIL import ImageFont, ImageDraw, Image  
import cv2  
import numpy as np  


def write(image,text,pos,font=None):
	text_to_show = text  
	x,y = pos
	# Convert the image to RGB (OpenCV uses BGR)  
	cv2_im_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  

	# Pass the image to PIL  
	pil_im = Image.fromarray(cv2_im_rgb)  

	draw = ImageDraw.Draw(pil_im)  
	# use a truetype font  
	
	# Draw the text  
	draw.text((x,y), text_to_show, font=font)  

	# Get back the image to OpenCV  
	cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)  

	return cv2_im_processed

if __name__=='__main__':
	im = np.zeros((480, 600, 3), dtype=np.uint8)
	font_30 = ImageFont.truetype('/home/kayshu/OpenCVProjects/quickmaths/fonts/raleway/Raleway-Light.ttf', 30)  
	
	im =  write(im,"Hello\nWorld Σ ",(10,20),font_30)
	cv2.imshow('Fonts', im)  
	cv2.waitKey(0)  

	cv2.destroyAllWindows()  
