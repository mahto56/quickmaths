import numpy as np
import cv2
import imutils
import math
from scipy import ndimage


# detect if input boundingBox contains a dot
def is_dot(bb):
	(x, y), (xw, yh) = bb
	area = (yh - y) * (xw - x)
	w = xw-x
	h = yh-y
	return w < 25 and h < 25 and float(w)/h < 1.5 and float(h)/w < 1.5 or area < 200 and 0.5 < (xw - x)/(yh - y) < 2 and abs(xw - x) < 20 and abs(yh - y) < 20  # 100 is migical number

# detect if input boundingBox contains a vertical bar
def is_verticalbar(bb):
	(x, y), (xw, yh) = bb
	return (yh - y) / (xw - x) > 2  

# detect if a given boundingBox contains a horizontal bar
def is_horizontalbar(bb):
    (x, y), (xw, yh) = bb
    return (xw - x) / (yh - y) > 3.5

# detect if input boundingBox contains a square (regular letters, numbers, operators)
def is_square(bb):
    (x, y), (xw, yh) = bb
    return (xw - x) > 8 and (yh - y) > 8 and 0.5 < (xw - x)/(yh - y) < 2

# detect if input three boundingBoxes are a division mark
def is_divisionmark(bb, bb1, bb2):
    (x, y), (xw, yh) = bb
    (x1, y1), (xw1, yh1) = bb1
    (x2, y2), (xw2, yh2) = bb2

    cenY1 = y1 + (yh1 - y1) / 2
    cenY2 = y2 + (yh2 - y2) / 2

    return (
    	is_horizontalbar(bb) and is_dot(bb1) and is_dot(bb2)
    	and x < x1 < x2 < xw and max(y1, y2) > y and min(y1, y2) < y
    	# and max(y1, y2) - min(y1, y2) < 3.2 * abs(xw - x)
    )

# detect if input two boundingBoxes are a lowercase i
def is_letteri(bb, bb1):
	(x, y), (xw, yh) = bb
	(x1, y1), (xw1, yh1) = bb1
	return (
		((is_dot(bb) and is_verticalbar(bb1)) or (is_dot(bb1) and is_verticalbar(bb))) 
		and abs(x1 - x) < 10
	)  # 10 is a magical number

def is_equationmark(bb, bb1):
	(x, y), (xw, yh) = bb
	(x1, y1), (xw1, yh1) = bb1
	return is_horizontalbar(bb) and is_horizontalbar(bb1) and abs(x1 - x) < 20 and abs(xw1 - xw) < 20 # 20 is a migical number

# detect if input three boundingBoxes are a ellipsis (three dots)
def is_dots(bb, bb1, bb2):
	(x, y), (xw, yh) = bb
	(x1, y1), (xw1, yh1) = bb1
	(x2, y2), (xw2, yh2) = bb2
	cenY = y + (yh - y) / 2
	cenY1 = y1 + (yh1 - y1) / 2
	cenY2 = y2 + (yh2 - y2) / 2
	return (is_dot(bb) and is_dot(bb1) and is_dot(bb2) and max(cenY, cenY1, cenY2) - min(cenY, cenY1, cenY2) < 50)  # 30 is a migical number


# detect if input two boundingBoxes are a plus-minus
def is_pm(bb, bb1):
    (x, y), (xw, yh) = bb
    (x1, y1), (xw1, yh1) = bb1
    cenX = x + (xw - x) / 2
    cenX1 = x1 + (xw1 - x1) / 2
    case1 = is_horizontalbar(bb) and is_square(bb1) and x < cenX1 < xw and -15 < y - yh1 < 35 and xw - cenX1 < 50
    case2 = is_square(bb) and is_horizontalbar(bb1) and x1 < cenX < xw1 and -15 < y1 - yh < 35 and xw1 - cenX < 50
    return case1 or case2  # magical number

def rotate(img,angle):
	# rotate the image to deskew it
	(h, w) = img.shape[:2]
	center = (w // 2, h // 2)
	# print("rotating")
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	# rotated = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	rotated = imutils.rotate_bound(img,-angle)
	return rotated


def get_houghlines_angle(bw):
	lines2 = cv2.HoughLinesP(bw, 2, np.pi / 180, threshold=50, minLineLength=2, maxLineGap=150)
	angle = 0
	if lines2 is not None:
		for x1,y1,x2,y2 in lines2[0]:
			angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
			angle = int(angle)
			print(angle)
			# print(angle)
			# cv2.line(bw,(x1,y1),(x2,y2),(255,255,255),2)
	return angle


# detect if input three boundingBoxes are a fraction
def is_fraction(bb, bb1, bb2):
    (x, y), (xw, yh) = bb
    (x1, y1), (xw1, yh1) = bb1
    (x2, y2), (xw2, yh2) = bb2
    cenX = x + (xw - x) / 2
    cenX1 = x1 + (xw1 - x1) / 2
    cenX2 = x2 + (xw2 - x2) / 2
    case1 = not is_dot(bb) and not is_dot(bb1) and is_horizontalbar(bb2) and (y < y2 < yh1 or y1 < y2 < yh)
    case2 = not is_dot(bb2) and not is_dot(bb) and is_horizontalbar(bb1) and (y2 < y1 < yh or y < y1 < yh2)
    case3 = not is_dot(bb1) and not is_dot(bb2) and is_horizontalbar(bb) and (y1 < y < yh2 or y2 < y < yh1)
    return (case1 or case2 or case3) and  max(cenX, cenX1, cenX2) - min(cenX, cenX1, cenX2) < 50  # 30 is a migical number


# return initial bounding boxes of input image
def initial_boxes(im):
	'''input: bw image; return: None'''

	#threshold
	im[im >= 127] = 255
	im[im < 127] = 0

	'''
	# set the morphology kernel size, the number in tuple is the bold pixel size
	kernel = np.ones((2,2),np.uint8)
	im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
	'''

	thresh = im
	thresh,cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = np.array(cnts)
	avgCntArea = np.mean([cv2.contourArea(c) for c in cnts])
	# bounding rectangle outside the individual element in image
	bbs = []
	for cnt in cnts:
		x,y,w,h = cv2.boundingRect(cnt)
		# exclude the whole size image and noisy point
		if x is 0: continue
		# if w*h < 25: continue
		if cv2.contourArea(cnt) < avgCntArea/20:
			continue
		bbs.append([(x,y), (x+w, y+h)])
	return bbs



def connect(im, bbs):
	final_bbs = []
	i = 0
	bbs.sort()
	while i < (len(bbs)-1):
		(x,y),(xw,yh) = bbs[i]
		(x1,y1),(xw1,yh1) = bbs[i+1]
		equation = False#is_equationmark(bbs[i],bbs[i+1])
		letteri = False#is_letteri(bbs[i],bbs[i+1])
		pm = False#is_pm(bbs[i],bbs[i+1])
		divmark = False
		dots = False
		fraction = False
		if i < len(bbs) - 2:
			(x2, y2), (xw2, yh2) = bbs[i+2]
			divmark = is_divisionmark(bbs[i],bbs[i+1],bbs[i+2])
			dots = is_dots(bbs[i],bbs[i+1],bbs[i+2])
			fraction = is_fraction(bbs[i],bbs[i+1],bbs[i+2])
		if (equation or letteri or pm) and not fraction:
			final_bbs.append([(min(x, x1), min(y, y1)), (max(xw, xw1), max(yh, yh1))])
			i+=2
		elif (divmark or dots):# and not fraction:
			final_bbs.append([(min(x, x1, x2), min(y, y1, y2)), (max(xw, xw1, xw2), max(yh, yh1, yh2))])
			i+=3
		else:
			final_bbs.append(bbs[i])
			i+=1

	while i < len(bbs):
		final_bbs.append(bbs[i])
		i+=1

	return final_bbs

def getBestShift(img):
	cy,cx = ndimage.measurements.center_of_mass(img)

	rows,cols = img.shape
	shiftx = np.round(cols/2.0-cx).astype(int)
	shifty = np.round(rows/2.0-cy).astype(int)

	return shiftx,shifty

def shift(img,sx,sy):
	rows,cols = img.shape
	M = np.float32([[1,0,sx],[0,1,sy]])
	shifted = cv2.warpAffine(img,M,(cols,rows))
	return shifted  


#resize symbol
def resize(sym,IMG_ROW,IMG_COL):
	# Resize
	border_v = 0
	border_h = 0
	if (IMG_COL/IMG_ROW) >= (sym.shape[0]/sym.shape[1]):
	    border_v = int((((IMG_COL/IMG_ROW)*sym.shape[1])-sym.shape[0])/2)
	else:
	    border_h = int((((IMG_ROW/IMG_COL)*sym.shape[0])-sym.shape[1])/2)
	sym = cv2.copyMakeBorder(sym, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
	sym = cv2.resize(sym, (IMG_ROW, IMG_COL),interpolation=cv2.INTER_AREA)
	# sym = cv2.cvtColor(sym, cv2.COLOR_BGR2GRAY)
	_,sym = cv2.threshold(sym.copy(), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	sym = cv2.dilate(sym,np.ones((3,3)))
	sym = cv2.erode(sym,np.ones((2,2)))

	#padding
	gray = sym

	while np.sum(gray[0]) == 0:
		gray = gray[1:]

	while np.sum(gray[:,0]) == 0:
		gray = np.delete(gray,0,1)

	while np.sum(gray[-1]) == 0:
  		gray = gray[:-1]

	while np.sum(gray[:,-1]) == 0:
		gray = np.delete(gray,-1,1)

	rows,cols = gray.shape

	if rows > cols:
		factor = 20.0/rows
		rows = 20
		cols = int(round(cols*factor))
		gray = cv2.resize(gray, (cols,rows))
	else:
		factor = 20.0/cols
		cols = 20
		rows = int(round(rows*factor))
		gray = cv2.resize(gray, (cols, rows))
	
	colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
	rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
	gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
	shiftx,shifty = getBestShift(gray)
	shifted = shift(gray,shiftx,shifty)
	gray = shifted
	return gray






def save(im,bbs,name):
	bbs = sorted(bbs,key=lambda box: (box[1][1]-box[0][1]) * (box[1][0]-box[0][0]))
	for i,bb in enumerate(bbs):
		(x,y),(xw,yh) = bb
		x-=1
		y-=1
		xw-=1
		yh+=1
		sym = im[y:yh,x:xw]
		symval,newimg = imageprepare(sym)
		symval = np.array(symval)
		cv2.imshow('img1',symval.reshape(28,28))
		# cv2.waitKey(0)
		# sym = cv2.resize(sym,(28,28),interpolation=cv2.INTER_NEAREST)
		
		sym = resize(sym,32,32)
		cv2.imshow('img2',sym)
		cv2.waitKey(0)
		if is_dot(bb):
			cv2.imwrite(f'{name}_dot_{y}_{yh}_{x}_{xw}.png',sym)
		else:
			cv2.imwrite(f'{name}_{y}_{yh}_{x}_{xw}.png',sym)
		cv2.rectangle(im,(x,y),(xw,yh),(0,0,0),-1)


def get_symlist(thresh):
	bbs = initial_boxes(thresh)
	#connected bounding boxes
	# bbs = connect(thresh,bbs)
	#sorted bounding boxes
	bbs = sorted(bbs, key=lambda box: (box[1][1]-box[0][1]) * (box[1][0]-box[0][0]))
	symlist = []
	# stacked = np.zeros([28,28],dtype=np.uint8)

	for bb in bbs:
		(x,y),(xw,yh) = bb
		x-=1
		y-=1
		xw +=1
		yh +=1
		w = xw - x - 2
		h = yh - y - 2
		sym = thresh[y:yh,x:xw]
		try:
			sym = resize(sym,32,32)
			# print(sym.shape)
			# stacked = np.concatenate((stacked, sym), axis=1)
			if w < 25 and h < 25 and float(w)/h < 1.5 and float(h)/w < 1.5 :
				sym = (sym,"dot",x,y,xw,yh)
				# cv2.imwrite(f'{name}_dot_{y}_{yh}_{x}_{xw}.png',sym)
			else:
				sym = (sym,"sym",x,y,xw,yh)
			symlist.append(sym)
		except:
			pass
			#print("error")
		cv2.rectangle(thresh,(x,y),(xw,yh),(0,0,0),-1)
		# cv2.imshow("thresh",thresh)
		# cv2.waitKey(0)
	# cv2.imshow('sym',stacked)
	return bbs,symlist



	
def main():
	im = cv2.imread('data/test.png')
	im2 = im.copy()
	imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
	bbs = initial_boxes(thresh)
	final_bbs = connect(thresh,bbs)
	for bb in final_bbs:
		(x,y),(xw,yh) = bb
		cv2.rectangle(im,(x,y),(xw,yh),(0,255,0),2)
	cv2.imwrite('data/tt2.png',im)
	save(im2,final_bbs,"data/annotated/sym")

if __name__ == '__main__':
	# main()
	im = cv2.imread('data/test.png')
	imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
	bbs,sl = get_symlist(thresh)
	for img,txt,x,y,xw,yh in sl:
		cv2.imshow('img',img)
		cv2.waitKey(0)

