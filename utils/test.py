import bbutils2
import cv2
import numpy as np
import imutils


#get bounding boxes
def getBBs(thresh):

	(thresh,cnts,_) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])

	digits = []
	boxes = []
	masks = []



	expr = ""

	for (i,c) in enumerate(cnts):
		if cv2.contourArea(c)<avgCntArea/10:
		    continue
		(x,y,w,h) = cv2.boundingRect(c)


		mask = np.zeros(thresh.shape,dtype="uint8")
		hull = cv2.convexHull(c)

		cv2.drawContours(mask,[c],-1,255,-1)
		# cv2.imshow('mask_before',mask)
		# cv2.waitKey(0)
		mask = cv2.bitwise_and(thresh,thresh,mask=mask)
		# cv2.imshow('mask_after',mask)
		# cv2.waitKey(0)
		
		cv2.rectangle(thresh,(x,y),(x+w,y+h),(255,255,255),1)


		digit = mask[y-8:y+h+8,x-8:x+w+8]
		try:
			# cv2.imshow(f'{i}',digit)
			# cv2.waitKey(0)
			# digit = bbutils2.resize(digit,32,32)
			expr += "C"
			# cv2.imshow(f'resized',digit)
			# cv2.waitKey(0)
			# digit = deskew(digit)
			# cv2.imshow(f'skewed{i}',digit)
			# cv2.waitKey(0)
			boxes.append((x,y,w,h))
			digits.append(digit)
			# digit = 255-digit
			masks.append(mask)
		except Exception as e:
		    print(e)

	# boxes,digits = sorted(zip(boxes,digits))
	# boxes.sort()
	# digits.sort()
	sorted_boxes = []
	sorted_digits = []
	for (x,y,w,h),d in sorted(zip(boxes,digits)):
		# cv2.imshow('d',d)
		# cv2.waitKey(0)
		sorted_boxes.append((x,y,w,h))
		sorted_digits.append(d)

	# for (x,y,w,h),d in zip(sorted_boxes,sorted_digits):
	# 	cv2.imshow('d',d)
	# 	cv2.waitKey(0)
		
	return np.array(sorted_boxes),np.array(sorted_digits),thresh.copy()

# detect if input boundingBox contains a dot
def is_dot(bb):
	# (x, y), (xw, yh) = bb
	x,y,w,h = bb
	area = w*h
	return area < 200 and 0.5 < w/h < 2 and w < 20 and h < 20  # 100 is migical number


# detect if input three boundingBoxes are a fraction
def is_fraction(bb, bb1, bb2):
    x, y,w,h = bb
    x1,y1,w1,h1 = bb1
    x2,y2,w2,h2 = bb2
    cenX = x + w / 2
    cenX1 = x1 + w1 / 2
    cenX2 = x2 + w2 / 2
    case1 = not is_dot(bb) and not is_dot(bb1) and is_horizontalbar(bb2) and (y < y2 < y1+h1 or y1 < y2 < y+h)
    case2 = not is_dot(bb2) and not is_dot(bb) and is_horizontalbar(bb1) and (y2 < y1 < y+h or y < y1 < y2+h2)
    case3 = not is_dot(bb1) and not is_dot(bb2) and is_horizontalbar(bb) and (y1 < y < y2+h2 or y2 < y < y1+h1)
    return (case1 or case2 or case3) and  max(cenX, cenX1, cenX2) - min(cenX, cenX1, cenX2) < 50  # 30 is a migical number



# detect if input three boundingBoxes are a division mark
def is_divisionmark(bb, bb1, bb2):
    x,y,w,h = bb
    x1,y1,w1,h1 = bb1
    x2,y2,w2,h2 = bb2
    cenY1 = y1 + h1 / 2
    cenY2 = y2 + h2 / 2
    return (
    	is_horizontalbar(bb) and is_dot(bb1) and is_dot(bb2)
    	and x < x1 < x2 < x+w and max(y1, y2) > y and min(y1, y2) < y
    	and max(y1, y2) - min(y1, y2) < 1.2 * abs(w)
    )

# detect if a given boundingBox contains a horizontal bar
def is_horizontalbar(bb):
    x,y,w,h = bb
    return w / h  > 3.5 #magic number


def is_equationmark(bb, bb1):
	x, y,w,h = bb
	xw=x+w
	yh=y+h

	x1, y1, w1, h1 = bb1
	xw1=x1+w1
	yh1=y1+h1


	return is_horizontalbar(bb) and is_horizontalbar(bb1) and abs(x1 - x) < 20 and abs(xw1 - xw) < 20 # 20 is a migical number



t2 = None
t3 = None

def combine_eq(box1,box2,sym1,sym2):
	x,y,w,h = box1
	x1,y1,w1,h1 = box2
	return (min(x, x1)-8, min(y, y1)-8, max(w,w1)+8,h+h1+abs(y-y1)+8)
	
	
def combine_div(box1,box2,box3,sym1,sym2,sym3):
	x,y,w,h = box1
	x1,y1,w1,h1 = box2
	x2,y2,w2,h2 = box3
	return (min(x, x1, x2), min(y, y1, y2),max(xw, xw1, xw2), h+h1+h2+abs(y-y1)+abs(y1-y2))

def connectBBs(boxes,digits):
	i=0
	while i < len(boxes)-1:
		
		x,y,w,h = boxes[i]
		x1,y1,w1,h1 = boxes[i+1]

		eqmark = is_equationmark(boxes[i],boxes[i+1])
		divmark = False
		fraction = False

		if i < len(boxes) - 2:
			x2,y2,w2,h2 = boxes[i+2]
			divmark = is_divisionmark(boxes[i],boxes[i+1],boxes[i+2])
			fraction = is_fraction(boxes[i],boxes[i+1],boxes[i+2])
		if fraction:
			cv2.imshow('num',digits[i+1])
			cv2.imshow('bar',digits[i])
			cv2.imshow('den',digits[i+2])
			cv2.waitKey(0)

		if eqmark and not fraction:
			box = combine_eq(boxes[i],boxes[i+1],digits[i],digits[i+1])
			x,y,w,h = box
			cv2.rectangle(t2,(x,y),(x+w,y+h),(255,255,255),1)
			digits = np.delete(digits,[i+1],axis=0)
			boxes = np.delete(boxes,[i+1],axis=0)	
			digits[i] = t3[y:y+h,x:x+w].copy()
			boxes[i] = box

		elif divmark and not fraction:
			box = combine_div(boxes[i],boxes[i+1],boxes[i+2],digits[i],digits[i+1],digits[i+2])
			x,y,w,h = box
			cv2.rectangle(t2,(x,y),(x+w,y+h),(255,255,255),1)
			digits = np.delete(digits,[i+1,i+2],axis=0)
			boxes = np.delete(boxes,[i+1,i+2],axis=0)	
			digits[i] = t3[y:y+h,x:x+w].copy()
			boxes[i] = box


			
		print(f'ishor: {is_horizontalbar(boxes[i])}, ratio: {w/h}, w: {w}, h: {h}')
		print(f'ishor: {is_horizontalbar(boxes[i+1])}, ratio: {w1/h1}, w1: {w1}, h1: {h1} ')
		
		cv2.imshow('im1',digits[i])
		cv2.imshow('im2',digits[i+1])
		cv2.waitKey(0)
		
		i+=1
		# cv2.waitKey(0)
	for i,d in enumerate(digits):
		digits[i] = bbutils2.resize(d,32,32)
		x,y,w,h = boxes[i]
		cv2.rectangle(t2,(x,y),(x+w,y+h),(255,255,255),1)
		cv2.imshow('dd',digits[i])
		cv2.waitKey(0)
		#fix the bounding box problem!!!!!
	# cv2.imshow('t2',t2)
	cv2.waitKey(0)
	return boxes,digits
	




if __name__=='__main__':
	im = cv2.imread('/home/kayshu/OpenCVProjects/quickmaths/data/test2.png')
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	thresh = gray.copy()
	_,thresh = cv2.threshold(thresh.copy(),127,255,cv2.THRESH_BINARY)
	# thresh = cv2.erode(thresh,np.ones((5,5)))
	# thresh = cv2.GaussianBlur(thresh,(1,1),cv2.BORDER_DEFAULT)
	# thresh = im
	t2 = thresh.copy()
	t3 = thresh.copy()
	boxes,digits,th = getBBs(thresh)
	
	boxes,digits = connectBBs(boxes,digits)
	cv2.imshow('im',t2)
	cv2.waitKey(0)
	
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
