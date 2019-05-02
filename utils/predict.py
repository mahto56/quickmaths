from quickmaths.utils import bbutils
from quickmaths.utils import scanutils as sc
from quickmaths.utils.latex2sympy.process_latex import process_sympy
import operator 
import numpy as np
import cv2
from sympy import *
import matplotlib.pyplot as plt
import re



symbols = ['0','1','2','3','4','5','6','7','8','9','+','x','-','(',')','sqrt']

class Predictor:
	def __init__(self,model):
		self.model = model
	
	def predict(self,im,symlist,main_img,x1,y1):
		# sorted_symlist = sorted(symlist,key=operator.itemgetter(2,3))
		syms = []
		boxes = []
		names = []
		im2 = im.copy()
		stacked = np.zeros([28,28],dtype=np.uint8)
		alt_symlist = []
		for sym,name,x,y,xw,yh in symlist:
			stacked = np.concatenate((stacked, sym), axis=1)
			syms.append(sym)
			boxes.append((x,y,xw,yh))
			names.append(name)

		syms  = np.array(syms)
		try:
			labels = self.model.predict_classes(syms.reshape(-1,28,28,1))
			# print(labels)
			for (x,y,xw,yh),label,name,sym in sorted(zip(boxes,labels,names,syms)):
				# cv2.rectangle(main_img,(x,y),(xw,yh),(0,0,255),1)
				# print(name)
				if name != "dot":
					cv2.putText(main_img,str(symbols[int(label)]),(x1+x+2,y1+y-5),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
					alt_symlist.append((sym,str(symbols[int(label)]),x,y,xw,yh))
				else:
					cv2.putText(im,".",(x+2,y-5),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),8)
					alt_symlist.append((sym,"dot",x,y,xw,yh))
		
			# cv2.imwrite(f'im_{len(labels)}.png',im)
			updated_symlist =  sc.update(im2,alt_symlist)
			
			latek = sc.toLatex(updated_symlist)
			latek = latek.replace("dot", ".")
			# latek = latek.replace("sq","sqrt")
			# print(latek)
			pprint(process_sympy(latek), use_unicode=False) 
			processed_latek = process_sympy(latek)
			# latek = latek.replace("sq","sqrt")
			res = sympify(processed_latek)
			try:
				res = res.evalf()
			except:
				res = res
			# preview(r'$$%s$$'%latek, viewer='file', filename='latek.png', euler=False)
			return res,stacked,updated_symlist

		except Exception as e:
			print(e)
			return None,stacked,updated_symlist





# predict(sl)
# cv2.imshow('im',im)
# cv2.waitKey(0)