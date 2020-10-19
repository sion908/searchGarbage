import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

# BGR -> HSV
def BGR2HSV(_img):
	img = _img.copy() / 255.

	hsv = np.zeros_like(img, dtype=np.float32)

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()
	min_arg = np.argmin(img, axis=2)

	# H
	hsv[..., 0][np.where(max_v == min_v)]= 0
	## if min == B
	ind = np.where(min_arg == 0)
	hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
	## if min == R
	ind = np.where(min_arg == 2)
	hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
	## if min == G
	ind = np.where(min_arg == 1)
	hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300
		
	# S
	hsv[..., 1] = max_v.copy() - min_v.copy()

	# V
	hsv[..., 2] = max_v.copy()
	
	return hsv

def imgchange(img,perf):
	max = np.max(img)
	min = np.min(img)

	img = (img - min) * 255 / (max - min)

	img = img // perf

	img = img *perf
	#EasyNoiseCut
	H,W = img.shape
	k=[[0,1,0],[1,0,1],[0,1,0]]
	for h in range(H-2):
		for w in range(W-2):
			
			if img[h,w+1]*3 == np.sum([[0,1,0],[0,0,1],[0,1,0]]*img[h:h+3,w:w+3]):
				img[h+1,w+1] = img[h,w+1]
			elif img[h,w+1]*3 == np.sum([[0,1,0],[1,0,0],[0,1,0]]*img[h:h+3,w:w+3]):
				img[h+1,w+1] = img[h,w+1]
			elif img[h,w+1]*3 == np.sum([[0,1,0],[1,0,1],[0,0,0]]*img[h:h+3,w:w+3]):
				img[h+1,w+1] = img[h,w+1]
			elif img[h+1,w]*3 == np.sum([[0,0,0],[1,0,1],[0,1,0]]*img[h:h+3,w:w+3]):
				img[h+1,w+1] = img[h+1,w]


	return img

def SeekGravity(img,perf):
	import math
	H,W = img.shape
	pernum = 255 // perf + 1
	SGline = np.empty((2,pernum))
	# print(H,W)
	img_make = np.zeros((H,W,3),dtype=int)
	colorparetto = np.array([[244,0,0],[244,244,0],[244,0,244],[0,244,0],[0,244,244],[0,0,244],[122,0,0],[0,122,0],[0,0,122]])
	for i in range(pernum):

		a = np.where(img==i*perf)

		len = a[0].shape
		SGh = np.sum( a[0] ) / len
		SGw = np.sum( a[1] ) / len
		img_make[a] = colorparetto[i]
		SGline[:,i] = [SGh,SGw]
	# print(SGline)
	SGline[0] -= H/2
	SGline[1] -= W/2

	ans = np.empty(pernum)
	for i in range(pernum):
		ans[i] = math.sqrt(SGline[0,i] ** 2 + SGline[1,i] ** 2)
	# print(SGline)
	return img_make,ans

def main(num):
	img_orig = cv2.imread('img/input/Dust-' + num + '.jpg')
	# RGB > HSV
	img_hsv = BGR2HSV(img_orig)
	img_v = img_hsv[...,2].copy()

	# cv2.imwrite(path,name) 0-255
	# cv2.imwrite('img/output/0.png',img_orig)

	for i in [30]: #1,20,30,50
		img_i = imgchange(img_v,i)
		cv2.imwrite('img/output/' + str(i) + '-' + num + '.png',img_i)
		img_c,gplace = SeekGravity(img_i,i)
		cv2.imwrite('img/output/' + str(i) + '-' + num + '-G.png',img_c)
		print(num,end='')
		print(gplace[-5:])
	# print('end')

if __name__ == '__main__':
	main('b-0')
	main('b-1')
	main('b-2')
	main('b-3')
	main('o-0')
	main('o-1')