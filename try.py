import numpy as np
import cv2
import sys
import math

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

def QuantizationImg(img,perf):
	max = 0.7
	min = 0.5
	print(max,min)
	img = (img - min) * 255 / (max - min)

	cv2.imwrite('img/output/ghjt.png',img)
	img = img // perf

	img = img *perf

	return img

def NoiseCut(img,perf):
	#EasyNoiseCut
	H,W = img.shape
	for h in range(H-2):
		for w in range(W-2):
			img[h+1,w+1] = np.median(img[h:h+3,w:w+3])
	return img
	#enddef NoiseCut

def SeekGravity(img,perf):
	
	H,W = img.shape
	pernum = 255 // perf + 1
	# print(pernum)
	SGline = np.zeros((2,pernum),dtype=int)
	# print(H,W)
	img_make = np.zeros((H,W,3),dtype=int)
	colorparetto = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[255,127,0],[0,255,127],[127,0,255]])
	imag = np.zeros((H,W),dtype=int)
	for i in range(pernum):

		a = np.where(img==i*perf)
		imag[a] = 255
		img_Noise = NoiseCut(imag,perf)
		# cv2.imwrite('img/output/color-' + str(i) + '.png',img_Noise)
		gravLine_img = np.where(img_Noise == 255)
		imag = np.zeros((H,W),dtype=int)
		len = gravLine_img[0].shape
		if len[0] != 0:
			SGh = int(np.sum( gravLine_img[0] ) / len)
			SGw = int(np.sum( gravLine_img[1] ) / len)
			SGline[:,i] = [SGh,SGw]
		else:
			SGline[:,i] = [0,0]
		img_make[gravLine_img] = colorparetto[i]
	# # print(SGline)
	# SGline[0] -= int(H/2)
	# SGline[1] -= int(W/2)
	# # print(SGline)
	ans = np.empty(pernum)
	img_serchcent = np.zeros((H,W,3),dtype=int)
	for i in range(pernum):
		img_serchcent[SGline[0,i]-2:SGline[0,i]+3,SGline[1,i]-2:SGline[1,i]+3]=colorparetto[i]
		ans[i] = math.sqrt((SGline[0,i]-SGline[0,pernum-2]) ** 2 + (SGline[1,i]-SGline[1,pernum-2]) ** 2)
	# cv2.imwrite('img/output/Cenyt-' + str(perf) + '.png',img_serchcent)
	# printScale
	# img_make[H//2-1:H//2+2,W//2-100:W//2+2] = [0,0,0]
	# img_make[H//2-6:H//2-1,W//2-1:W//2+2] = [0,0,0]
	# img_make[H//2-6:H//2-1,W//2-100:W//2-97] = [0,0,0]
	return img_make,img_serchcent,ans

def main(num):
	img_orig = cv2.imread('img/input/Dust/Dust-' + num + '.jpg')
	# RGB > HSV
	img_hsv = BGR2HSV(img_orig)
	img_v = img_hsv[...,2].copy()

	# cv2.imwrite(path,name) 0-255

	for i in [30]: #1,20,30,50
		img_i = QuantizationImg(img_v,i)
		# img_i = NoiseCut(img_i,i)
		# cv2.imwrite('img/output/' + str(i) + '-' + num + '.png',img_i)
		img_c,img_cent,gplace = SeekGravity(img_i,i)
		H,W,_ = img_c.shape
		img_ans = np.zeros((H*2,W*2,3),dtype=int)
		img_ans[ :H, :W,0] = img_i
		img_ans[ :H,W: ] = img_c
		img_ans[H: , :W] = img_cent

		cv2.imwrite('img/output/' + str(i) + '-' + num + '.png',img_ans)
		# print(num,end='')
		# print(gplace[-6:-1]) 
	# print('end')

if __name__ == '__main__':
	main('b-0')
	main('b-1')
	main('b-2')
	main('b-3')
	main('o-0')
	main('o-1')
	main('s-0')
	main('s-1')
	main('s-2')