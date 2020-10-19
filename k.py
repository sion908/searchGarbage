import numpy as np
import cv2
colorparetto = np.array([[[244,0,0],[244,244,0],[244,0,244]],[[0,244,0],[0,244,244],[0,0,244]],[[122,0,0],[0,122,0],[0,0,122]]])


cv2.imwrite('a.png',colorparetto)