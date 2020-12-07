import os
import cv2
import numpy as np
import pickle
import sys

from chainer.datasets import tuple_dataset
import glob

# BGR -> HSV
def BGR2V(_img):
    # get max in RGB
    max_v = np.max(_img, axis=2).copy()
    max_v = max_v - 140
    max_v[max_v < 0] = 0
    max_v[max_v > 40] = 40
    max_v = max_v / 40* 255    

    return max_v

	# return hsv
#endRGB2V

def defMastadata(original_path,filename):
    target_path = os.path.join(original_path, filename)
    
    _img = cv2.imread(target_path)  # 画像読み込み.dtype=uint8

    # RGB > HSV
    img_v = BGR2V(_img)
    print(np.max(img_v),np.min(img_v))
    ans = filename.split('.')[0].split('-')[1]
    if ans=='o':
        ansSymbol=0
    else:
        ansSymbol=1

    return img_v,ansSymbol

#enddef

def main():
    bace = os.path.dirname(os.path.abspath(__file__)) # 実行ファイルのディレクトリ名
    path = "img/input/Dust"
    doOne = True
    original_path = os.path.join(bace, path)
    files = os.listdir(original_path)    # <class 'list'> ['1-1.PNG', '1-2.PNG'...]
    imgLen=len(files)
    imgMastaData=np.empty((imgLen,480,720),dtype=np.uint8) #480,720 np.concatenate([a,b],axis=0)
    ansMastaData=np.empty(imgLen,dtype=np.int32)
    #とりあえず突っ込む
    for i,filename in enumerate(files):
        imgMastaData[i],ansMastaData[i] = defMastadata(original_path,filename)
    an = imgMastaData[0].copy().reshape(-1)
    print(an.shape)
    MastaData = [(an,ansMastaData[0])]

    for i in range(imgLen):
        img,ans = imgMastaData[i].copy(),ansMastaData[i].copy()
        # temAns = [img.copy().reshape(-1),ans]
        MastaData.append((img.copy().reshape(-1).astype(np.float32),ans))

        for j in [0,1,(0,1)]:
            createdImg = np.flip(img,j).reshape(-1).astype(np.float32).copy()
            # temAns = [createdImg,ans]
            MastaData.append((createdImg,ans))
            print(j,end='')
            del createdImg
        print(i)
    print(MastaData.pop(0))
    for k in MastaData:
        print(k,np.max(k[0]),np.min(k[0]))
    
    with open('data/realdustData.pickle', mode='wb') as fo1:
            pickle.dump(MastaData, fo1)
    
#end main

if __name__=='__main__':
    main()
    print('finish!!!')
#endif