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
    return max_v

	# return hsv
#endRGB2V

def defMastadata(original_path,filename):
    target_path = os.path.join(original_path, filename)
    
    _img = cv2.imread(target_path)  # 画像読み込み.dtype=uint8

    # RGB > HSV
    img_v = BGR2V(_img)
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
    imgMastaData=np.empty((imgLen,480,720),dtype=np.float32) #480,720 np.concatenate([a,b],axis=0)
    ansMastaData=np.empty(imgLen,dtype=np.int32)
    #とりあえず突っ込む
    for i,filename in enumerate(files):
        imgMastaData[i],ansMastaData[i] = defMastadata(original_path,filename)
    
    MastaData = [imgMastaData[0].copy().reshape(-1),ansMastaData[0]]

    for i in range(imgLen):
        img,ans = imgMastaData[i].copy(),ansMastaData[i].copy()
        # temAns = [img.copy().reshape(-1),ans]
        MastaData.append((img.copy().reshape(-1),ans))

        for j in [0,1,(0,1)]:
            createdImg = np.flip(img,j).reshape(-1).copy()
            # temAns = [createdImg,ans]
            MastaData.append((createdImg,ans))
            print(j,end='')
            del createdImg
        print(i)
    
    

    print(len(MastaData))
    

    with open('data/dustData.pickle', mode='wb') as fo1:
            pickle.dump(MastaData, fo1)
    
#end main

if __name__=='__main__':
    main()
    print('finish!!!')
#endif

