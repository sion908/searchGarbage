import os
import cv2
import numpy as np
import pickle
import sys

from chainer.datasets import tuple_dataset
import glob

def oneTransImg(img,imgNum,p_original,mastaData):
    times = 43
    for i in range(4):
        direction = [0,0,0,0]
        direction[i] = 1

        transImg(img,imgNum,p_original,mastaData,times,(0,0,0,0))
        print(' dir{}'.format(i))
    #endfor
#enddef
def bothTransImg(img,imgNum,p_original,mastaData):
    times = 11
    transImg(img,imgNum,p_original,mastaData,times,(1,1,0,0))
    transImg(img,imgNum,p_original,mastaData,times,(0,0,1,1))
    transImg(img,imgNum,p_original,mastaData,times,(1,1,1,1))
    print('both')
#enddef

def transImg(img,imgNum,p_original,mastaData,times,dir):
    # dir = [up,down,right,left]
    for i in range(times):
        lenDiv = i
        #変更後の座標
        p_trans    = np.float32([[dir[3]*lenDiv,dir[0]*lenDiv], [dir[3]*lenDiv, 50-dir[1]*lenDiv],  [50-dir[2]*lenDiv,dir[0]*lenDiv], [50-dir[2]*lenDiv,50-dir[1]*lenDiv]])  #変更後の座標
        # 変換マトリクスと射影変換
        M = cv2.getPerspectiveTransform(p_original, p_trans)

        i_trans = cv2.warpPerspective(img, M, (50, 50))
        # path_out = os.path.join(bace, 'testImg/') + imgNum +'-0.png' 
        # cv2.imwrite(path_out, i_trans)
        imgList1 = []
        for widthLine in i_trans:
            for num  in widthLine:
                imgList1.append(num/255)
        #endfor
        imgmt = np.array(imgList1,dtype=np.float32)
        mastaData.append((imgmt,imgNum))

        print(times,end='')
#enddef

def defMastadata(original_path,filename,mastaData):
    target_path = os.path.join(original_path, filename) # パスの連結
    # PDF/testImg/1-1.PNG
    img = cv2.imread(target_path, 0)  # 画像読み込み

    imgshape   = img.shape
    p_original = np.float32([[0, 0], [0, imgshape[0]], [imgshape[1], 0], [imgshape[1], imgshape[0]]])  #変更前の座標
    imgNum     = int(filename.split('.')[0].split('-')[1])
    p_trans    = np.float32([[0,0], [0, 50],  [50,0], [50,50]])  #変更後の座標
    # 変換マトリクスと射影変換
    M = cv2.getPerspectiveTransform(p_original, p_trans)

    i_trans = cv2.warpPerspective(img, M, (50, 50))
    imgList1 = []
    for widthLine in i_trans:
        for num  in widthLine:
            imgList1.append(num/255)
    #endfor
    imgmt = np.array(imgList1,dtype=np.float32)
    mastaData=[(imgmt,imgNum)]
    
#enddef

if __name__=='__main__':
    bace = os.path.dirname(os.path.abspath(__file__)) # 実行ファイルのディレクトリ名
    path = "data/img/org_num"
    doOne = True
    original_path = os.path.join(bace, path)
    files = os.listdir(original_path)    # <class 'list'> ['1-1.PNG', '1-2.PNG'...]
    print(files)
    mastaData=[]
    defMastadata(original_path,files[0],mastaData)
    sys.exit()
    for filename in files:
        target_path = os.path.join(original_path, filename) # パスの連結
        # PDF/testImg/1-1.PNG
        img = cv2.imread(target_path, 0)  # 画像読み込み

        imgshape = img.shape
        p_original = np.float32([[0, 0], [0, imgshape[0]], [imgshape[1], 0], [imgshape[1], imgshape[0]]])  #変更前の座標
        imgNum=int(filename.split('.')[0].split('-')[1])
            
        oneTransImg(img,imgNum,p_original,mastaData)
        bothTransImg(img,imgNum,p_original,mastaData)
        
        print(filename)
    # train_data = []
    # train_label = []
    # data_raw = open("data.txt")
    # for line in data_raw:
    #     train = np.array([np.float32(int(x)/255.0) for x in line.split(",")[0:input_num]])
    #     label = np.int32(line.split(",")[input_num])
    #     train_data.append(train)
    #     train_label.append(label)

    # threshold = np.int32(len(imageData)/10*9)
    # # train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
    test  = tuple_dataset.TupleDataset(mastaData)
    print(len(mastaData))
    with open('./RoboMasData.pickle', mode='wb') as fo1:
            pickle.dump(mastaData, fo1)
    print('finish')
#endif
            