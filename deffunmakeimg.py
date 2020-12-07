import numpy as np
import cv2
import itertools
import random

def makelist(a=[2,1,0]):
    x=True
    y=False
    for c in itertools.combinations_with_replacement(a,5):
        if sum(c)==5:
            # c = [[2,2,1,0,0],[2,1,1,1,0],[1,1,1,1,1]]
            for d in itertools.permutations(c):
                if x:
                    b=[d]
                    x=False
                for e in b:
                    if e==d:
                        y=True
                        break
                if y:
                    y=False
                    continue
                b.append(d)
    return b

def defCirZone(insideSquare = [252 ,468 ,154 ,326]):
    S=[insideSquare[1]-insideSquare[0],insideSquare[3]-insideSquare[2]]
    coodLen = 380*620-S[0]*S[1]
    GargCood = np.empty((2,coodLen))
    lastnum=0
    for i,a in enumerate(range(50,insideSquare[2])):
        GargCood[0,i*620:(i+1)*620] = int(a)
    lastnum = (i+1)*620
    
    for i,a in enumerate(range(insideSquare[2],insideSquare[3])):
        GargCood[0,lastnum+i*(620-S[0]):lastnum+(i+1)*(620-S[0])] = int(a)
    lastnum = lastnum+(i+1)*(620-S[0])
    
    for i,a in enumerate(range(insideSquare[3],430)):
        GargCood[0,lastnum+i*620:lastnum+(i+1)*620] = int(a)
    lastnum = lastnum+(i+1)*620

    for i,a in enumerate(range(50,insideSquare[2])):
        GargCood[1,i*620:(i+1)*620] = np.arange(50,670,dtype=np.int32)
    lastnum = (i+1)*620
    for i,a in enumerate(range(insideSquare[2],insideSquare[3])):
        GargCood[1,lastnum+i*(620-S[0]):lastnum+(i+1)*(620-S[0])] = np.append(np.arange(50,insideSquare[0],dtype=np.int32),np.arange(insideSquare[1],670,dtype=np.int32))
    lastnum = lastnum+(i+1)*(620-S[0])
    for i,a in enumerate(range(insideSquare[3],430)):
        GargCood[1,lastnum+i*620:lastnum+(i+1)*620] = np.arange(50,670,dtype=np.int32)
    lastnum = lastnum+(i+1)*620

    GargCood =GargCood.astype(np.int32)

    ans_GargCood = np.empty((2,300),dtype=np.int32)
    count=0
    for i in range(coodLen):
        if random.random() < 0.0001:
            ans_GargCood[0][count]=GargCood[0][i]
            ans_GargCood[1][count]=GargCood[1][i]
            if count==299:
                break
            count += 1
    # print(i)


    return ans_GargCood

def mNoiseImg(noise=0.001,nSize=5,noiseColor=30):
    arr = np.random.rand(480*720).reshape((480,720))
    arr_ans = np.zeros((480,720))
    
    undercood = np.where(arr<noise)
    upcood = np.where(arr>1-noise)

    for i in range(len(undercood[0])):
        y,x=undercood[0][i],undercood[1][i]
        arr_ans[y:y+nSize,x:x+nSize] = 1
    for i in range(len(upcood[0])):
        y,x=upcood[0][i],upcood[1][i]
        arr_ans[y:y+nSize,x:x+nSize] = -1

    arr_ans *=  noiseColor

    return(arr_ans)

def mRouNoiseImg(color=30,xsize=360,ysize=240):
    img_ans = np.zeros((480,720))
    img_ans = cv2.ellipse(img_ans, (0,240),(xsize,ysize), angle=0,
                    startAngle=0, endAngle=360, color=30,
                    thickness=-1, lineType=8, shift=0)
    return img_ans


if __name__=="__main__":
    print(defCirZone().shape)