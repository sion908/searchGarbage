import numpy as np
import cv2
import sys
import itertools
from deffunmakeimg import makelist,defCirZone,mNoiseImg,mRouNoiseImg
import pickle


    

def main():
    img = np.empty((480,720))
    # img = (Mat& img, Point center, Size axes, double angle,
    #        double startAngle, double endAngle, const Scalar& color,
    #        int thickness=1, int lineType=8, int shift=0)
    splitRaite = np.array([0,4,1,1,1,1,1,1,2])
    insideSquare = np.empty(4,dtype=np.int32)

    MastaData=[(0)]

    for j,list in enumerate(makelist()):
    # for j,list in enumerate([[1,1,1,1,1]]):
        circleRaite = np.append(np.array([0,0,0,0]),np.array(list))+splitRaite
        circleSize = np.sum(circleRaite)
        print(list)

        #メインの楕円をつける
        for i in range(8):
            circleSize -= circleRaite[i]
            cSize = (circleSize*36,int(circleSize*24*1.2))
            insideSquare[:] = [360-cSize[0],360+cSize[0],240-cSize[1],240+cSize[1]]
            img = cv2.ellipse(img, (360,240),cSize, angle=0,
                startAngle=0, endAngle=360, color=i*30,
                thickness=-1, lineType=8, shift=0)
            print(i,end='')
        print('')
        
        an = img.copy().reshape(-1).astype(np.float32)
        MastaData.append((an,0))
        print(MastaData)
        cv2.imwrite('img/make/0.png',an.reshape(480,720))
        del an

        for noisesize in [3,5]:
        # for noisesize in [3]:
            ranNoise_img = mNoiseImg(0.001,noisesize)#(noise=0.001,nSize=5,noiseColor=30)
            img_Snoise =   img.copy() - ranNoise_img
            an = img_Snoise.copy().reshape(-1).astype(np.float32)
            MastaData.append((an,0))
            cv2.imwrite('img/make/1.png',an.reshape(480,720))
            print('noisera ', end='')
            del an

            GargCoods = defCirZone(insideSquare) #ゴミの座標を決める
            for k in range(len(GargCoods[0])):
            # for k in range(50,51):
                #ゴミの付与
                img_dust = np.zeros((480,720))
                img_dust = cv2.ellipse(img_dust, (GargCoods[0][k],GargCoods[1][k]),(50,50), angle=0,
                        startAngle=0, endAngle=360, color=30,
                        thickness=-1, lineType=8, shift=0)
                am = img_Snoise - img_dust
                an = am.copy().reshape(-1).astype(np.float32)
                MastaData.append((an,1))
                cv2.imwrite('img/make/1-1.png',an.reshape(480,720))
                print(' {} '.format(k),end='')
                del an
            
            for RouNoiseColor in [30,60]:
            # for RouNoiseColor in [90]:
                img_dustn = cv2.ellipse(img_Snoise.copy(), (GargCoods[0][k],GargCoods[1][k]),(50,50), angle=0,
                        startAngle=0, endAngle=360, color=RouNoiseColor,
                        thickness=-1, lineType=8, shift=0)
                an = am.copy().reshape(-1).astype(np.float32)
                MastaData.append((an,1))
                cv2.imwrite('img/make/1-2.png',an.reshape(480,720))
                print(RouNoiseColor)
                del an
            print('')

        # for noisesize in [3,5]:
        for noisesize in [3]:
            ranNoise_img = mNoiseImg(0.001,noisesize)#(noise=0.001,nSize=5,noiseColor=30)
            ranNoisen_img = img.copy()-ranNoise_img
            an = ranNoisen_img.copy().reshape(-1).astype(np.float32)
            MastaData.append((an,0))
            cv2.imwrite('img/make/2.png',an.reshape(480,720))
            print('noisera ', end='')
            del an

            for noisecolor in [30,60]:
            # for noisecolor in [30]:
                RouNoise_img = mRouNoiseImg(noisecolor,insideSquare[1])#(color=30,xsize=360,ysize=240)
                cv2.imwrite('img/make/33.png',RouNoise_img)
                print(np.max(RouNoise_img),np.min(RouNoise_img))
                ranuNoisen_img = ranNoisen_img.copy()-RouNoise_img
                ranuNoisen_img[ranuNoisen_img<0] = 0
                an = ranuNoisen_img.copy().reshape(-1).astype(np.float32)
                MastaData.append((an,0))
                cv2.imwrite('img/make/3.png',an.reshape(480,720))
                print('noiseRO ', end='')
                del an


                GargCoods = defCirZone(insideSquare) #ゴミの座標を決める
                for k in range(len(GargCoods[0])):
                # for k in range(50,51):
                    #ゴミの付与
                    img_dust = np.zeros((480,720))
                    img_dust = cv2.ellipse(img_dust, (GargCoods[0][k],GargCoods[1][k]),(50,50), angle=0,
                            startAngle=0, endAngle=360, color=30,
                            thickness=-1, lineType=8, shift=0)
                    am = ranuNoisen_img.copy() - img_dust
                    an = am.copy().reshape(-1).astype(np.float32)
                    an[an<0]=0
                    MastaData.append((an,1))
                    cv2.imwrite('img/make/3-1.png',an.reshape(480,720))
                    print('k',end='')
                    del an
                print('')

                    
                for RouNoiseColor in [30,60]:
                # for RouNoiseColor in [90]:
                    img_dust = cv2.ellipse(ranuNoisen_img, (GargCoods[0][k],GargCoods[1][k]),(50,50), angle=0,
                            startAngle=0, endAngle=360, color=RouNoiseColor,
                            thickness=-1, lineType=8, shift=0)
                    
                    an = am.copy().reshape(-1).astype(np.float32)
                    an[an<0]=0
                    MastaData.append((an,1))
                    cv2.imwrite('img/make/3-2.png',an.reshape(480,720))
                    print(RouNoiseColor,end="")
                    del an
                print('')


        #ダミーの要素を削除
        print(MastaData.pop(0))
        print(len(MastaData))
        print(MastaData[0])
        
        with open('data/dustData'+str(j)+'.pickle', mode='wb') as fo1:
            pickle.dump(MastaData, fo1)
        
        del MastaData
        MastaData=[(0)]
        break

#enddef

if __name__=='__main__':
    
    main()
#endif