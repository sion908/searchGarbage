import numpy as np
import sys

def reshapeImg(img):

    #480 -> 50
    
    heightStart,widthStart = 0,0
    ans = np.empty((50*50),dtype=int)
    for Height in range(50):
        if 14 < Height and Height < 35:
            heightDiv = 9
            heightStart = 150 - 9 * 15
        else:
            heightDiv = 10 

        if Height == 35:
            heightStart = 330 - 10 * 35
        widthStart = 0
        for Width in range(50):
            if 14 < Width and Width < 35:
                widthDiv = 9
                widthStart = 150 - 9 * 15
            else:
                widthDiv = 10

            if Width == 35:
                widthStart = 330 - 10 * 35

            ans[Height*50 + Width] = np.mean(img[heightStart+Height*heightDiv:heightStart+(Height+1)*heightDiv , widthStart+Width*widthDiv:widthStart+(Width+1)*widthDiv])
            # sys.exit()
    
    return ans

def reshapeImge(img):
    #480 -> 15
    
    heightStart,widthStart = 0,0
    ans = np.empty((15*15),dtype=int)
    for Height in range(15):
        if 14 < Height and Height < 35:
            heightDiv = 9
            heightStart = 150
        else:
            heightDiv = 10

        if Height == 35:
            heightStart = 330

        for Width in range(15):
            if 14 < Height and Height < 35:
                widthDiv = 9
                widthStart = 150
            else:
                widthDiv = 10

            if Height == 35:
                heightStart = 330
            ans[Height + Width] = np.mean(img[heightStart+Height*heightDiv:heightStart+(Height+1)*heightDiv , widthStart+Width*widthDiv:widthStart+(Width+1)*widthDiv])

    return ans

def play(img):
    import pickle
    import matplotlib.pyplot as plt
    import chainer
    import chainer.links as L
    import chainer.functions as F
    from chainer import serializers
    import def_fun

    # ネットワークのインスタンスを作る
    infer_net = def_fun.MLP()

    # ネットワークに学習済みのパラメータを読み込む
    serializers.load_npz('./data/model/Robomas_mnist.model', infer_net)

    # データの準備
    

    # #  形の変換
    # ans=[]
    # for x in reshapeImg(img):
    #     for y in x:
    #         g = 1 - sum(y) / 3 / 255
    #         ans.append(g)
            
    # x = np.array([reshapeImg(img)],dtype=np.float32) #入力が480*480の場合
    # x = np.array(img,dtype=np.float32) #入力を50*50の画像にした時,平坦化しておくこと
    x=img
    # ネットワークと同じデバイス上にデータを送る
    x = infer_net.xp.asarray(x)

    # モデルのforward関数に渡す
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = infer_net(x)

    # Variable形式で出てくるので中身を取り出す
    ans = y.array.argmax(axis=1)
    # print(y)
    return int(ans[0])
    # return ans

if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    import time
    
    filepath = './data/img/0.png'
    # img = Image
    img = np.array(Image.open(filepath).resize((50,50)))
    # print(np.ravel(img).shape)
    # print(np.ravel(img))
    start = time.time()
    
    result = play(np.array([np.ravel(img).tolist()],dtype=np.float32))
    elapsed_time = time.time() - start
    print(result)
    # print(type(result))
    # print(result.shape)
    print(elapsed_time)