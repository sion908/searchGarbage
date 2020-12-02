import pickle
import matplotlib.pyplot as plt
import random
import numpy
import chainer
from chainer.datasets import split_dataset_random
from chainer import iterators
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
from chainer import serializers
import def_fun
import sys


# 乱数を初期化する関数
def reset_seed(seed=0):
    random.seed(seed)
    numpy.random.seed(seed)



# # ローカルファイルからデータセットを読み込む
# with open('test.pickle', mode='rb') as fi1:
#     test = pickle.load(fi1)
# with open('test.pickle', mode='rb') as fi2:
#     train_val = pickle.load(fi2)
with open('data/dustData.pickle', mode='rb') as fi2:
    dataset = pickle.load(fi2)

# train:valid:test = 7:1:2
n_train = int(len(dataset) * 0.7)
n_valid = int(len(dataset) * 0.1)
# データセットを学習用と検証用に分割する
train, valid_test = split_dataset_random(dataset, n_train, seed=0)
valid, test = split_dataset_random(valid_test, n_valid, seed=0)
# イテレーターの設定
batchsize = 5
train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)
test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

# 乱数の初期設定
reset_seed(0)

# ネットワークのインスタンスを作る
net = def_fun.MLP()

# オプティマイザーの設定
optimizer = optimizers.SGD(lr=0.01).setup(net)

# 学習の実行
max_epoch = 10
gpu_id = -1 # GPU不使用
count = 0
output =[]
while train_iter.epoch < max_epoch:
    print(count)
    count+=1
    train_batch = train_iter.next()

    x, t = concat_examples(train_batch, gpu_id)
    
    # 予測値の計算
    y = net(x)

    # ロスの計算
    loss = F.softmax_cross_entropy(y, t)

    # 勾配の計算
    net.cleargrads()
    loss.backward()

    # パラメータの更新
    optimizer.update()

    # 1エポック終わったらロスと精度を表示する
    if train_iter.is_new_epoch:
        # ロスの表示
        print('epoch:{:02d} train_loss:{:.04f} '.format(train_iter.epoch, float(to_cpu(loss.data))), end='')
        output.append(count)
        valid_losses = []
        valid_accuracies = []
        while True:
            valid_batch = valid_iter.next()
            x_valid, t_valid = concat_examples(valid_batch, gpu_id)

            # Validationデータをforward
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                y_valid = net(x_valid)

            # ロスを計算
            loss_valid = F.softmax_cross_entropy(y_valid, t_valid)
            valid_losses.append(to_cpu(loss_valid.array))

            # 精度を計算
            accuracy = F.accuracy(y_valid, t_valid)
            accuracy.to_cpu()
            valid_accuracies.append(accuracy.array)

            if valid_iter.is_new_epoch:
                valid_iter.reset()
                break

        print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(numpy.mean(valid_losses), numpy.mean(valid_accuracies)))


# テストデータでの評価
test_accuracies = []
while True:
    test_batch = test_iter.next()
    x_test, t_test = concat_examples(test_batch, gpu_id)

    # テストデータをforward
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y_test = net(x_test)

    # 精度を計算
    accuracy = F.accuracy(y_test, t_test)
    accuracy.to_cpu()
    test_accuracies.append(accuracy.array)

    if test_iter.is_new_epoch:
        test_iter.reset()
        break

print('test_accuracy:{:.04f}'.format(numpy.mean(test_accuracies)))

# 学習結果の保存
serializers.save_npz('./data/model/Robomas_mnist.model', net)
# 出力はこうなります。

# epoch:01 train_loss:0.9035 val_loss:0.9046 val_accuracy:0.8071
# epoch:02 train_loss:0.4777 val_loss:0.5205 val_accuracy:0.8667
# epoch:03 train_loss:0.4600 val_loss:0.4219 val_accuracy:0.8851
# epoch:04 train_loss:0.3510 val_loss:0.3747 val_accuracy:0.8955
# epoch:05 train_loss:0.2335 val_loss:0.3468 val_accuracy:0.9021
# epoch:06 train_loss:0.2353 val_loss:0.3288 val_accuracy:0.9049
# epoch:07 train_loss:0.3196 val_loss:0.3137 val_accuracy:0.9100
# epoch:08 train_loss:0.2127 val_loss:0.2988 val_accuracy:0.9156
# epoch:09 train_loss:0.4406 val_loss:0.2892 val_accuracy:0.9155
# epoch:10 train_loss:0.2950 val_loss:0.2790 val_accuracy:0.9194
# test_accuracy:0.9233