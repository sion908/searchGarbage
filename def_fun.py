import sys
sys.path.append(r'c:\programdata\anaconda3\lib\site-packages')#chainer導入のための絶対パス
import chainer
import chainer.links as L
import chainer.functions as F

# ネットワークを定義するクラス
class MLP(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=11):
        super(MLP, self).__init__()

        # 各層の定義
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_mid_units)
            self.l4 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        # ネットワークの定義
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return self.l4(h3)



if __name__=='__main__':
    print('クラス定義')
