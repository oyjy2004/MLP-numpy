from layers import FullConnectLayer, Relulayer, SoftmaxLossLayer

class NetWork(object):
    """
    建立一个神经网络，其结构如下
    输入->全连接层->激活层->全连接层->激活层->全连接层->输出层
    """
    def __init__(self, n_in, hidden1, hidden2, n_out):
        self.f1 = FullConnectLayer(n_in, hidden1)
        self.r1 = Relulayer()
        self.f2 = FullConnectLayer(hidden1, hidden2)
        self.r2 = Relulayer()
        self.f3 = FullConnectLayer(hidden2, n_out)
        self.soft = SoftmaxLossLayer()
        self.update_layer_list = [self.f1, self.f2, self.f3]

    def forward(self, input):
        """
        神经网络的前传递
        """
        h1 = self.f1.forward(input)
        h1 = self.r1.forward(h1)
        h2 = self.f2.forward(h1)
        h2 = self.r2.forward(h2)
        h3 = self.f3.forward(h2)
        output = self.soft.forward(h3)
        return output

    def backward(self):
        """
        神经网络的后传递
        """
        dloss = self.soft.backward()
        dh3 = self.f3.backward(dloss)
        dh2 = self.r2.backward(dh3)
        dh2 = self.f2.backward(dh2)
        dh1 = self.r1.backward(dh2)
        dh1 = self.f1.backward(dh1)

    def update(self):
        """
        神经网络的参数更新
        """
        for layer in self.update_layer_list:
            layer.update()

    def train(self, datas, y_tures):
        """
        神经网络的训练函数
        """
        num = 0
        losses = []
        # for data, y_ture in zip(datas, y_tures):
        for i in range(10000):
            self.forward(datas)
            self.soft.get_loss(y_tures)
            self.backward()
            self.update()
            num += 1
            if num % 10 == 0:
                loss = self.soft.get_loss(y_tures)
                losses.append(loss)
        return losses