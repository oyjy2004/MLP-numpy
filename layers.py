import numpy as np

class FullConnectLayer(object):
    """
    全连接层类别定义
    """
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.w = np.random.normal(0, 0.01, (self.n_in, self.n_out))
        self.b = np.zeros(self.n_out)

    def forward(self, input):
        """
        前传递函数
        """
        self.input = input
        self.output = np.matmul(input, self.w) + self.b
        return self.output

    def backward(self, top_d):
        """
        后传递函数
        """
        self.d_w = np.dot(self.input.T, top_d)
        self.d_b = np.sum(top_d, axis=0)
        bottom_d = np.dot(top_d, self.w.T)
        return bottom_d

    def update(self):
        """
        对全连接层的参数进行更新
        """
        self.w = self.w - 0.01 * self.d_w
        self.b = self.b - 0.01 * self.d_b


class Relulayer(object):
    """
    Relu函数激活层
    """
    def __init__(self):
        return

    def forward(self, input):
        """
        前传递函数
        """
        self.input = input
        output = np.maximum(0, input)
        return output

    def backward(self, top_d):
        """
        后传递函数
        """
        bottom_d = top_d
        bottom_d[self.input < 0] = 0
        return bottom_d


class SoftmaxLossLayer(object):
    """
    softmax输出层
    """
    def __init__(self):
        return

    def forward(self, input):
        """
        前传递函数
        """
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.y_pred = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.y_pred

    def get_loss(self, y_ture):
        """
        计算损失
        """
        self.out_num = self.y_pred.shape[0]
        self.y_ture_onehot = np.zeros_like(self.y_pred)
        self.y_ture_onehot[np.arange(self.out_num), y_ture] = 1.0
        loss = -np.sum(np.log(self.y_pred) * self.y_ture_onehot) / self.out_num
        return loss

    def backward(self):
        """
        后传递函数
        """
        bottom_d = (self.y_pred - self.y_ture_onehot) / self.out_num
        return bottom_d
