import os
import numpy as np
from BulidNetwork import NetWork
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

network = NetWork(64, 32, 16, 10)

digits = load_digits()
train_datas = digits.data[0:1000]
train_tures = digits.target[0:1000]
# print(datas[1].shape)

lossdatas = network.train(train_datas, train_tures)
epoch = range(0, 10000, 10)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Neural Network Loss vs. Epochs")
plt.plot(epoch, lossdatas)

if (os.path.isfile("page.pgn")):
    os.remove("page.png")
plt.savefig(fname="page.png")
plt.show()

test_datas = digits.data[1000:1500]
test_tures = digits.target[1000:1500]
test_pred = np.argmax(network.forward(test_datas), axis=1)
print(test_pred)
print(test_tures)

count = 0
for i, j in zip(test_tures, test_pred):
    if i == j:
        count += 1

print(count/500)