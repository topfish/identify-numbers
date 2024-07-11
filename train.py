import gzip
import os
import sys
import numpy as np
import struct
import matplotlib.pyplot as plt
import logging
import joblib


# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if sys.version_info > (3, 0):
    writemode = 'wb'
else:
    writemode = 'w'


def unzipdatasources():
    logger.info("unZip data")
    zipped_mnist = [f for f in os.listdir() if f.endswith('ubyte.gz')]
    for z in zipped_mnist:
        with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
            outfile.write(decompressed.read())


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    logger.info("load data -> " + kind)
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels


# 展示图片
def show(X_train, y_train):
    logger.info("show image")
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    # plt.savefig('images/12_5.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(25):
        img = X_train[y_train == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    # plt.savefig('images/12_6.png', dpi=300)
    plt.show()


# 数据保存到'mnist_scaled.npz'
def dumpmnist(X_train, y_train, X_test, y_test):
    logger.info("dump mnist")
    np.savez_compressed('mnist_scaled.npz',
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test)


# 定义多层感知器
class NeuralNetMLP(object):
    # 初始化参数
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    # onehot编码
    def _onehot(self, y, n_classes):
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    # sigmod激活函数
    def _sigmoid(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    # 前向传播计算
    def _forward(self, X):
        # step 1: net input of hidden layer
        z_h = np.dot(X, self.w_h) + self.b_h

        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)

        # step 3: net input of output layer
        z_out = np.dot(a_h, self.w_out) + self.b_out

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    # L2正则
    def _compute_cost(self, y_enc, output):

        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term

        return cost

    # 预测
    def predict(self, X):
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    # 训练模型---核心代码
    def fit(self, X_train, y_train, X_valid, y_valid):

        # 计算y_train中不同类别的数量，即输出层神经元数量。
        n_output = np.unique(y_train).shape[0]
        # 同上，计算特征数，即输入层神经元数量
        n_features = X_train.shape[1]

        # 初始化隐藏层的偏置为零向量，权重为正态分布
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # 初始化输出层的偏置和权重
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        # 中间数据收集
        epoch_strlen = len(str(self.epochs))  # 进度条总长
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        # 标签值进行独热编码
        y_train_enc = self._onehot(y_train, n_output)

        # 外循环，按照设置的epochs数循环（训练轮数）
        for i in range(self.epochs):

            # 按照训练集的长度创建一个数组，用于后面小批量数据取样
            indices = np.arange(X_train.shape[0])

            # 洗牌
            if self.shuffle:
                self.random.shuffle(indices)

            # 内循环，小批量循环：起始，终止，步长
            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                      1, self.minibatch_size):

                # 取一个小循环的数据
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # 1. 执行前向计算（a_out是前向计算后经激活函数后的值）
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                # 接下来是反向传播部分
                # 2. 下面三步是计算误差
                # 计算输出层的误差（a_out是模型的预测值，batch_idx是真实值）
                delta_out = a_out - y_train_enc[batch_idx]

                # 隐藏层激活函数倒数
                sigmoid_derivative_h = a_h * (1. - a_h)

                # 计算隐藏层的误差
                delta_h = (np.dot(delta_out, self.w_out.T) *
                           sigmoid_derivative_h)

                # 3. 下面两步是梯度计算
                # 计算输入到隐藏层权重和偏置的梯度
                grad_w_h = np.dot(X_train[batch_idx].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)

                # 计算隐藏到输出层权重和偏置的梯度。
                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)

                # 4. 下面两步是参数更新
                # 应用 L2 正则化，并更新隐藏层到输入层的权重和偏置。
                delta_w_h = (grad_w_h + self.l2 * self.w_h)
                delta_b_h = grad_b_h  # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                # 应用 L2 正则化，更新隐藏层到输出层的权重和偏置。
                delta_w_out = (grad_w_out + self.l2 * self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            # 在整个训练集上执行前向传播。
            z_h, a_h, z_out, a_out = self._forward(X_train)

            # 计算成本函数。
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            # 使用当前模型对训练集和验证集进行预测。
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            # 计算训练集和验证集的准确率。
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float64) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float64) /
                         X_valid.shape[0])

            # 打印当前的训练进度和性能指标。
            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i + 1, self.epochs, cost,
                              train_acc * 100, valid_acc * 100))
            sys.stderr.flush()

            # 将当前周期的成本和准确率存储到 self.eval_ 字典中。
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self


def main():
    # 数据保存到'mnist_scaled.npz'
    if not os.path.exists('mnist_scaled.npz'):
        unzipdatasources()
        X_train, y_train = load_mnist('', kind='train')
        X_test, y_test = load_mnist('', kind='t10k')
        # show(X_train, y_train)
        dumpmnist(X_train, y_train, X_test, y_test)

    # load mnist
    logger.info("load mnist")
    mnist = np.load('mnist_scaled.npz')
    X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train', 'y_train',
                                                           'X_test', 'y_test']]

    if not os.path.exists('EvaNumber.model'):
        # 模型实例化
        import config
        model = NeuralNetMLP(n_hidden=config.n_hidden,
                             l2=config.l2,
                             epochs=config.epochs,
                             eta=config.eta,
                             minibatch_size=config.minibatch_size,
                             shuffle=config.minibatch_size,
                             seed=config.seed)
        logger.info("训练模型")
        model.fit(X_train[:55000],
                  y_train[:55000],
                  X_train[55000:],
                  y_train[55000:])
        # 训练完成的模型保存一下
        joblib.dump(model, 'EvaNumber.model')
        logger.info("model dump finsh")
    # 加载模型
    logger.info("load EvaNumber.model")
    model = joblib.load('EvaNumber.model')
    # 预测测试集
    y_test_pred = model.predict(X_test)
    acc = (np.sum(y_test == y_test_pred)
           .astype(np.float64) / X_test.shape[0])

    print('Test accuracy: %.2f%%' % (acc * 100))


if __name__ == "__main__":
    main()
