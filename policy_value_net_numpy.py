# -*- coding: utf-8 -*-
"""
使用numpy实现策略价值网络,这样我们就可以在不安装任何DL框架的情况下玩训练好的AI模型了

作者:Junxiao Song
"""

from __future__ import print_function
import numpy as np

# 一些实用函数
def softmax(x): # softmax函数
    probs = np.exp(x - np.max(x)) # 指数化并减去最大值
    probs /= np.sum(probs) # 归一化
    return probs # 返回概率值

# relu函数
def relu(X): 
    out = np.maximum(X, 0) # 取X和0中的最大值
    return out # 返回结果


# 定义卷积前向传播函数
def conv_forward(X, W, b, stride=1, padding=1):
    # 获取卷积核的形状
    n_filters, d_filter, h_filter, w_filter = W.shape
    # theano conv2d 在计算时先翻转（旋转180度）卷积核
    W = W[:, :, ::-1, ::-1]
    # 获取输入数据的形状
    n_x, d_x, h_x, w_x = X.shape
    # 计算输出数据的高度和宽度
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    # 将输出数据的高度和宽度转换为整数
    h_out, w_out = int(h_out), int(w_out)
    # 将输入数据转换为矩阵形式
    X_col = im2col_indices(X, h_filter, w_filter,
                           padding=padding, stride=stride)
    # 将卷积核转换为矩阵形式
    W_col = W.reshape(n_filters, -1)
    # 计算卷积
    out = (np.dot(W_col, X_col).T + b).T
    # 将输出数据转换为四维张量形式
    out = out.reshape(n_filters, h_out, w_out, n_x)
    # 转换输出数据的维度顺序
    out = out.transpose(3, 0, 1, 2)
    # 返回输出数据
    return out

# 定义全连接层的前向传播函数
def fc_forward(X, W, b):
    # 计算全连接层的输出
    out = np.dot(X, W) + b
    return out

# 定义获取卷积层im2col操作的索引函数
def get_im2col_indices(x_shape, field_height,
                       field_width, padding=1, stride=1):
    # 获取输入数据的维度信息
    N, C, H, W = x_shape
    # 判断卷积操作后输出的高度和宽度是否为整数
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    # 计算卷积操作后输出的高度和宽度
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    # 计算im2col操作的索引
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))

# 定义im2col_indices函数，用于将输入的x转换为cols
def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # 对输入进行零填充
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    # 获取k, i, j
    k, i, j = get_im2col_indices(x.shape, field_height,
                                 field_width, padding, stride)

    # 获取cols
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    # 转置cols
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

# 定义一个使用numpy实现的策略价值网络类
class PolicyValueNetNumpy():
    """定义一个使用numpy实现的策略价值网络类 """
    def __init__(self, board_width, board_height, net_params):
        # 棋盘宽度
        self.board_width = board_width
        # 棋盘高度
        self.board_height = board_height
        # 神经网络参数
        self.params = net_params

    def policy_value_fn(self, board):
        """
        输入：棋盘
        输出：每个可用动作的(action, probability)元组列表和棋盘状态的分数
        """
        # 获取可用位置
        legal_positions = board.availables
        # 获取当前状态
        current_state = board.current_state()

        # 将当前状态转换为神经网络的输入格式
        X = current_state.reshape(-1, 4, self.board_width, self.board_height)
        # 前三个卷积层使用ReLu非线性激活函数
        for i in [0, 2, 4]:
            X = relu(conv_forward(X, self.params[i], self.params[i+1]))
        # 策略网络
        X_p = relu(conv_forward(X, self.params[6], self.params[7], padding=0))
        X_p = fc_forward(X_p.flatten(), self.params[8], self.params[9])
        # 计算每个动作的概率
        act_probs = softmax(X_p)
        # 价值网络
        X_v = relu(conv_forward(X, self.params[10],
                                self.params[11], padding=0))
        X_v = relu(fc_forward(X_v.flatten(), self.params[12], self.params[13]))
        # 计算当前状态的分数
        value = np.tanh(fc_forward(X_v, self.params[14], self.params[15]))[0]
        # 将可用位置和对应的概率组成元组列表
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value

