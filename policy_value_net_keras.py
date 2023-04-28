# -*- coding: utf-8 -*-
"""
使用Keras实现的policyValueNet
在Keras 2.0.5和tensorflow-gpu 1.2.1作为后端测试通过

@author: Mingxu Zhang
""" 

from __future__ import print_function

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K

from keras.utils import np_utils

import numpy as np
import pickle


class PolicyValueNet():
    """策略-价值网络"""
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width  # 棋盘宽度
        self.board_height = board_height  # 棋盘高度
        self.l2_const = 1e-4  # l2正则化系数
        self.create_policy_value_net()  # 创建策略-价值网络
        self._loss_train_op()  # 损失函数和训练操作

        if model_file:
            net_params = pickle.load(open(model_file, 'rb'))  # 加载模型参数
            self.model.set_weights(net_params)  # 设置模型参数
        
    # 创建策略价值网络
    def create_policy_value_net(self):
        # 输入层
        in_x = network = Input((4, self.board_width, self.board_height))

        # 卷积层
        # 第一层卷积
        network = Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        # 第二层卷积
        network = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        # 第三层卷积
        network = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)

        # 策略网络层
        policy_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        policy_net = Flatten()(policy_net)
        self.policy_net = Dense(self.board_width*self.board_height, activation="softmax", kernel_regularizer=l2(self.l2_const))(policy_net)

        # 价值网络层
        value_net = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        value_net = Flatten()(value_net)
        value_net = Dense(64, kernel_regularizer=l2(self.l2_const))(value_net)
        self.value_net = Dense(1, activation="tanh", kernel_regularizer=l2(self.l2_const))(value_net)

        # 定义模型
        self.model = Model(in_x, [self.policy_net, self.value_net])
        
        # 定义策略价值函数
        def policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results
        self.policy_value = policy_value

        
    def policy_value_fn(self, board):
        """
        输入：棋盘
        输出：每个可用动作的（动作，概率）元组列表和棋盘状态的分数
        """
        # 获取可用位置
        legal_positions = board.availables
        # 获取当前状态
        current_state = board.current_state()
        # 获取动作概率和状态值
        act_probs, value = self.policy_value(current_state.reshape(-1, 4, self.board_width, self.board_height))
        # 将动作概率和可用位置打包成元组列表
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        # 返回动作概率和状态值
        return act_probs, value[0][0]

    # 定义_loss_train_op函数
    def _loss_train_op(self):
        """
        三个损失项：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """

        # 获取训练操作
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=opt, loss=losses)

        # 定义self_entropy函数
        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        # 定义train_step函数
        def train_step(state_input, mcts_probs, winner, learning_rate):
            state_input_union = np.array(state_input)
            mcts_probs_union = np.array(mcts_probs)
            winner_union = np.array(winner)
            # 计算损失
            loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            # 预测动作概率
            action_probs, _ = self.model.predict_on_batch(state_input_union)
            # 计算熵
            entropy = self_entropy(action_probs)
            # 设置学习率
            K.set_value(self.model.optimizer.lr, learning_rate)
            # 模型训练
            self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            return loss[0], entropy
        
        # 将train_step函数赋值给self.train_step
        self.train_step = train_step

    def get_policy_param(self):
        # 获取神经网络的参数
        net_params = self.model.get_weights()        
        return net_params

    def save_model(self, model_file):
        # 将模型参数保存到文件中
        net_params = self.get_policy_param()
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)

