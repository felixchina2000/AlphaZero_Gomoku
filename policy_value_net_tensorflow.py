# -*- coding: utf-8 -*-
"""
Tensorflow中policyValueNet的实现
在Tensorflow 1.4和1.5中测试

作者:Xiang Zhong
"""

import numpy as np
import tensorflow as tf


# 定义策略价值网络类
class PolicyValueNet():
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height

        # 定义tensorflow神经网络
        # 1. 输入层:
        self.input_states = tf.placeholder(
                tf.float32, shape=[None, 4, board_height, board_width])
        self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1])
        # 2. 公共网络层
        self.conv1 = tf.layers.conv2d(inputs=self.input_state,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", data_format="channels_last",
                                      activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        # 3-1 动作网络层
        self.action_conv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                            kernel_size=[1, 1], padding="same",
                                            data_format="channels_last",
                                            activation=tf.nn.relu)
        # 将张量展平
        self.action_conv_flat = tf.reshape(
                self.action_conv, [-1, 4 * board_height * board_width])
        # 3-2 全连接层，输出每个棋盘格上移动的对数概率
        self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                         units=board_height * board_width,
                                         activation=tf.nn.log_softmax)
        # 4 评估网络层
        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                data_format="channels_last",
                                                activation=tf.nn.relu)
        self.evaluation_conv_flat = tf.reshape(
                self.evaluation_conv, [-1, 2 * board_height * board_width])
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu)
        # 输出当前状态评估的分数
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                            units=1, activation=tf.nn.tanh)

        # 定义损失函数
        # 1. 标签：包含每个状态游戏是否获胜的数组
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 2. 预测：包含每个状态评估分数的数组，即self.evaluation_fc2
        # 3-1. 值损失函数
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                    self.evaluation_fc2)
        # 3-2. 策略损失函数
        self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, board_height * board_width])
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
        # 3-3. L2正则化项
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 将上述三项相加得到损失函数
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # 定义用于训练的优化器
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # 创建会话
        self.session = tf.Session()

        # 计算策略熵，仅用于监控
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # 初始化变量
        init = tf.global_variables_initializer()
        self.session.run(init)

        # 用于保存和恢复模型
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)

    def policy_value(self, state_batch):
        """
        输入：一批状态
        输出：一批动作概率和状态值
        """
        # 计算动作概率和状态值
        log_act_probs, value = self.session.run(
                [self.action_fc, self.evaluation_fc2],
                feed_dict={self.input_states: state_batch}
                )
        # 计算指数动作概率
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board):
        """
        输入：棋盘
        输出：每个可用动作的（动作，概率）元组列表和棋盘状态的分数
        """
        # 获取可用位置
        legal_positions = board.availables
        # 获取当前状态
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        # 获取动作概率和状态分数
        act_probs, value = self.policy_value(current_state)
        # 将动作概率和可用位置打包成元组列表
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """执行一步训练"""
        # 将winner_batch变形为(-1, 1)的形状
        winner_batch = np.reshape(winner_batch, (-1, 1))
        # 运行session，获取loss、entropy和optimizer
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr})
        return loss, entropy

    def save_model(self, model_path):
        """保存模型"""
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        """恢复模型"""
        self.saver.restore(self.session, model_path)
