# -*- coding: utf-8 -*-
"""
一个基于Theano和Lasagne的policyValueNet的实现

作者Junxiao Song
"""

from __future__ import print_function
import theano
import theano.tensor as T
import lasagne
import pickle


class PolicyValueNet():
    """策略价值网络"""
    def __init__(self, board_width, board_height, model_file=None):
        # 棋盘宽度
        self.board_width = board_width
        # 棋盘高度
        self.board_height = board_height
        # 学习率
        self.learning_rate = T.scalar('learning_rate')
        # L2正则化系数
        self.l2_const = 1e-4  
        # 创建策略价值网络
        self.create_policy_value_net()
        # 计算损失函数并进行训练
        self._loss_train_op()
        # 如果有模型文件，则加载模型参数
        if model_file:
            try:
                net_params = pickle.load(open(model_file, 'rb'))
            except:
                # 为了支持在python3中加载预训练模型
                net_params = pickle.load(open(model_file, 'rb'),
                                         encoding='bytes')
            # 设置策略网络和价值网络的参数
            lasagne.layers.set_all_param_values(
                    [self.policy_net, self.value_net], net_params
                    )

    # 创建策略价值网络
    def create_policy_value_net(self):
        # 定义输入变量
        self.state_input = T.tensor4('state')  # 状态输入
        self.winner = T.vector('winner')  # 胜者
        self.mcts_probs = T.matrix('mcts_probs')  # MCTS概率

        # 定义网络结构
        network = lasagne.layers.InputLayer(
                shape=(None, 4, self.board_width, self.board_height),
                input_var=self.state_input
                )
        # 卷积层
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(3, 3), pad='same')  # 32个3x3的卷积核
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=64, filter_size=(3, 3), pad='same')  # 64个3x3的卷积核
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=128, filter_size=(3, 3), pad='same')  # 128个3x3的卷积核

        # 策略网络层
        policy_net = lasagne.layers.Conv2DLayer(
                network, num_filters=4, filter_size=(1, 1))  # 4个1x1的卷积核
        self.policy_net = lasagne.layers.DenseLayer(
                policy_net, num_units=self.board_width*self.board_height,
                nonlinearity=lasagne.nonlinearities.softmax)  # 输出层，使用softmax激活函数

        # 价值网络层
        value_net = lasagne.layers.Conv2DLayer(
                network, num_filters=2, filter_size=(1, 1))  # 2个1x1的卷积核
        value_net = lasagne.layers.DenseLayer(value_net, num_units=64)  # 隐藏层，64个神经元
        self.value_net = lasagne.layers.DenseLayer(
                value_net, num_units=1,
                nonlinearity=lasagne.nonlinearities.tanh)  # 输出层，使用tanh激活函数

        # 获取动作概率和状态分数值
        self.action_probs, self.value = lasagne.layers.get_output(
                [self.policy_net, self.value_net])
        self.policy_value = theano.function([self.state_input],
                                                [self.action_probs, self.value],
                                                allow_input_downcast=True)  # 编译函数

    # 定义策略价值函数
    def policy_value_fn(self, board):
        """
        输入：棋盘
        输出：每个可用动作的（动作，概率）元组列表和棋盘状态的分数
        """
        # 获取可用位置和当前状态
        legal_positions = board.availables
        current_state = board.current_state()
        # 获取动作概率和价值
        act_probs, value = self.policy_value(
            current_state.reshape(-1, 4, self.board_width, self.board_height)
            )
        # 将动作概率和可用位置打包成元组列表
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        # 返回元组列表和分数
        return act_probs, value[0][0]

    # 定义训练操作
    def _loss_train_op(self):
        """
        三个损失项：
        loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        """
        # 获取可训练参数
        params = lasagne.layers.get_all_params(
                [self.policy_net, self.value_net], trainable=True)
        # 计算价值损失
        value_loss = lasagne.objectives.squared_error(
                self.winner, self.value.flatten())
        # 计算策略损失
        policy_loss = lasagne.objectives.categorical_crossentropy(
                self.action_probs, self.mcts_probs)
        # 计算L2正则化项
        l2_penalty = lasagne.regularization.apply_penalty(
                params, lasagne.regularization.l2)
        # 计算总损失
        self.loss = self.l2_const*l2_penalty + lasagne.objectives.aggregate(
                value_loss + policy_loss, mode='mean')
        # 计算策略熵，仅用于监控
        self.entropy = -T.mean(T.sum(
                self.action_probs * T.log(self.action_probs + 1e-10), axis=1))
        # 获取训练操作
        updates = lasagne.updates.adam(self.loss, params,
                                    learning_rate=self.learning_rate)
        self.train_step = theano.function(
            [self.state_input, self.mcts_probs, self.winner, self.learning_rate],
            [self.loss, self.entropy],
            updates=updates,
            allow_input_downcast=True
            )

    # 获取策略参数
    def get_policy_param(self):
        net_params = lasagne.layers.get_all_param_values(
                [self.policy_net, self.value_net])
        return net_params

    # 保存模型
    def save_model(self, model_file):
        """ 将模型参数保存到文件 """
        # 获取模型参数
        net_params = self.get_policy_param()
        # 将模型参数保存到文件
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
