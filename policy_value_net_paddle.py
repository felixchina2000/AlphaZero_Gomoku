# -*- coding: utf-8 -*-
"""
paddle中policyValueNet的实现

作者:Junxiao Song
"""

# 导入PaddlePaddle和Numpy库
import paddle
import numpy as np

# 导入PaddlePaddle的神经网络模块和函数模块
import paddle.nn as nn 
import paddle.nn.functional as F


class Net(paddle.nn.Layer):
    def __init__(self,board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # 公共网络层
        # 第一层卷积层，输入通道数为4，输出通道数为32，卷积核大小为3，padding为1
        self.conv1 = nn.Conv2D(in_channels=4,out_channels=32,kernel_size=3,padding=1)
        # 第二层卷积层，输入通道数为32，输出通道数为64，卷积核大小为3，padding为1
        self.conv2 = nn.Conv2D(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        # 第三层卷积层，输入通道数为64，输出通道数为128，卷积核大小为3，padding为1
        self.conv3 = nn.Conv2D(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        # 行动策略网络层
        # 第一层卷积层，输入通道数为128，输出通道数为4，卷积核大小为1，padding为0
        self.act_conv1 = nn.Conv2D(in_channels=128,out_channels=4,kernel_size=1,padding=0)
        # 第一层全连接层，输入大小为4*棋盘宽度*棋盘高度，输出大小为棋盘宽度*棋盘高度
        self.act_fc1 = nn.Linear(4*self.board_width*self.board_height,
                                 self.board_width*self.board_height)
        # 估值网络层
        # 第一层卷积层，输入通道数为128，输出通道数为2，卷积核大小为1，padding为0
        self.val_conv1 = nn.Conv2D(in_channels=128,out_channels=2,kernel_size=1,padding=0)
        # 第一层全连接层，输入大小为2*棋盘宽度*棋盘高度，输出大小为64
        self.val_fc1 = nn.Linear(2*self.board_width*self.board_height, 64)
        # 第二层全连接层，输入大小为64，输出大小为1
        self.val_fc2 = nn.Linear(64, 1)


    def forward(self, inputs):
        # 公共网络层 
        x = F.relu(self.conv1(inputs)) # 第一层卷积，使用ReLU激活函数
        x = F.relu(self.conv2(x)) # 第二层卷积，使用ReLU激活函数
        x = F.relu(self.conv3(x)) # 第三层卷积，使用ReLU激活函数
        # 行动策略网络层
        x_act = F.relu(self.act_conv1(x)) # 行动策略网络的第一层卷积，使用ReLU激活函数
        x_act = paddle.reshape(
                x_act, [-1, 4 * self.board_height * self.board_width]) # 将行动策略网络的输出展平
        x_act  = F.log_softmax(self.act_fc1(x_act)) # 行动策略网络的全连接层，使用log_softmax激活函数        
        # 状态价值网络层
        x_val  = F.relu(self.val_conv1(x)) # 状态价值网络的第一层卷积，使用ReLU激活函数
        x_val = paddle.reshape(
                x_val, [-1, 2 * self.board_height * self.board_width]) # 将状态价值网络的输出展平
        x_val = F.relu(self.val_fc1(x_val)) # 状态价值网络的第一层全连接层，使用ReLU激活函数
        x_val = F.tanh(self.val_fc2(x_val)) # 状态价值网络的第二层全连接层，使用tanh激活函数

        return x_act,x_val # 返回行动策略网络和状态价值网络的输出

# 定义策略&值网络类
class PolicyValueNet():
    """策略&值网络 """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu  # 是否使用GPU
        self.board_width = board_width  # 棋盘宽度
        self.board_height = board_height  # 棋盘高度
        self.l2_const = 1e-3  # l2正则化系数
        
        # 创建策略&值网络
        self.policy_value_net = Net(self.board_width, self.board_height)        
        
        # 定义优化器
        self.optimizer  = paddle.optimizer.Adam(learning_rate=0.02,
                                parameters=self.policy_value_net.parameters(), weight_decay=self.l2_const)
                                     
        # 如果有模型文件，则加载模型参数
        if model_file:
            net_params = paddle.load(model_file)
            self.policy_value_net.set_state_dict(net_params)
            
    # 定义policy_value函数
    def policy_value(self, state_batch):
        """
        输入：一批状态
        输出：一批动作概率和状态值
        """
        # 将状态批量转换为paddle张量
        state_batch = paddle.to_tensor(state_batch)
        # 通过策略价值网络获取动作概率和状态值
        log_act_probs, value = self.policy_value_net(state_batch)
        # 将对数概率转换为概率
        act_probs = np.exp(log_act_probs.numpy())
        return act_probs, value.numpy()

    # 定义policy_value_fn函数
    def policy_value_fn(self, board):
        """
        输入：棋盘
        输出：每个可用动作的（动作，概率）元组列表和棋盘状态的分数
        """
        # 获取可用位置和当前状态
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height)).astype("float32")
        # 将当前状态转换为paddle张量
        current_state = paddle.to_tensor(current_state)
        # 通过策略价值网络获取动作概率和状态值
        log_act_probs, value = self.policy_value_net(current_state)
        # 将对数概率转换为概率
        act_probs = np.exp(log_act_probs.numpy().flatten())
        # 将可用位置和对应的概率组成元组
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value.numpy()

    # 执行一步训练
    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        """执行一步训练"""
        # 将数据转换为Tensor
        state_batch = paddle.to_tensor(state_batch)
        mcts_probs = paddle.to_tensor(mcts_probs)
        winner_batch = paddle.to_tensor(winner_batch)

        # 梯度清零
        self.optimizer.clear_gradients()
        # 设置学习率
        self.optimizer.set_lr(lr)

        # 前向传播
        log_act_probs, value = self.policy_value_net(state_batch)
        # 定义损失函数 = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # 注意：L2惩罚项已经在优化器中实现
        value = paddle.reshape(x=value, shape=[-1])
        value_loss = F.mse_loss(input=value, label=winner_batch)
        policy_loss = -paddle.mean(paddle.sum(mcts_probs*log_act_probs, axis=1))
        loss = value_loss + policy_loss
        # 反向传播并优化
        loss.backward()
        self.optimizer.minimize(loss)
        # 计算策略熵，仅用于监控
        entropy = -paddle.mean(
                paddle.sum(paddle.exp(log_act_probs) * log_act_probs, axis=1)
                )
        return loss.numpy(), entropy.numpy()[0]    

    # 获取策略网络参数
    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    # 保存模型参数到文件
    def save_model(self, model_file):
        """保存模型参数到文件"""
        net_params = self.get_policy_param()  # 获取模型参数
        paddle.save(net_params, model_file)
