# -*- coding: utf-8 -*-
"""
paddle中policyValueNet的实现

作者:Junxiao Song
"""

import paddle  # 导入Paddle库
import paddle.nn as nn  # 导入神经网络模块
import paddle.optimizer as optim  # 导入优化器模块
import paddle.nn.functional as F  # 导入函数模块
from paddle.autograd import Variable  # 导入自动求导模块
import numpy as np  # 导入numpy库

def set_learning_rate(optimizer, lr):
    """将学习率设置为给定值"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # 设置学习率


class Net(nn.Layer):
    """策略-价值网络模块"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width  # 棋盘宽度
        self.board_height = board_height  # 棋盘高度
        # 公共层
        self.conv1 = nn.Conv2D(4, 32, kernel_size=3, padding=1)  # 卷积层1
        self.conv2 = nn.Conv2D(32, 64, kernel_size=3, padding=1)  # 卷积层2
        self.conv3 = nn.Conv2D(64, 128, kernel_size=3, padding=1)  # 卷积层3
        # 动作策略层
        self.act_conv1 = nn.Conv2D(128, 4, kernel_size=1)  # 卷积层4
        self.act_fc1 = nn.Linear(4*board_width*board_height,  # 全连接层1
                                 board_width*board_height)
        # 状态价值层
        self.val_conv1 = nn.Conv2D(128, 2, kernel_size=1)  # 卷积层5
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)  # 全连接层2
        self.val_fc2 = nn.Linear(64, 1)  # 全连接层3

    def forward(self, state_input):
        # 公共层
        # 第一层卷积，使用ReLU激活函数
        x = F.relu(self.conv1(state_input))
        # 第二层卷积，使用ReLU激活函数
        x = F.relu(self.conv2(x))
        # 第三层卷积，使用ReLU激活函数
        x = F.relu(self.conv3(x))
        # 动作策略层
        x_act = F.relu(self.act_conv1(x))
        # 将x_act展平
        x_act = x_act.reshape(-1, 4*self.board_width*self.board_height)
        # 使用log_softmax激活函数
        x_act = F.log_softmax(self.act_fc1(x_act))
        # 状态价值层
        x_val = F.relu(self.val_conv1(x))
        # 将x_val展平
        x_val = x_val.reshape(-1, 2*self.board_width*self.board_height)
        # 第一层全连接层，使用ReLU激活函数
        x_val = F.relu(self.val_fc1(x_val))
        # 第二层全连接层，使用tanh激活函数
        x_val = F.tanh(self.val_fc2(x_val))
        # 返回动作策略层和状态价值层
        return x_act, x_val

class PolicyValueNet():
    """策略-价值网络"""
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu  # 是否使用GPU
        self.board_width = board_width  # 棋盘宽度
        self.board_height = board_height  # 棋盘高度
        self.l2_const = 1e-4  # l2正则化系数
        # 策略价值网络模块
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()  # 使用GPU
        else:
            self.policy_value_net = Net(board_width, board_height)  # 不使用GPU
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)  # Adam优化器

        if model_file:
            net_params = torch.load(model_file)  # 加载模型参数
            self.policy_value_net.load_state_dict(net_params)  # 加载模型参数到策略价值网络

    def policy_value(self, state_batch):
        """
        输入：一批状态
        输出：一批动作概率和状态值
        """
        # 如果使用GPU
        if self.use_gpu:
            # 将状态批量转换为张量并移动到GPU上
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            # 通过策略价值网络获取动作概率的对数和状态值
            log_act_probs, value = self.policy_value_net(state_batch)
            # 将动作概率的对数转换为概率
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            # 返回动作概率和状态值
            return act_probs, value.data.cpu().numpy()
        # 如果不使用GPU
        else:
            # 将状态批量转换为张量
            state_batch = Variable(torch.FloatTensor(state_batch))
            # 通过策略价值网络获取动作概率的对数和状态值
            log_act_probs, value = self.policy_value_net(state_batch)
            # 将动作概率的对数转换为概率
            act_probs = np.exp(log_act_probs.data.numpy())
            # 返回动作概率和状态值
            return act_probs, value.data.numpy()

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
        if self.use_gpu:
            # 使用GPU计算
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            # 计算动作概率
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            # 使用CPU计算
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            # 计算动作概率
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        # 将动作和概率组合成元组
        act_probs = zip(legal_positions, act_probs[legal_positions])
        # 获取分数
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """执行一步训练"""
        # 将输入数据转换为Variable类型，并放到GPU上
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda()) # 状态
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda()) # MCTS概率
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda()) # 胜者
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # 梯度清零
        self.optimizer.zero_grad()
        # 设置学习率
        set_learning_rate(self.optimizer, lr)

        # 前向传播
        log_act_probs, value = self.policy_value_net(state_batch)
        # 定义损失函数 = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # 注意：L2正则化已经在优化器中实现
        value_loss = F.mse_loss(value.view(-1), winner_batch) # 值函数损失
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1)) # 策略损失
        loss = value_loss + policy_loss # 总损失
        # 反向传播并优化
        loss.backward()
        self.optimizer.step()
        # 计算策略熵，仅用于监控
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.data[0], entropy.data[0]
        # 对于pytorch版本>=0.5，请使用以下代码
        # return loss.item(), entropy.item()

    def get_policy_param(self):
        # 获取神经网络的参数
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        # 将模型参数保存到文件中
        net_params = self.get_policy_param()  # 获取模型参数
        torch.save(net_params, model_file)
