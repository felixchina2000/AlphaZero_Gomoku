# -*- coding: utf-8 -*-
"""
人类对抗AI模型
请以“行，列”的格式输入您的落子位置：

作者：宋俊霄
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras


# 定义一个人类玩家类
class Human(object):
    """
    人类玩家
    """

    def __init__(self):
        self.player = None

    # 设置玩家编号2
    def set_player_ind(self, p):
        self.player = p

    # 获取玩家行动
    def get_action(self, board):
        try:
            # 获取玩家输入
            location = input("Your move: ")
            # 如果是字符串，转换为数字列表
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            # 将数字列表转换为落子位置
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        # 如果落子位置无效，重新获取玩家行动
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5  # 棋子连成一线的数量
    width, height = 8, 8  # 棋盘的宽度和高度
    model_file = 'best_policy_8_8_5.model'  # 存储最佳策略的文件名
    try:
        board = Board(width=width, height=height, n_in_row=n)  # 初始化棋盘
        game = Game(board)  # 初始化游戏

        # ############### 人类 VS AI ###################
        # 加载训练好的策略价值网络，支持Theano/Lasagne、PyTorch或TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # 加载提供的模型（在Theano/Lasagne中训练）到一个纯numpy写的MCTS player中
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')  # 为了支持python3
        best_policy = PolicyValueNetNumpy(width, height, policy_param)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)  # 设置更大的n_playout以获得更好的性能

        # 取消下面一行的注释以使用纯MCTS（即使n_playout更大，它也更弱）
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # 人类玩家，以“2,3”的格式输入您的移动
        human = Human()

        # 设置start_player=0以人类先手
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
