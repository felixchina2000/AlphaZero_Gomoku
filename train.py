# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
AlphaZero的Gomoku训练流程的实现

作者:Junxiao Song
"""


from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
#from policy_value_net import PolicyValueNet  # Theano and Lasagne
#from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras
from policy_value_net_paddle import PolicyValueNet  # Paddle


class TrainPipeline():
    def __init__(self, init_model=None):
        # 棋盘和游戏的参数
        self.board_width = 6  # 棋盘宽度
        self.board_height = 6  # 棋盘高度
        self.n_in_row = 4  # 连续棋子的个数
        self.board = Board(width=self.board_width,  # 初始化棋盘
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)  # 初始化游戏
        # 训练参数
        self.learn_rate = 2e-3  # 学习率
        self.lr_multiplier = 1.0  # 根据KL自适应调整学习率
        self.temp = 1.0  # 温度参数
        self.n_playout = 400  # 每个动作的模拟次数
        self.c_puct = 5  # PUCT算法中的常数
        self.buffer_size = 10000  # 数据缓存区大小
        self.batch_size = 512  # 训练时的小批量大小
        self.data_buffer = deque(maxlen=self.buffer_size)  # 数据缓存区
        self.play_batch_size = 1  # 自我对弈时的批量大小
        self.epochs = 5  # 每次更新的训练步数
        self.kl_targ = 0.02  # KL散度的目标值
        self.check_freq = 50  # 每隔多少次自我对弈检查一次模型
        self.game_batch_num = 1500  # 自我对弈的批量数
        self.best_win_ratio = 0.0  # 最佳胜率
        # 用于评估训练好的策略的纯MCTS的模拟次数
        self.pure_mcts_playout_num = 1000
        if init_model:
            # 从一个初始的策略-价值网络开始训练
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # 从一个新的策略-价值网络开始训练
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)

            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        
        # 初始化一个MCTSPlayer                                           
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """通过旋转和翻转扩充数据集
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state]) # 旋转棋盘状态
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i) # 旋转MCTS概率
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state]) # 翻转棋盘状态
                equi_mcts_prob = np.fliplr(equi_mcts_prob) # 翻转MCTS概率
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """收集自我对弈数据以进行训练"""
        for i in range(n_games):
            # 进行自我对弈
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            # 复制play_data
            play_data = list(play_data)[:]
            # 记录每个episode的长度
            self.episode_len = len(play_data)
            # 数据增强
            play_data = self.get_equi_data(play_data)
            # 将增强后的数据加入数据缓存
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """更新策略价值网络"""
        # 从数据缓存中随机抽取一批数据
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        # 获取状态批次
        state_batch = [data[0] for data in mini_batch]
        # 获取蒙特卡洛树搜索概率批次
        mcts_probs_batch = [data[1] for data in mini_batch]
        # 获取胜者批次
        winner_batch = [data[2] for data in mini_batch]
        # 获取旧的概率和价值
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        # 迭代训练
        for i in range(self.epochs):
            # 计算损失和熵
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            # 获取新的概率和价值
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            # 计算KL散度
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            # 如果KL散度大于目标值的4倍，则提前停止训练
            if kl > self.kl_targ * 4:  
                break
        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # 计算解释方差
        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        # 打印训练结果
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        # 返回损失和熵
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        通过与纯MCTS玩家对战来评估训练好的策略
        注意：这仅用于监视训练进度
        """
        # 创建当前MCTS玩家
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        # 创建纯MCTS玩家
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        # 初始化胜利次数
        win_cnt = defaultdict(int)
        # 进行n_games次对战
        for i in range(n_games):
            # 开始对战
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            # 记录胜利次数
            win_cnt[winner] += 1
        # 计算胜率
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        # 输出结果
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        # 返回胜率
        return win_ratio

    def run(self):
        """运行训练流程"""
        try:
            for i in range(self.game_batch_num):
                # 收集自我博弈数据
                self.collect_selfplay_data(self.play_batch_size)
                # 打印当前批次和每个episode的长度
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                # 如果数据缓存区中的数据量超过batch_size，则进行策略更新
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # 检查当前模型的性能，并保存模型参数
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    # 如果胜率超过历史最佳胜率，则更新最佳策略
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # 更新最佳策略
                        self.policy_value_net.save_model('./best_policy.model')
                        # 如果最佳胜率达到1.0且纯蒙特卡罗树搜索次数小于5000，则增加纯蒙特卡罗树搜索次数
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

# 如果这个文件被直接运行，那么执行以下代码
if __name__ == '__main__':
    # 创建一个训练流水线对象
    training_pipeline = TrainPipeline()
    # 运行训练流水线
    training_pipeline.run()

