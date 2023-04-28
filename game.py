# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np


# 定义棋盘类
class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        # 棋盘宽度
        self.width = int(kwargs.get('width', 8))
        # 棋盘高度
        self.height = int(kwargs.get('height', 8))
        # 棋盘状态存储为字典
        # 键：棋子位置
        # 值：玩家棋子类型
        self.states = {}
        # 获胜所需连续棋子数
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        # 玩家1和玩家2
        self.players = [1, 2]

    def init_board(self, start_player=0):
        # 棋盘宽度和高度不能小于获胜所需连续棋子数
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('棋盘宽度和高度不能小于{}'.format(self.n_in_row))
        # 当前玩家
        self.current_player = self.players[start_player]
        # 可用的落子位置列表
        self.availables = list(range(self.width * self.height))
        # 棋盘状态
        self.states = {}
        # 上一步落子位置
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3棋盘的落子位置如下：
        6 7 8
        3 4 5
        0 1 2
        落子位置5的坐标为(1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        # 坐标必须是长度为2的列表
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        # 落子位置必须在棋盘范围内
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """返回当前玩家的棋盘状态。
        状态形状：4*宽度*高度
        """

        # 初始化一个4*宽度*高度的全0数组
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            # 获取所有落子位置和玩家
            moves, players = np.array(list(zip(*self.states.items())))
            # 获取当前玩家的落子位置和对手的落子位置
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            # 在square_state中标记当前玩家和对手的落子位置
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # 标记最后一步落子位置
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            # 标记下一步落子的颜色
            square_state[3][:, :] = 1.0  
        # 返回翻转后的square_state
        return square_state[:, ::-1, :]

    # 执行一步棋
    def do_move(self, move):
        # 记录当前棋子的位置
        self.states[move] = self.current_player
        # 从可用的位置中移除当前位置
        self.availables.remove(move)
        # 切换到下一个玩家
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        # 记录最后一步棋的位置
        self.last_move = move

    #判断是否胜利
    def has_a_winner(self):
        # 获取棋盘宽度
        width = self.width
        # 获取棋盘高度
        height = self.height
        # 获取棋盘状态
        states = self.states
        # 获取连成多少颗棋子算胜利
        n = self.n_in_row

        # 获取已下棋子的位置
        moved = list(set(range(width * height)) - set(self.availables))
        # 如果已下棋子数量小于2n-1，则无法胜利
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        # 遍历已下棋子的位置
        for m in moved:
            # 获取该位置所在的行和列
            h = m // width
            w = m % width
            # 获取该位置的玩家
            player = states[m]

            # 判断是否横向连成n颗棋子
            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            # 判断是否纵向连成n颗棋子
            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            # 判断是否斜向（左上到右下）连成n颗棋子
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            # 判断是否斜向（右上到左下）连成n颗棋子
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        # 未胜利
        return False, -1

    # 检查游戏是否结束
    def game_end(self):
        # 判断是否有胜者
        win, winner = self.has_a_winner()
        if win:
            # 如果有胜者，返回True和胜者
            return True, winner
        elif not len(self.availables):
            # 如果没有胜者，但是没有可用的位置了，返回True和-1
            return True, -1
        # 如果没有胜者，也还有可用的位置，返回False和-1
        return False, -1

    # 获取当前玩家
    def get_current_player(self):
        return self.current_player


class Game(object):
    """游戏服务器"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """绘制棋盘并显示游戏信息"""
        width = board.width
        height = board.height

        # 打印玩家信息
        print("玩家", player1, "使用 X".rjust(3))
        print("玩家", player2, "使用 O".rjust(3))
        print()

        # 打印列号
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')

        # 打印棋盘
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """开始两个玩家之间的游戏"""
        # 检查起始玩家是否合法
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        # 初始化棋盘
        self.board.init_board(start_player)
        # 获取玩家1和玩家2
        p1, p2 = self.board.players
        # 设置玩家1和玩家2的编号
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        # 将玩家1和玩家2存入字典中
        players = {p1: player1, p2: player2}
        # 如果需要展示棋盘，则展示
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        # 开始游戏
        while True:
            # 获取当前玩家
            current_player = self.board.get_current_player()
            # 获取当前玩家的对象
            player_in_turn = players[current_player]
            # 获取当前玩家的行动
            move = player_in_turn.get_action(self.board)
            # 在棋盘上执行当前玩家的行动
            self.board.do_move(move)
            # 如果需要展示棋盘，则展示
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            # 判断游戏是否结束
            end, winner = self.board.game_end()
            if end:
                # 如果需要展示棋盘，则展示
                if is_shown:
                    if winner != -1:
                        print("游戏结束。获胜者是", players[winner])
                    else:
                        print("游戏结束。平局")
                # 返回获胜者的编号
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """开始自我对弈，使用MCTS玩家，重用搜索树，并存储自我对弈数据：（状态，mcts_probs，z）用于训练"""
        self.board.init_board() # 初始化棋盘
        p1, p2 = self.board.players # 获取玩家1和玩家2
        states, mcts_probs, current_players = [], [], [] # 初始化状态、概率和当前玩家
        while True:
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1) # 获取行动和概率
            # 存储数据
            states.append(self.board.current_state()) # 存储当前状态
            mcts_probs.append(move_probs) # 存储行动概率
            current_players.append(self.board.current_player) # 存储当前玩家
            # 执行行动
            self.board.do_move(move) # 执行行动
            if is_shown:
                self.graphic(self.board, p1, p2) # 显示棋盘
            end, winner = self.board.game_end() # 判断游戏是否结束
            if end:
                # 从每个状态的当前玩家的角度确定胜者
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置MCTS根节点
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("游戏结束。胜者是玩家：", winner)
                    else:
                        print("游戏结束。平局")
                return winner, zip(states, mcts_probs, winners_z) # 返回胜者和自我对弈数据
