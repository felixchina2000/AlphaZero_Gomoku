# -*- coding: utf-8 -*-
"""
蒙特卡罗树搜索采用AlphaGo Zero风格,使用策略价值网络来指导树搜索和评估叶节点。

作者：宋俊霄

"""

import numpy as np
import copy


def softmax(x):
    # 计算指数
    probs = np.exp(x - np.max(x))
    # 归一化
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """MCTS树中的一个节点。

    每个节点都跟踪其自身的值Q、先前概率P和其访问次数调整后的先前分数u。
    """

    def __init__(self, parent, prior_p):
        self._parent = parent  # 父节点
        self._children = {}  # 一个从动作到TreeNode的映射
        self._n_visits = 0  # 访问次数
        self._Q = 0  # 值Q
        self._u = 0  # 先前分数u
        self._P = prior_p  # 先前概率P

    def expand(self, action_priors):
        """通过创建新的子节点来扩展树。
        action_priors: 一个由动作和它们根据策略函数的先前概率组成的元组列表。
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """在子节点中选择给出最大动作值Q和奖励u(P)的动作。
        返回：一个元组(action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """更新节点值，从叶子节点的评估中得到。
        leaf_value: 从当前玩家的角度来看，子树评估的值。
        """
        # 计算访问次数。
        self._n_visits += 1
        # 更新Q，所有访问的值的运行平均值。
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """类似于调用update()，但递归应用于所有祖先。
        """
        # 如果不是根节点，则应首先更新此节点的父节点。
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算并返回此节点的值。
        它是叶子评估Q和此节点的先验概率的组合，根据其访问计数u进行调整。
        c_puct: (0，inf)之间的数字，控制值Q和先前概率P对此节点得分的相对影响。
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """检查是否为叶节点（即，此节点下面没有扩展的节点）。"""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """蒙特卡罗树搜索的实现。"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: 一个函数，它接受一个棋盘状态并输出
            一个由(action, probability)元组组成的列表，还有一个在[-1,1]范围内的分数
            （即从当前玩家的角度来看，期望的最终得分），用于当前玩家。
        c_puct: 一个在(0,inf)范围内的数字，用于控制探索的速度
            收敛到最大值策略。一个更高的值意味着
            更依赖于先前的策略。
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """从根到叶子运行单个模拟，获取叶子的值并将其传播回其父级。
        状态是原地修改的，因此必须提供副本。"""
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # 贪婪地选择下一个移动。
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 使用输出(action, probability)元组列表和[-1,1]范围内的分数v评估叶子
        # 用于当前玩家。
        action_probs, leaf_value = self._policy(state)
        # 检查游戏是否结束。
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # 对于结束状态，返回“真实”的叶值
            if winner == -1:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # 更新此遍历中节点的值和访问计数。
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """运行所有模拟并返回可用动作及其相应的概率。
        state: 当前游戏状态
        temp: 温度参数在(0,1]之间，控制探索的程度
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 基于根节点的访问计数计算移动概率
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """向前迈进树，保留我们已经知道的子树的所有内容。
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """基于MCTS的AI玩家"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        # 初始化MCTS
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        # 设置玩家编号
        self.player = p

    def reset_player(self):
        # 重置玩家
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        # 获取可行的落子位置
        sensible_moves = board.availables
        # 从MCTS获取落子概率向量pi
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # 对概率向量pi添加狄利克雷噪声以进行探索（自我对弈训练需要）
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 更新根节点并重用搜索树
                self.mcts.update_with_move(move)
            else:
                # 对于默认的temp=1e-3，它几乎等价于选择概率最高的落子
                move = np.random.choice(acts, p=probs)
                # 重置根节点
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("警告：棋盘已满")

    def __str__(self):
        return "MCTS {}".format(self.player)
