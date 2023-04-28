# -*- coding: utf-8 -*-
"""
纯蒙特卡罗树搜索(MCTS)的实现

@author: Junxiao Song
"""

# 导入必要的库
import numpy as np
import copy
from operator import itemgetter


# 定义模拟策略函数
def rollout_policy_fn(board):
    """用于模拟阶段的策略函数，是一个粗略、快速的版本。"""
    # 随机模拟
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """定义一个函数，输入一个棋盘状态，输出一个列表，其中包含了所有可行的落子位置及其对应的概率
    同时输出当前状态的分数"""
    # 对于纯MCTS算法，返回均匀概率和0分
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    """MCTS树中的一个节点。每个节点都会跟踪自己的价值Q、先验概率P和其访问次数调整后的先验分数u。"""

    def __init__(self, parent, prior_p):
        self._parent = parent  # 父节点
        self._children = {}  # 一个从动作到TreeNode的映射
        self._n_visits = 0  # 访问次数
        self._Q = 0  # 节点的价值Q
        self._u = 0  # 节点的访问次数调整后的先验分数u
        self._P = prior_p  # 先验概率P

    def expand(self, action_priors):
        """通过创建新的子节点来扩展树。
        action_priors: 一个由动作和它们在策略函数中的先验概率组成的元组列表。
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """在子节点中选择给定动作中提供最大动作价值Q加上奖励u(P)的动作。
        返回：一个由动作和下一个节点组成的元组。
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """从叶子节点的评估中更新节点值。
        leaf_value: 从当前玩家的角度评估的子树评估值。
        """
        # 计算访问次数
        self._n_visits += 1
        # 更新Q，即所有访问的值的运行平均值
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """递归更新所有祖先节点的值,类似于调用update()。
        """
        # 如果不是根节点，应该先更新此节点的父节点。
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算并返回此节点的值。
        它是叶子节点评估Q和此节点的先验概率的组合,
        调整为其访问次数u。
        c_puct: (0,inf)之间的数字,控制值Q和先验概率P对此节点得分的相对影响。
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """检查是否为叶子节点（即此节点下面没有扩展节点）。
        """
        return self._children == {}

    def is_root(self):
        """检查是否为根节点。
        """
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: 一个函数，它接受一个棋盘状态并输出
        一个由（动作，概率）元组组成的列表，还有一个得分在[-1,1]之间
        （即从当前玩家的角度看，预期的最终游戏得分）为当前玩家。
        c_puct: 一个在(0,inf)之间的数字,用于控制探索速度
        收敛到最大值策略。更高的值意味着
        更多地依赖于先前的策略。
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """从根到叶子运行单个模拟，获取值
        在叶子和通过其父级向后传播。
        状态在原地修改，因此必须提供副本。
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # 贪心地选择下一个移动。
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        # 检查游戏结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        # 通过随机模拟评估叶节点
        leaf_value = self._evaluate_rollout(state)
        # 更新此遍历中节点的值和访问计数。
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """使用模拟策略玩到游戏结束，
        如果当前玩家获胜，则返回+1,如果对手获胜,则返回-1,
        如果是平局,则返回0。
        """
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # 如果循环没有中断，则发出警告。
            print("警告：模拟达到移动限制")
        if winner == -1:  # 平局
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        """顺序运行所有模拟，并返回最多访问的动作。
        state:当前游戏状态
        
        返回：所选动作
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """向前迈进树，保留我们已经知道的所有内容
        关于子树。
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=2000):
        # 初始化MCTS算法，传入策略价值函数、c_puct常数和模拟次数
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        # 设置玩家编号
        self.player = p

    def reset_player(self):
        # 重置MCTS算法
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        # 获取可行的落子位置
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            # 通过MCTS算法获取最佳落子位置
            move = self.mcts.get_move(board)
            # 更新MCTS算法
            self.mcts.update_with_move(-1)
            return move
        else:
            # 若无可行落子位置，输出警告信息
            print("WARNING: the board is full")

    def __str__(self):
        # 返回玩家信息
        return "MCTS {}".format(self.player)
