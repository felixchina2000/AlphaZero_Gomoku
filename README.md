## AlphaZero-Gomoku
这是一个使用纯自我对弈训练玩简单棋盘游戏五子棋（也称为五子棋或五子棋）的AlphaZero算法的实现。五子棋比围棋或国际象棋简单得多，因此我们可以专注于AlphaZero的训练方案，并在几个小时内在单个PC上获得非常好的AI模型。

参考文献：
1. AlphaZero：通过自我对弈和通用强化学习算法掌握国际象棋和将棋
2. AlphaGo Zero：在没有人类知识的情况下掌握围棋

### 更新2018.2.24：支持使用TensorFlow进行训练！
### 更新2018.1.17：支持使用PyTorch进行训练！

### 训练模型之间的示例游戏
- 每个移动都有400个MCTS播放：
![playout400](https://raw.githubusercontent.com/junxiaosong/AlphaZero_Gomoku/master/playout400.gif)

### 要求
要玩已经训练好的AI模型，只需要：
- Python >= 2.7
- Numpy >= 1.11

要从头开始训练AI模型，还需要以下内容之一：
- Theano >= 0.7 和 Lasagne >= 0.1      
或者
- PyTorch >= 0.2.0    
或者
- TensorFlow

**PS**: 如果你的Theano版本> 0.7，请按照此[问题](https://github.com/aigamedev/scikit-neuralnetwork/issues/235)安装Lasagne，
否则，强制pip将Theano降级到0.7“pip install --upgrade theano == 0.7.0”

如果您想使用其他DL框架训练模型，则只需重写policy_value_net.py。

### 入门指南
要使用提供的模型进行游戏，请从目录中运行以下脚本：
```
python human_play.py  
```
您可以修改human_play.py以尝试不同的提供的模型或纯MCTS。

要从头开始训练AI模型，使用Theano和Lasagne，直接运行：
```
python train.py
```
修改PyTorch或TensorFlow中的文件[train.py](https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/train.py)，即注释掉该行
```
from policy_value_net import PolicyValueNet  # Theano and Lasagne
```
and uncomment the line 
```
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
or
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
```
然后执行：``python train.py``（要在PyTorch中使用GPU，请设置``use_gpu = True``，并在policy_value_net_pytorch.py中的train_step函数中使用``return loss.item（），entropy.item（）``，如果您的pytorch版本大于0.5）

模型（best_policy.model和current_policy.model）将在每个更新后保存几个（默认为50）。

**注意：**提供的4个模型是使用Theano / Lasagne训练的，要在PyTorch中使用它们，请参阅[issue 5]（https://github.com/junxiaosong/AlphaZero_Gomoku/issues/5）。

**培训提示：**
1.从6 * 6板和4个连续开始是很好的。对于这种情况，我们可以在大约2小时内通过500〜1000个自我玩游戏获得合理好的模型。
2.对于8 * 8板和5个连续的情况，可能需要2000〜3000个自我玩游戏才能获得良好的模型，并且在单个PC上可能需要大约2天。

###更多阅读
我在中文中描述了一些实现细节的文章：[https://zhuanlan.zhihu.com/p/32089487]（https://zhuanlan.zhihu.com/p/32089487）
