import pandas as pd
import numpy as np
import torch
import os
import random
import torch.nn as nn
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score,mean_absolute_error,mean_squared_error


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    # Collect the probability of the target word and take the negative logarithm
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    # Only keep the part with a value of 1 in the mask and find the mean
    loss = crossEntropy.masked_select(mask).mean()
    # loss = loss.to(DEVICE)
    return loss, nTotal.item()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Metrics():
    def __init__(self, threshold=0.5):
        super(Metrics, self).__init__()   # 父类属性用父类方法来进行初始化。
        self.threshold = threshold
        self.history_auc = []
        self.history_acc = []
        self.history_F1 = []
        self.history_recall = []
        self.avg_auc = 0.
        self.avg_acc = 0.
        self.avg_F1 = 0.
        self.avg_recall = 0.
        self.avg_mae = 0.
        self.avg_rmse = 0.
        self.y_proba_in_epoch = []
        self.y_true_in_epoch = []
        self.y_pred = []

    def step(self, y_proba, y_true, input_type='list'):   # y_proba=out（即预测的概率，0到1之间的数）, y_true=seq_y（即真实标签值1或0）
        if input_type != 'list':
            print("ERROR")
        self.y_proba_in_epoch += y_proba    # 这里的 += 的意思： 就是赋值 ，把y_proba 赋值给空list[] 。
        self.y_true_in_epoch += y_true  # 其实就是 传参

        # print("y_proba:", y_proba)
        # print("y_true:", y_true)
        # print("y_proba_in_epoch:",self.y_proba_in_epoch)
        # print("y_true_in_epoch:", self.y_true_in_epoch)
        # print(self.y_proba_in_epoch == y_proba)  # 返回值 True
        # print(self.y_true_in_epoch == y_true)  # 返回值 True
        # self.y_pred = list(map(lambda x: 1 if x > self.threshold else 0, self.y_proba_in_epoch)) 预测值大于0，5为1，否则为0.
        # print(self.y_pred)
        # print(self.y_proba_in_epoch)

    def compute(self):  # 计算 各类 评价 指标
        self.y_pred = list(map(lambda x: 1 if x > self.threshold else 0, self.y_proba_in_epoch))  # y_pred 值[0,1]
        # print(self.y_pred)
        # print(self.y_proba_in_epoch)
        self.avg_acc = precision_score(self.y_true_in_epoch, self.y_pred)
        self.avg_auc = roc_auc_score(self.y_true_in_epoch, self.y_proba_in_epoch)  # y_proba_in_epoch 值 [0~1]
        self.avg_F1 = f1_score(self.y_true_in_epoch, self.y_pred)   # 真实标签和预测标签的评价指标
        self.avg_recall = recall_score(self.y_true_in_epoch, self.y_pred)
        self.avg_mae = mean_absolute_error(self.y_true_in_epoch, self.y_pred)
        self.avg_rmse = mean_squared_error(self.y_true_in_epoch, self.y_pred) ** 0.5
        # roc_auc_score：返回值是auc； auc就是曲线下面积，这个数值越高，则分类器越优秀

    def next_epoch(self):  # 存储self.avg_auc ， self.avg_acc 等
        self.history_auc.append(self.avg_auc)
        self.history_acc.append(self.avg_acc)
        self.history_F1.append(self.avg_F1)
        self.history_recall.append(self.avg_recall)
        self.avg_auc = 0.
        self.avg_acc = 0.
        self.avg_F1 = 0.
        self.avg_recall = 0.
        self.y_proba_in_epoch = []
        self.y_true_in_epoch = []
        self.y_pred = []


class LossFunc(nn.Module):   # 没用到
    def __init__(self):
        super(LossFunc, self).__init__()
        loss = torch.Tensor([0.0])

    def forward(self, preds, seq_y, mask):
        preds = preds * mask
        seq_y = seq_y * mask

        pass
